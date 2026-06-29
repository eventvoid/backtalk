"""Tokenizer training and JSONL stream preparation for BackTalk pretraining."""
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
DEFAULT_TEXT_FIELD = "text"

# Compatibility for the old story UI; the corpus pretraining path does not use it.
INPUT_ORDER = ["hero", "setting", "problem", "helper_or_item", "lesson"]


@dataclass
class TokenizerTrainConfig:
    src: str = "data/corpus/backtalk_reversed.jsonl"
    out_dir: str = "tokenizers/backtalk-tokenizer"
    text_field: str = DEFAULT_TEXT_FIELD
    vocab_size: int = 32768
    min_frequency: int = 2
    limit_docs: Optional[int] = None


@dataclass
class PrepareConfig:
    src: str = "data/corpus/backtalk_reversed.jsonl"
    tokenizer: str = "tokenizers/backtalk-tokenizer/tokenizer.json"
    out_dir: str = "data/prepared/backtalk-ctx2048"
    text_field: str = DEFAULT_TEXT_FIELD
    val_frac: float = 0.005
    context_length: int = 2048
    seed: int = 1337
    limit_docs: Optional[int] = None
    dtype: str = "uint32"


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise SystemExit(
            "The 'tokenizers' package is required. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return Tokenizer, decoders, models, pre_tokenizers, trainers


def iter_jsonl_texts(path: str, text_field: str = DEFAULT_TEXT_FIELD,
                     limit_docs: Optional[int] = None) -> Iterable[str]:
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit_docs is not None and seen >= limit_docs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                continue
            seen += 1
            yield str(obj[text_field])


def iter_jsonl_texts_with_progress(path: str, text_field: str = DEFAULT_TEXT_FIELD,
                                   limit_docs: Optional[int] = None,
                                   desc: str = "corpus",
                                   output_path: Optional[str] = None) -> Iterable[str]:
    """Yield JSONL text values while tqdm reports byte-level corpus progress."""
    total = os.path.getsize(path) if limit_docs is None else None
    seen = 0
    postfix = {"out": output_path} if output_path else None
    with open(path, "rb") as f, tqdm(
        total=total,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=2.0,
        maxinterval=10.0,
        postfix=postfix,
    ) as bar:
        for raw in f:
            bar.update(len(raw))
            if limit_docs is not None and seen >= limit_docs:
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                continue
            seen += 1
            if seen % 1000 == 0:
                bar.set_postfix({"docs": f"{seen:,}", "out": output_path or ""}, refresh=False)
            yield str(obj[text_field])


def reverse_text(s: str) -> str:
    """Simple Unicode code-point reversal for sampling readable text."""
    return s[::-1]


def format_input(inp: dict) -> str:
    return "\n".join(f"{key}: {inp[key]}" for key in INPUT_ORDER)


def train_tokenizer(config: TokenizerTrainConfig):
    """Train a fresh byte-level BPE tokenizer on the full reversed corpus."""
    Tokenizer, decoders, models, pre_tokenizers, trainers = _require_tokenizers()
    os.makedirs(config.out_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    t0 = time.time()
    print(
        f"training tokenizer | src={config.src} | out={config.out_dir} "
        f"| vocab_size={config.vocab_size} | full_corpus={config.limit_docs is None}",
        flush=True,
    )
    tokenizer.train_from_iterator(
        iter_jsonl_texts_with_progress(
            config.src,
            config.text_field,
            config.limit_docs,
            desc="tokenizer corpus",
            output_path=config.out_dir,
        ),
        trainer=trainer,
    )
    tokenizer_path = os.path.join(config.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    try:
        tokenizer.model.save(config.out_dir)
    except Exception:
        pass

    ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    meta = {
        "kind": "byte-level-bpe",
        "src": config.src,
        "text_field": config.text_field,
        "vocab_size_requested": config.vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "min_frequency": config.min_frequency,
        "limit_docs": config.limit_docs,
        "special_tokens": ids,
        "elapsed_sec": round(time.time() - t0, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(config.out_dir, "tokenizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(config.out_dir, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    return tokenizer_path, meta


def load_tokenizer(path: str):
    Tokenizer, *_ = _require_tokenizers()
    return Tokenizer.from_file(path)


def encode_document(tokenizer, text: str):
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer is missing <|bos|> or <|eos|>")
    return [bos_id] + tokenizer.encode(text).ids + [eos_id]


def _dtype(name: str):
    if name == "uint16":
        return np.uint16
    if name == "uint32":
        return np.uint32
    raise ValueError(f"unsupported dtype: {name}")


def _append_ids(path: str, ids, dtype):
    arr = np.asarray(ids, dtype=dtype)
    with open(path, "ab") as f:
        arr.tofile(f)
    return int(arr.size)


def prepare_dataset(config: PrepareConfig, force: bool = False):
    """Encode JSONL text documents into train/val memmap streams."""
    os.makedirs(config.out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(config.out_dir, "train.bin"),
        "val": os.path.join(config.out_dir, "val.bin"),
        "meta": os.path.join(config.out_dir, "meta.json"),
    }
    if not force and all(os.path.exists(p) for p in paths.values()):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            return paths, json.load(f)

    for key in ("train", "val"):
        if os.path.exists(paths[key]):
            os.remove(paths[key])

    tokenizer = load_tokenizer(config.tokenizer)
    dtype = _dtype(config.dtype)
    rng = random.Random(config.seed)
    t0 = time.time()

    docs = train_docs = val_docs = no_text = 0
    train_tokens = val_tokens = chars = words = 0
    max_doc_tokens = 0
    batch_size = 512
    pending_texts = []
    pending_is_val = []
    total = os.path.getsize(config.src) if config.limit_docs is None else None
    print(
        f"preparing dataset | src={config.src} | tokenizer={config.tokenizer} "
        f"| out={config.out_dir} | ctx={config.context_length}",
        flush=True,
    )

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer is missing <|bos|> or <|eos|>")

    def flush_batch():
        nonlocal train_tokens, val_tokens, max_doc_tokens
        if not pending_texts:
            return
        encodings = tokenizer.encode_batch(pending_texts)
        train_ids = []
        val_ids = []
        for encoding, is_val in zip(encodings, pending_is_val):
            ids = [bos_id] + encoding.ids + [eos_id]
            max_doc_tokens = max(max_doc_tokens, len(ids))
            if is_val:
                val_ids.extend(ids)
            else:
                train_ids.extend(ids)
        if train_ids:
            train_tokens += _append_ids(paths["train"], train_ids, dtype)
        if val_ids:
            val_tokens += _append_ids(paths["val"], val_ids, dtype)
        pending_texts.clear()
        pending_is_val.clear()

    with open(config.src, "rb") as f, tqdm(
        total=total,
        desc="prepare corpus",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=2.0,
        maxinterval=10.0,
        postfix={"out": config.out_dir},
    ) as bar:
        for raw in f:
            bar.update(len(raw))
            if config.limit_docs is not None and docs >= config.limit_docs:
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if config.text_field not in obj:
                no_text += 1
                continue
            docs += 1
            text = str(obj[config.text_field])
            chars += len(text)
            words += len(text.split())
            is_val = rng.random() < config.val_frac
            pending_texts.append(text)
            pending_is_val.append(is_val)
            if is_val:
                val_docs += 1
            else:
                train_docs += 1
            if len(pending_texts) >= batch_size:
                flush_batch()
            if docs % 500 == 0:
                bar.set_postfix({
                    "docs": f"{docs:,}",
                    "tokens": f"{train_tokens + val_tokens:,}",
                    "out": config.out_dir,
                }, refresh=False)
        flush_batch()

    meta = {
        **asdict(config),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "n_docs": docs,
        "n_train_docs": train_docs,
        "n_val_docs": val_docs,
        "docs_without_text": no_text,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "chars": chars,
        "words": words,
        "max_doc_tokens": max_doc_tokens,
        "elapsed_sec": round(time.time() - t0, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return paths, meta


class BinDataset:
    def __init__(self, train_path: str, val_path: str, block_size: int, dtype: str = "uint32"):
        self.train = np.memmap(train_path, dtype=_dtype(dtype), mode="r")
        self.val = np.memmap(val_path, dtype=_dtype(dtype), mode="r")
        self.block_size = int(block_size)
        if len(self.train) <= self.block_size + 1:
            raise ValueError("train split is too small for block_size")
        if len(self.val) <= self.block_size + 1:
            raise ValueError("val split is too small for block_size")

    def get_batch(self, split: str, batch_size: int, device: str):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(data[int(i):int(i) + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[int(i) + 1:int(i) + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)
