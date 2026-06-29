"""Supervised fine-tuning for BackTalk's already-reversed dialogue data."""
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
from tqdm import tqdm

from data import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_tokenizer, reverse_text
from model import GPT, GPTConfig, config_to_dict


IGNORE_INDEX = -100


@dataclass
class SFTPrepareConfig:
    src: str = "data/sft_reversed.jsonl"
    tokenizer: str = "tokenizers/backtalk-tokenizer/tokenizer.json"
    out_dir: str = "data/prepared/backtalk-sft-ctx2048"
    block_size: int = 2048
    val_frac: float = 0.02
    seed: int = 1337


@dataclass
class SFTConfig:
    data_dir: str = "data/prepared/backtalk-sft-ctx2048"
    tokenizer: str = "tokenizers/backtalk-tokenizer/tokenizer.json"
    base_checkpoint: str = "checkpoints/backtalk-base/ckpt_best.pt"
    out_dir: str = "checkpoints/backtalk-sft"
    block_size: int = 2048
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 4000
    lr: float = 2e-5
    min_lr: float = 2e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay_steps: int = 4000
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.005
    seed: int = 1337
    device: str = "auto"
    precision: str = "bf16"
    resume: bool = False


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def _write_split(out_dir, split, sequences, assistant_starts):
    token_path = os.path.join(out_dir, f"{split}.bin")
    offset_path = os.path.join(out_dir, f"{split}_offsets.npy")
    start_path = os.path.join(out_dir, f"{split}_assistant_starts.npy")
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    with open(token_path, "wb") as f:
        for i, sequence in enumerate(sequences):
            np.asarray(sequence, dtype=np.uint32).tofile(f)
            offsets[i + 1] = offsets[i] + len(sequence)
    np.save(offset_path, offsets)
    np.save(start_path, np.asarray(assistant_starts, dtype=np.uint16))
    return {
        "tokens": token_path,
        "offsets": offset_path,
        "assistant_starts": start_path,
        "examples": len(sequences),
        "token_count": int(offsets[-1]),
    }


def prepare_sft(config: SFTPrepareConfig, force=False):
    os.makedirs(config.out_dir, exist_ok=True)
    meta_path = os.path.join(config.out_dir, "meta.json")
    expected = [
        meta_path,
        *[
            os.path.join(config.out_dir, f"{split}{suffix}")
            for split in ("train", "val")
            for suffix in (".bin", "_offsets.npy", "_assistant_starts.npy")
        ],
    ]
    if not force and all(os.path.exists(path) for path in expected):
        return json.load(open(meta_path, encoding="utf-8"))

    tokenizer = load_tokenizer(config.tokenizer)
    ids = {
        "pad": tokenizer.token_to_id(PAD_TOKEN),
        "bos": tokenizer.token_to_id(BOS_TOKEN),
        "eos": tokenizer.token_to_id(EOS_TOKEN),
    }
    if any(value is None for value in ids.values()):
        raise ValueError("tokenizer must contain pad, bos, and eos special tokens")

    splits = {
        "train": {"sequences": [], "starts": []},
        "val": {"sequences": [], "starts": []},
    }
    max_sequence_tokens = config.block_size + 1
    rows = truncated = invalid = 0
    max_tokens = 0
    total_supervised = 0
    t0 = time.time()
    total_bytes = os.path.getsize(config.src)

    with open(config.src, "rb") as f, tqdm(
        total=total_bytes,
        desc="prepare SFT",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        postfix={"out": config.out_dir},
    ) as bar:
        for raw in f:
            bar.update(len(raw))
            if not raw.strip():
                continue
            obj = json.loads(raw)
            user = obj.get("user")
            assistant = obj.get("assistant")
            if not isinstance(user, str) or not user or not isinstance(assistant, str) or not assistant:
                invalid += 1
                continue

            user_ids = tokenizer.encode(user).ids
            assistant_ids = tokenizer.encode(assistant).ids
            available = max_sequence_tokens - 4
            if len(assistant_ids) > available:
                assistant_ids = assistant_ids[:available]
                user_ids = []
                truncated += 1
            elif len(user_ids) + len(assistant_ids) > available:
                user_ids = user_ids[-(available - len(assistant_ids)):]
                truncated += 1

            sequence = (
                [ids["bos"]] + user_ids + [ids["eos"], ids["bos"]]
                + assistant_ids + [ids["eos"]]
            )
            assistant_start = len(user_ids) + 3
            digest = hashlib.blake2b(raw, digest_size=8, person=b"BackTalk").digest()
            bucket = int.from_bytes(digest, "big") / 2**64
            split = "val" if bucket < config.val_frac else "train"
            splits[split]["sequences"].append(sequence)
            splits[split]["starts"].append(assistant_start)
            rows += 1
            max_tokens = max(max_tokens, len(sequence))
            total_supervised += len(assistant_ids) + 1
            if rows % 1000 == 0:
                bar.set_postfix({"rows": f"{rows:,}", "out": config.out_dir}, refresh=False)

    if not splits["train"]["sequences"] or not splits["val"]["sequences"]:
        raise ValueError("SFT preparation produced an empty train or validation split")

    result = {}
    for split, values in splits.items():
        result[split] = _write_split(
            config.out_dir, split, values["sequences"], values["starts"]
        )
    meta = {
        **asdict(config),
        "source_sha256": hashlib.sha256(open(config.src, "rb").read()).hexdigest(),
        "rows": rows,
        "invalid_rows": invalid,
        "truncated_rows": truncated,
        "max_sequence_tokens": max_tokens,
        "supervised_tokens": total_supervised,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": ids,
        "splits": result,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    _atomic_json(meta_path, meta)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


class SFTDataset:
    def __init__(self, data_dir, split, pad_id, block_size):
        self.tokens = np.memmap(
            os.path.join(data_dir, f"{split}.bin"), dtype=np.uint32, mode="r"
        )
        self.offsets = np.load(os.path.join(data_dir, f"{split}_offsets.npy"))
        self.assistant_starts = np.load(
            os.path.join(data_dir, f"{split}_assistant_starts.npy")
        )
        self.pad_id = int(pad_id)
        self.block_size = int(block_size)
        self.lengths = np.diff(self.offsets)
        self.order = np.argsort(self.lengths)

    def __len__(self):
        return len(self.assistant_starts)

    def batch(self, batch_size, device, generator=None):
        max_start = max(1, len(self.order) - batch_size)
        start = int(torch.randint(max_start, (1,), generator=generator).item())
        indices = self.order[start:start + batch_size]
        if len(indices) < batch_size:
            indices = np.resize(indices, batch_size)
        sequences = [
            np.asarray(self.tokens[self.offsets[i]:self.offsets[i + 1]], dtype=np.int64)
            for i in indices
        ]
        width = min(self.block_size + 1, max(len(sequence) for sequence in sequences))
        x = torch.full((batch_size, width - 1), self.pad_id, dtype=torch.long)
        y = torch.full((batch_size, width - 1), IGNORE_INDEX, dtype=torch.long)
        input_tokens = supervised_tokens = 0
        for row, (index, sequence) in enumerate(zip(indices, sequences)):
            sequence = sequence[:width]
            usable = len(sequence) - 1
            x[row, :usable] = torch.from_numpy(sequence[:-1].copy())
            labels = torch.from_numpy(sequence[1:].copy())
            first_target = max(0, int(self.assistant_starts[index]) - 1)
            y[row, first_target:usable] = labels[first_target:]
            input_tokens += usable
            supervised_tokens += max(0, usable - first_target)
        return x.to(device), y.to(device), input_tokens, supervised_tokens


def _device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast(device, precision):
    if device == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device == "cuda" and precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _device_info(device):
    parts = [f"device={device}", f"torch={torch.__version__}", f"platform={platform.machine()}"]
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        parts.extend([
            f"cuda={torch.cuda.get_device_name(0)}",
            f"vram={props.total_memory / 1024**3:.1f}GB",
        ])
    return " | ".join(parts)


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _lr(step, cfg):
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.lr_decay_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.lr_decay_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save(
    path, model, optimizer, model_cfg, cfg, step,
    best_val, plateau_val, stale_evals, val_loss,
):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_to_dict(model_cfg),
        "sft_config": asdict(cfg),
        "step": step,
        "best_val": best_val,
        "plateau_val": plateau_val,
        "stale_evals": stale_evals,
        "val_loss": val_loss,
        "tokenizer": cfg.tokenizer,
        "kind": "backtalk-sft",
    }, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def _evaluate(model, datasets, cfg, device):
    model.eval()
    results = {}
    generator = torch.Generator().manual_seed(cfg.seed + 991)
    for split, dataset in datasets.items():
        losses = []
        for _ in range(cfg.eval_batches):
            x, y, _, _ = dataset.batch(cfg.batch_size, device, generator)
            with _autocast(device, cfg.precision):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def run_sft(cfg: SFTConfig):
    device = _device(cfg.device)
    if device == "cuda":
        if not torch.cuda.is_bf16_supported() and cfg.precision == "bf16":
            raise RuntimeError("requested BF16 but CUDA device does not support it")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(cfg.data_dir, "meta.json"), encoding="utf-8"))
    pad_id = int(meta["special_tokens"]["pad"])
    datasets = {
        split: SFTDataset(cfg.data_dir, split, pad_id, cfg.block_size)
        for split in ("train", "val")
    }
    latest_path = os.path.join(cfg.out_dir, "latest.pt")
    best_path = os.path.join(cfg.out_dir, "best.pt")
    start_step = stale_evals = 0
    best_val = float("inf")
    plateau_val = float("inf")

    checkpoint_path = latest_path if cfg.resume else cfg.base_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = GPTConfig(**checkpoint["config"])
    model_cfg.gradient_checkpointing = False
    if model_cfg.block_size != cfg.block_size:
        raise ValueError(
            f"checkpoint block_size={model_cfg.block_size}, SFT block_size={cfg.block_size}"
        )
    model = GPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    if cfg.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to_device(optimizer, device)
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_val", best_val))
        plateau_val = float(checkpoint.get("plateau_val", best_val))
        stale_evals = int(checkpoint.get("stale_evals", 0))

    print(_device_info(device), flush=True)
    print(
        f"SFT | base={checkpoint_path} | params={model.num_params():,} "
        f"| train={len(datasets['train']):,} val={len(datasets['val']):,} "
        f"| batch={cfg.batch_size} grad_accum={cfg.gradient_accumulation_steps} "
        f"| precision={cfg.precision} out={cfg.out_dir}",
        flush=True,
    )
    model.train()
    started = interval_started = time.time()
    interval_input = interval_supervised = interval_examples = 0
    running = None
    stop_reason = "max_steps"

    for step in range(start_step + 1, cfg.max_steps + 1):
        lr = _lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            x, y, input_tokens, supervised_tokens = datasets["train"].batch(
                cfg.batch_size, device
            )
            with _autocast(device, cfg.precision):
                _, loss = model(x, y)
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.item())
            interval_input += input_tokens
            interval_supervised += supervised_tokens
            interval_examples += cfg.batch_size
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_value = step_loss / cfg.gradient_accumulation_steps
        running = loss_value if running is None else 0.9 * running + 0.1 * loss_value

        if step == 1 or step % cfg.log_interval == 0:
            now = time.time()
            elapsed = now - started
            dt = max(1e-6, now - interval_started)
            avg_step = elapsed / max(1, step - start_step)
            eta = avg_step * max(0, cfg.max_steps - step)
            epochs = (
                (step - start_step) * cfg.batch_size * cfg.gradient_accumulation_steps
                / len(datasets["train"])
            )
            vram = ""
            if device == "cuda":
                vram = f" | vram_alloc={torch.cuda.max_memory_allocated()/1024**3:.1f}GB"
            print(
                f"step {step}/{cfg.max_steps} ({100*step/cfg.max_steps:.2f}%) "
                f"| loss {loss_value:.4f} ema {running:.4f} | lr {lr:.2e} "
                f"| examples/sec {interval_examples/dt:,.1f} "
                f"| tokens/sec {interval_input/dt:,.0f} "
                f"| supervised_tokens/sec {interval_supervised/dt:,.0f} "
                f"| epochs {epochs:.3f} | elapsed {_fmt_time(elapsed)} ETA {_fmt_time(eta)} "
                f"| checkpoint {latest_path}{vram} | {_device_info(device)}",
                flush=True,
            )
            interval_started = now
            interval_input = interval_supervised = interval_examples = 0

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = _evaluate(model, datasets, cfg, device)
            improved = losses["val"] < best_val
            significant = losses["val"] < plateau_val - cfg.early_stopping_min_delta
            if improved:
                best_val = losses["val"]
            if significant:
                plateau_val = losses["val"]
                stale_evals = 0
            else:
                stale_evals += 1
            _save(
                latest_path, model, optimizer, model_cfg, cfg, step,
                best_val, plateau_val, stale_evals, losses["val"],
            )
            if improved:
                _save(
                    best_path, model, optimizer, model_cfg, cfg, step,
                    best_val, plateau_val, stale_evals, losses["val"],
                )
            print(
                f"eval step {step}/{cfg.max_steps} | train_loss {losses['train']:.4f} "
                f"| val_loss {losses['val']:.4f} | best_val {best_val:.4f} "
                f"| stale {stale_evals}/{cfg.early_stopping_patience} "
                f"| best_checkpoint {best_path}",
                flush=True,
            )
            if stale_evals >= cfg.early_stopping_patience:
                stop_reason = f"early_stopping patience={cfg.early_stopping_patience}"
                break

    print(
        f"done | reason={stop_reason} | best_val={best_val:.4f} "
        f"| elapsed={_fmt_time(time.time()-started)} | best={best_path} latest={latest_path}",
        flush=True,
    )


@torch.no_grad()
def sample_sft(checkpoint_path, tokenizer_path, prompt, device="auto",
               max_new_tokens=256, temperature=0.7, top_k=50, seed=1337):
    device = _device(device)
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**checkpoint["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path or checkpoint.get("tokenizer"))
    bos = tokenizer.token_to_id(BOS_TOKEN)
    eos = tokenizer.token_to_id(EOS_TOKEN)
    reversed_prompt = reverse_text(prompt)
    prefix = [bos] + tokenizer.encode(reversed_prompt).ids + [eos, bos]
    idx = torch.tensor([prefix[-cfg.block_size:]], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -cfg.block_size:])
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        if next_id == eos:
            break
        generated.append(next_id)
        idx = torch.cat([
            idx, torch.tensor([[next_id]], dtype=torch.long, device=device)
        ], dim=1)
    reversed_answer = tokenizer.decode(generated, skip_special_tokens=True)
    return reversed_answer, reverse_text(reversed_answer)


def load_sft_config(path):
    values = json.load(open(path, encoding="utf-8"))
    defaults = asdict(SFTConfig())
    defaults.update({key: value for key, value in values.items() if key in defaults})
    return SFTConfig(**defaults)
