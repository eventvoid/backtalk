"""Byte-level tokenization + data preparation for the reversed-story model.

Tokenizer: raw UTF-8 bytes (ids 0..255) plus a single EOS token (id 256),
so the vocabulary is exactly 257 tokens (no PAD token: stream sampling never
pads).
This needs no vocabulary scan, handles any UTF-8, and is a natural fit for
character-reversed text (we encode the already-reversed string to bytes).

Data layout follows nanoGPT: every training_prompt is encoded to bytes,
followed by EOS, and all examples are concatenated into one uint16 stream per
split. Training samples random fixed-length blocks from the stream.
"""
import json
import os
import random

import numpy as np
import torch

# Strict input field order (must match data preparation).
INPUT_ORDER = ["hero", "setting", "problem", "helper_or_item", "lesson"]

# Vocabulary: 256 raw byte ids (0..255) + 1 EOS token (256) = 257 total.
# Stream training samples contiguous blocks, so no padding/PAD token is needed.
EOS_ID = 256
VOCAB_SIZE = 257


# --------------------------------------------------------------------------- #
# Text helpers (shared by training data prep and inference)
# --------------------------------------------------------------------------- #
def format_input(inp: dict) -> str:
    """Render the input dict to the canonical plain-text form, strict order."""
    return "\n".join(f"{key}: {inp[key]}" for key in INPUT_ORDER)


def reverse_text(s: str) -> str:
    """Character-wise reversal (matches dataset preparation)."""
    return s[::-1]


def build_story_prefix(reversed_input: str) -> str:
    """The conditioning prefix the model is asked to continue at inference."""
    return f"<|input|>\n{reversed_input}\n\n<|story|>\n"


def encode_str(s: str):
    """String -> list of byte ids."""
    return list(s.encode("utf-8"))


def decode_ids(ids) -> str:
    """List/iterable of ids -> string (special tokens dropped)."""
    bs = bytes(int(b) for b in ids if int(b) < 256)
    return bs.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Dataset preparation (cached binary streams)
# --------------------------------------------------------------------------- #
def _iter_prompts(jsonl_path, max_examples=None):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["training_prompt"]


def _write_stream(prompts, out_path):
    """Encode prompts to bytes + EOS and write a uint16 .bin, low-memory."""
    n_tokens = 0
    buf = []
    with open(out_path, "wb") as fout:
        for p in prompts:
            arr = np.frombuffer(p.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
            buf.append(arr)
            buf.append(np.array([EOS_ID], dtype=np.uint16))
            if len(buf) >= 8192:
                chunk = np.concatenate(buf)
                chunk.tofile(fout)
                n_tokens += chunk.size
                buf = []
        if buf:
            chunk = np.concatenate(buf)
            chunk.tofile(fout)
            n_tokens += chunk.size
    return n_tokens


def prepare(jsonl_path, cache_dir, val_frac=0.01, max_examples=None, seed=1337, force=False):
    """Build (or load cached) train/val byte streams. Returns (paths, meta)."""
    tag = "full" if max_examples is None else f"sub{max_examples}"
    paths = {
        "train": os.path.join(cache_dir, f"train_{tag}.bin"),
        "val": os.path.join(cache_dir, f"val_{tag}.bin"),
    }
    meta_path = os.path.join(cache_dir, f"meta_{tag}.json")

    if not force and os.path.exists(meta_path) and all(os.path.exists(p) for p in paths.values()):
        with open(meta_path, "r", encoding="utf-8") as f:
            return paths, json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    prompts = list(_iter_prompts(jsonl_path, max_examples=max_examples))
    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_val = max(1, int(len(prompts) * val_frac))
    val_prompts = prompts[:n_val]
    train_prompts = prompts[n_val:]

    train_tokens = _write_stream(train_prompts, paths["train"])
    val_tokens = _write_stream(val_prompts, paths["val"])

    meta = {
        "tag": tag,
        "vocab_size": VOCAB_SIZE,
        "eos_id": EOS_ID,
        "n_train_examples": len(train_prompts),
        "n_val_examples": len(val_prompts),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "val_frac": val_frac,
        "seed": seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return paths, meta


def get_batch(bin_path, block_size, batch_size, device):
    """Sample a random batch of (x, y) blocks from a memmapped uint16 stream."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
