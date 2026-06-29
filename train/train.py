"""GPT pretraining for the BackTalk reversed corpus."""
import argparse
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

from data import BinDataset
from model import GPT, GPTConfig, config_to_dict


@dataclass
class TrainConfig:
    data_dir: str = "data/prepared/backtalk-ctx2048"
    tokenizer: str = "tokenizers/backtalk-tokenizer/tokenizer.json"
    out_dir: str = "checkpoints/backtalk"
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_steps: int = 100000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 10
    target_train_loss: float = 0.0
    target_val_loss: float = 0.0
    seed: int = 1337
    device: str = "auto"
    dtype: str = "uint32"
    precision: str = "auto"
    compile_model: bool = False
    resume: bool = False


def pick_device(requested: str):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info(device: str):
    bits = [
        f"device={device}",
        f"torch={torch.__version__}",
        f"mps_available={torch.backends.mps.is_available()}",
        f"mps_built={torch.backends.mps.is_built()}",
        f"platform={platform.machine()}",
    ]
    if torch.cuda.is_available():
        bits.append(f"cuda={torch.cuda.get_device_name(0)}")
    return " | ".join(bits)


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)


def pick_precision(device: str, requested: str):
    if requested != "auto":
        return requested
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def autocast_context(device: str, precision: str):
    if device == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_lr(step: int, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step > cfg.lr_decay_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def fmt_time(seconds: float):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h:02d}h"
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def optimizer_to_device(optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def estimate_loss(model, dataset: BinDataset, cfg: TrainConfig, device: str, precision: str):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = dataset.get_batch(split, cfg.batch_size, device)
            with autocast_context(device, precision):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, model_cfg, train_cfg, step, best_val, val_loss, data_meta):
    model_to_save = getattr(model, "_orig_mod", model)
    torch.save({
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_to_dict(model_cfg),
        "train_config": asdict(train_cfg),
        "iter": step,
        "step": step,
        "best_val": best_val,
        "val_loss": val_loss,
        "data_meta": data_meta,
        "tokenizer": train_cfg.tokenizer,
    }, path)


def load_json_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(base: TrainConfig, values: dict):
    data = asdict(base)
    for key, value in values.items():
        if key in data and value is not None:
            data[key] = value
    return TrainConfig(**data)


def run_training(cfg: TrainConfig):
    device = pick_device(cfg.device)
    precision = pick_precision(device, cfg.precision)
    if cfg.lr_decay_steps > cfg.max_steps:
        cfg.lr_decay_steps = cfg.max_steps
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(cfg.seed, device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    meta_path = os.path.join(cfg.data_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        data_meta = json.load(f)

    dataset = BinDataset(
        os.path.join(cfg.data_dir, "train.bin"),
        os.path.join(cfg.data_dir, "val.bin"),
        block_size=cfg.block_size,
        dtype=data_meta.get("dtype", cfg.dtype),
    )

    model_cfg = GPTConfig(
        vocab_size=int(data_meta["vocab_size"]),
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    best_path = os.path.join(cfg.out_dir, "ckpt_best.pt")
    start_step = 0
    best_val = float("inf")

    if cfg.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_cfg = GPTConfig(**ckpt["config"])
        model = GPT(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(optimizer, device)
        start_step = int(ckpt.get("step", ckpt.get("iter", 0)))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"resumed checkpoint={ckpt_path} step={start_step} best_val={best_val:.4f}", flush=True)
    else:
        model = GPT(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

    if cfg.compile_model:
        model = torch.compile(model)

    n_params = model.num_params()
    print(device_info(device), flush=True)
    print(
        f"model={model_cfg.n_layer}L/{model_cfg.n_head}H/{model_cfg.n_embd}D "
        f"ctx={model_cfg.block_size} vocab={model_cfg.vocab_size} "
        f"params={n_params:,} ({n_params / 1e6:.2f}M) "
        f"micro_batch={cfg.batch_size} grad_accum={cfg.gradient_accumulation_steps} "
        f"precision={precision} compile={cfg.compile_model}",
        flush=True,
    )
    print(
        f"data=train_tokens={data_meta['train_tokens']:,} "
        f"val_tokens={data_meta['val_tokens']:,} data_dir={cfg.data_dir}",
        flush=True,
    )

    model.train()
    train_start = time.time()
    interval_start = train_start
    interval_tokens = 0
    running = None
    last_ckpt = ckpt_path
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps
    total_train_tokens = max(1, int(data_meta["train_tokens"]))
    target_epochs = (cfg.max_steps * tokens_per_step) / total_train_tokens
    print(
        f"schedule=tokens_per_step={tokens_per_step:,} "
        f"target_epochs={target_epochs:.3f} max_steps={cfg.max_steps:,} "
        f"target_train_loss={cfg.target_train_loss or 'off'} "
        f"target_val_loss={cfg.target_val_loss or 'off'}",
        flush=True,
    )

    stop_reason = None
    for step in range(start_step + 1, cfg.max_steps + 1):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_t0 = time.time()
        for _ in range(cfg.gradient_accumulation_steps):
            xb, yb = dataset.get_batch("train", cfg.batch_size, device)
            with autocast_context(device, precision):
                _, loss = model(xb, yb)
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        lossf = step_loss / cfg.gradient_accumulation_steps
        running = lossf if running is None else 0.9 * running + 0.1 * lossf
        interval_tokens += tokens_per_step

        if step % cfg.log_interval == 0 or step == 1:
            now = time.time()
            interval_dt = max(1e-6, now - interval_start)
            tok_s = interval_tokens / interval_dt
            elapsed = now - train_start
            avg_step = elapsed / max(1, step - start_step)
            eta = avg_step * max(0, cfg.max_steps - step)
            pct = 100.0 * step / max(1, cfg.max_steps)
            epochs = (step * tokens_per_step) / total_train_tokens
            print(
                f"step {step}/{cfg.max_steps} ({pct:.2f}%) | loss {lossf:.4f} | ema {running:.4f} "
                f"| lr {lr:.2e} | tokens/step {tokens_per_step:,} | tokens/sec {tok_s:,.0f} "
                f"| epochs {epochs:.3f}/{target_epochs:.3f} | elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)} "
                f"| checkpoint {last_ckpt} | {device_info(device)}",
                flush=True,
            )
            interval_start = now
            interval_tokens = 0

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, dataset, cfg, device, precision)
            val_loss = losses["val"]
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            save_checkpoint(ckpt_path, model, optimizer, model_cfg, cfg, step, best_val, val_loss, data_meta)
            last_ckpt = ckpt_path
            if is_best:
                save_checkpoint(best_path, model, optimizer, model_cfg, cfg, step, best_val, val_loss, data_meta)
                last_ckpt = f"{ckpt_path}, {best_path}"
            print(
                f"eval step {step}/{cfg.max_steps} ({100.0 * step / max(1, cfg.max_steps):.2f}%) "
                f"| train_loss {losses['train']:.4f} "
                f"| val_loss {val_loss:.4f} "
                f"| epochs {(step * tokens_per_step) / total_train_tokens:.3f}/{target_epochs:.3f} "
                f"| checkpoint {last_ckpt}",
                flush=True,
            )
            if cfg.target_val_loss > 0 and val_loss <= cfg.target_val_loss:
                stop_reason = f"target_val_loss reached ({val_loss:.4f} <= {cfg.target_val_loss:.4f})"
                break
            if cfg.target_train_loss > 0 and losses["train"] <= cfg.target_train_loss:
                stop_reason = f"target_train_loss reached ({losses['train']:.4f} <= {cfg.target_train_loss:.4f})"
                break

    suffix = f" | stop_reason {stop_reason}" if stop_reason else ""
    print(f"done | best_val {best_val:.4f} | checkpoint_dir {cfg.out_dir}{suffix}", flush=True)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="JSON config file")
    for field, default in asdict(TrainConfig()).items():
        arg = "--" + field
        if isinstance(default, bool):
            p.add_argument(arg, dest=field, action="store_true")
            p.add_argument("--no_" + field, dest=field, action="store_false")
            p.set_defaults(**{field: None})
        elif isinstance(default, int):
            p.add_argument(arg, type=int, default=None)
        elif isinstance(default, float):
            p.add_argument(arg, type=float, default=None)
        else:
            p.add_argument(arg, default=None)
    return p


def main():
    p = build_arg_parser()
    ns = p.parse_args()
    cfg = TrainConfig()
    if ns.config:
        cfg = merge_config(cfg, load_json_config(ns.config))
    overrides = vars(ns)
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    run_training(cfg)


if __name__ == "__main__":
    main()
