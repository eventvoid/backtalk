"""Train the byte-level reversed-story GPT (MPS-friendly).

Examples
--------
Smoke test (small subset, tiny model, few iters):
    python train/train.py --max_examples 2000 --n_layer 4 --n_embd 128 \
        --n_head 4 --block_size 256 --batch_size 32 --max_iters 300 \
        --eval_interval 50 --out_dir checkpoints/smoke

Full training (whole dataset, default ~11M model):
    python train/train.py --out_dir checkpoints/full --max_iters 20000

Resume:
    python train/train.py --out_dir checkpoints/full --resume
"""
import argparse
import math
import os
import time

import torch

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from data import prepare, get_batch, VOCAB_SIZE
from model import GPT, GPTConfig, config_to_dict


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.lr * (it + 1) / max(1, args.warmup_iters)
    if it > args.lr_decay_iters:
        return args.min_lr
    ratio = (it - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, paths, args, device):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = get_batch(paths[split], args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/stories_reversed_prompts.jsonl")
    p.add_argument("--cache_dir", default="data/prepared")
    p.add_argument("--out_dir", default="checkpoints/full")
    p.add_argument("--max_examples", type=int, default=None, help="limit #examples (smoke); None=full")
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1337)
    # model
    p.add_argument("--block_size", type=int, default=640)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    # optim
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--lr_decay_iters", type=int, default=None, help="default = max_iters")
    # loop / io
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_progress", action="store_true", help="disable the tqdm progress bar (plain log lines)")
    args = p.parse_args()
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")

    paths, meta = prepare(args.data, args.cache_dir, val_frac=args.val_frac,
                          max_examples=args.max_examples, seed=args.seed)
    print(f"data: train_tokens={meta['train_tokens']:,} val_tokens={meta['val_tokens']:,} "
          f"(examples train={meta['n_train_examples']:,} val={meta['n_val_examples']:,})")

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, dropout=args.dropout)

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    best_path = os.path.join(args.out_dir, "ckpt_best.pt")
    start_iter = 0
    best_val = float("inf")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, args.beta2), weight_decay=args.weight_decay)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at iter {start_iter} (best_val={best_val:.4f})")
    else:
        model = GPT(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, args.beta2), weight_decay=args.weight_decay)

    n_params = model.num_params()
    print(f"model: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}D block={cfg.block_size} "
          f"params={n_params:,} ({n_params/1e6:.2f}M)")

    def save(path, it, val_loss):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_to_dict(cfg),
            "iter": it,
            "best_val": best_val,
            "val_loss": val_loss,
            "args": vars(args),
        }, path)

    # Progress bar: tqdm gives elapsed, ETA/remaining, rate (it/s or s/it),
    # %, and iter/max for free. `initial=start_iter` makes resume continue from
    # the right place. Logs go through `log()` so eval/checkpoint lines print
    # cleanly above the bar instead of corrupting it.
    use_progress = _HAS_TQDM and not args.no_progress
    pbar = None
    if use_progress:
        pbar = tqdm(total=args.max_iters, initial=start_iter, desc="train",
                    unit="it", dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    def log(msg):
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

    model.train()
    t0 = time.time()
    running = None
    last_eval = ""
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, paths, args, device)
            save(ckpt_path, it, losses["val"])
            saved = "ckpt.pt"
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(best_path, it, losses["val"])
                saved = "ckpt.pt + ckpt_best.pt (new best)"
            last_eval = f"eval@{it} train {losses['train']:.3f} val {losses['val']:.3f}"
            log(f"[eval] iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | saved {saved}")
            if pbar is not None:
                pbar.set_postfix_str(last_eval, refresh=False)

        if it == args.max_iters:
            break

        xb, yb = get_batch(paths["train"], args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        lossf = loss.item()
        running = lossf if running is None else 0.9 * running + 0.1 * lossf

        if pbar is not None:
            pbar.set_postfix_str(
                f"loss {lossf:.3f} ema {running:.3f} lr {lr:.1e}"
                + (f" | {last_eval}" if last_eval else ""),
                refresh=False)
            pbar.update(1)
        elif it % args.log_interval == 0:
            dt = (time.time() - t0) / max(1, args.log_interval)
            t0 = time.time()
            print(f"iter {it}/{args.max_iters}: loss {lossf:.4f} (ema {running:.4f}) "
                  f"| lr {lr:.2e} | {dt*1000:.0f} ms/iter", flush=True)

    if pbar is not None:
        pbar.close()
    print(f"done. best val loss {best_val:.4f}. checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
