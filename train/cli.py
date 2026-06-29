"""BackTalk pretraining CLI."""
import argparse
import json
import os
from dataclasses import asdict

from data import PrepareConfig, TokenizerTrainConfig, prepare_dataset, train_tokenizer
from infer import load_model, generate
from data import reverse_text
from train import device_info, set_seed
from train import TrainConfig, load_json_config, merge_config, run_training
from sft import (
    SFTConfig,
    SFTPrepareConfig,
    load_sft_config,
    prepare_sft,
    run_sft,
    sample_sft,
)

DEFAULT_CONFIG = "configs/pretrain_cuda.json"


def _add_common_train_args(p):
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_iters", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--lr_decay_steps", type=int, default=None)
    p.add_argument("--target_train_loss", type=float, default=None)
    p.add_argument("--target_val_loss", type=float, default=None)
    p.add_argument("--device", default=None)


def _train_cfg(args, resume=False, smoke=False):
    cfg = TrainConfig()
    if args.config and os.path.exists(args.config):
        cfg = merge_config(cfg, load_json_config(args.config))
    overrides = {
        key: getattr(args, key)
        for key in ("data_dir", "tokenizer", "out_dir", "max_steps",
                    "eval_interval", "eval_iters", "log_interval",
                    "lr", "min_lr", "warmup_steps", "lr_decay_steps",
                    "target_train_loss", "target_val_loss", "device")
        if getattr(args, key, None) is not None
    }
    cfg = merge_config(cfg, overrides)
    if smoke:
        cfg = merge_config(cfg, {
            "out_dir": args.out_dir or "checkpoints/backtalk-smoke",
            "max_steps": args.max_steps or 50,
            "eval_interval": args.eval_interval or 10,
            "eval_iters": args.eval_iters or 2,
            "log_interval": args.log_interval or 1,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 256,
            "gradient_accumulation_steps": 1,
        })
    cfg.resume = resume
    return cfg


def cmd_train_tokenizer(args):
    cfg = TokenizerTrainConfig(
        src=args.src,
        out_dir=args.out_dir,
        text_field=args.text_field,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_docs=args.limit_docs,
    )
    path, meta = train_tokenizer(cfg)
    print(f"tokenizer={path}")
    print(json.dumps(meta, indent=2))


def cmd_prepare(args):
    cfg = PrepareConfig(
        src=args.src,
        tokenizer=args.tokenizer,
        out_dir=args.out_dir,
        text_field=args.text_field,
        val_frac=args.val_frac,
        context_length=args.context_length,
        seed=args.seed,
        limit_docs=args.limit_docs,
        dtype=args.dtype,
    )
    paths, meta = prepare_dataset(cfg, force=args.force)
    print(json.dumps({"paths": paths, "meta": meta}, indent=2))


def cmd_smoke(args):
    run_training(_train_cfg(args, resume=False, smoke=True))


def cmd_pretrain(args):
    run_training(_train_cfg(args, resume=False, smoke=False))


def cmd_resume(args):
    run_training(_train_cfg(args, resume=True, smoke=False))


def cmd_sample(args):
    model, cfg, tokenizer, device = load_model(args.ckpt, args.tokenizer, args.device)
    if args.seed is not None:
        set_seed(args.seed, device)
    prompt = args.prompt or ""
    model_prompt = reverse_text(prompt) if args.reverse_prompt else prompt
    generated = generate(
        model, cfg, tokenizer, device, model_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(device_info(device))
    print("=" * 60)
    print("MODEL PROMPT")
    print(model_prompt)
    print("=" * 60)
    print("GENERATED REVERSED TEXT")
    print(generated)
    if args.show_readable:
        print("=" * 60)
        print("GENERATED READABLE TEXT")
        print(reverse_text(generated))


def cmd_prepare_sft(args):
    prepare_sft(SFTPrepareConfig(
        src=args.src,
        tokenizer=args.tokenizer,
        out_dir=args.out_dir,
        block_size=args.block_size,
        val_frac=args.val_frac,
        seed=args.seed,
    ), force=args.force)


def _sft_cfg(args, resume=False):
    cfg = load_sft_config(args.config) if args.config else SFTConfig()
    values = vars(args)
    for key in asdict(cfg):
        value = values.get(key)
        if value is not None:
            setattr(cfg, key, value)
    cfg.resume = resume
    return cfg


def cmd_sft(args):
    run_sft(_sft_cfg(args, resume=False))


def cmd_resume_sft(args):
    run_sft(_sft_cfg(args, resume=True))


def cmd_sample_sft(args):
    reversed_answer, answer = sample_sft(
        args.ckpt, args.tokenizer, args.prompt, args.device,
        args.max_new_tokens, args.temperature, args.top_k, args.seed,
    )
    print("REVERSED")
    print(reversed_answer)
    print("READABLE")
    print(answer)


def _add_sft_train_args(parser):
    parser.add_argument("--config", default="configs/sft_cuda.json")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", default=None)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    tok = sub.add_parser("train-tokenizer", help="train a fresh BPE tokenizer")
    tok.add_argument("--src", default="data/corpus/backtalk_reversed.jsonl")
    tok.add_argument("--out_dir", default="tokenizers/backtalk-tokenizer")
    tok.add_argument("--text_field", default="text")
    tok.add_argument("--vocab_size", type=int, default=32768)
    tok.add_argument("--min_frequency", type=int, default=2)
    tok.add_argument("--limit_docs", type=int, default=None)
    tok.set_defaults(func=cmd_train_tokenizer)

    prep = sub.add_parser("prepare", help="encode JSONL corpus into train/val memmaps")
    prep.add_argument("--src", default="data/corpus/backtalk_reversed.jsonl")
    prep.add_argument("--tokenizer", default="tokenizers/backtalk-tokenizer/tokenizer.json")
    prep.add_argument("--out_dir", default="data/prepared/backtalk-ctx2048")
    prep.add_argument("--text_field", default="text")
    prep.add_argument("--val_frac", type=float, default=0.005)
    prep.add_argument("--context_length", type=int, default=2048)
    prep.add_argument("--seed", type=int, default=1337)
    prep.add_argument("--limit_docs", type=int, default=None)
    prep.add_argument("--dtype", default="uint32", choices=["uint16", "uint32"])
    prep.add_argument("--force", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    smoke = sub.add_parser("smoke-test", help="run a tiny training smoke test")
    _add_common_train_args(smoke)
    smoke.set_defaults(func=cmd_smoke)

    pretrain = sub.add_parser("pretrain", help="run full pretraining")
    _add_common_train_args(pretrain)
    pretrain.set_defaults(func=cmd_pretrain)

    resume = sub.add_parser("resume", help="resume full pretraining from ckpt.pt")
    _add_common_train_args(resume)
    resume.set_defaults(func=cmd_resume)

    sample = sub.add_parser("sample", help="sample from a checkpoint")
    sample.add_argument("--ckpt", default="checkpoints/backtalk/ckpt_best.pt")
    sample.add_argument("--tokenizer", default="tokenizers/backtalk-tokenizer/tokenizer.json")
    sample.add_argument("--device", default="auto")
    sample.add_argument("--prompt", default="")
    sample.add_argument("--reverse-prompt", action="store_true")
    sample.add_argument("--show-readable", action="store_true")
    sample.add_argument("--max_new_tokens", type=int, default=256)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top_k", type=int, default=50)
    sample.add_argument("--seed", type=int, default=None)
    sample.set_defaults(func=cmd_sample)

    sft_prep = sub.add_parser("prepare-sft", help="prepare reversed user/assistant SFT data")
    sft_prep.add_argument("--src", default="data/sft_reversed.jsonl")
    sft_prep.add_argument("--tokenizer", default="tokenizers/backtalk-tokenizer/tokenizer.json")
    sft_prep.add_argument("--out_dir", default="data/prepared/backtalk-sft-ctx2048")
    sft_prep.add_argument("--block_size", type=int, default=2048)
    sft_prep.add_argument("--val_frac", type=float, default=0.02)
    sft_prep.add_argument("--seed", type=int, default=1337)
    sft_prep.add_argument("--force", action="store_true")
    sft_prep.set_defaults(func=cmd_prepare_sft)

    sft_train = sub.add_parser("sft", help="fine-tune from a pretraining checkpoint")
    _add_sft_train_args(sft_train)
    sft_train.set_defaults(func=cmd_sft)

    sft_resume = sub.add_parser("resume-sft", help="resume SFT from latest.pt")
    _add_sft_train_args(sft_resume)
    sft_resume.set_defaults(func=cmd_resume_sft)

    sft_sample = sub.add_parser("sample-sft", help="sample readable text from an SFT checkpoint")
    sft_sample.add_argument("--ckpt", default="checkpoints/backtalk-sft/best.pt")
    sft_sample.add_argument("--tokenizer", default="tokenizers/backtalk-tokenizer/tokenizer.json")
    sft_sample.add_argument("--prompt", required=True)
    sft_sample.add_argument("--device", default="auto")
    sft_sample.add_argument("--max_new_tokens", type=int, default=256)
    sft_sample.add_argument("--temperature", type=float, default=0.7)
    sft_sample.add_argument("--top_k", type=int, default=50)
    sft_sample.add_argument("--seed", type=int, default=1337)
    sft_sample.set_defaults(func=cmd_sample_sft)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
