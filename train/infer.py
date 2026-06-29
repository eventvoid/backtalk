"""Sample from a BackTalk reversed-text language model checkpoint."""
import argparse
import sys

import torch

from data import EOS_TOKEN, load_tokenizer, reverse_text
from model import GPT, GPTConfig
from train import device_info, pick_device, set_seed

DEFAULT_CKPT = "checkpoints/backtalk/ckpt_best.pt"
DEFAULT_TOKENIZER = "tokenizers/backtalk-tokenizer/tokenizer.json"


def load_model(ckpt_path=DEFAULT_CKPT, tokenizer_path=None, device="auto"):
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path or ckpt.get("tokenizer") or DEFAULT_TOKENIZER)
    return model, cfg, tokenizer, device


@torch.no_grad()
def generate(model, cfg, tokenizer, device, prompt, max_new_tokens=256,
             temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt).ids
    if len(ids) == 0:
        bos = tokenizer.token_to_id("<|bos|>")
        ids = [bos] if bos is not None else []
    if len(ids) > cfg.block_size:
        ids = ids[-cfg.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and int(next_id.item()) == eos_id:
            break

    generated_ids = idx[0].tolist()[len(ids):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def read_prompt(args):
    if args.prompt is not None:
        return args.prompt
    text = sys.stdin.read()
    return text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--prompt", default=None, help="Prompt text. If omitted, stdin is used.")
    p.add_argument("--reverse-prompt", action="store_true",
                   help="Reverse the prompt before encoding. Use for normal readable prompts.")
    p.add_argument("--show-readable", action="store_true",
                   help="Also reverse generated text back to readable order.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    model, cfg, tokenizer, device = load_model(args.ckpt, args.tokenizer, args.device)
    if args.seed is not None:
        set_seed(args.seed, device)

    prompt = read_prompt(args)
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


if __name__ == "__main__":
    main()
