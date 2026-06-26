"""Inference for the reversed-story model.

Takes normal (un-reversed) input fields, reverses the input text, asks the model
to generate the reversed story, then reverses the story back for readable output.

The core logic lives in `load_model()` and `generate_story()` so it can be reused
(e.g. by the web UI in web/server.py) without duplication.

CLI input can be given via flags or as a 'key: value' block on stdin:
    printf 'hero: dog\\nsetting: snowy field\\nproblem: big storm\\n'\\
'helper_or_item: little boat\\nlesson: sharing\\n' | \\
        python train/infer.py --ckpt checkpoints/full/ckpt_best.pt
"""
import argparse
import sys

import torch

from data import (INPUT_ORDER, EOS_ID, format_input, reverse_text,
                  build_story_prefix, encode_str, decode_ids)
from model import GPT, GPTConfig

DEFAULT_CKPT = "checkpoints/full/ckpt_best.pt"
END_MARKER = "<|end|>"


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed, device):
    """Seed RNGs so generation is reproducible for a given seed + params."""
    torch.manual_seed(seed)
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path=DEFAULT_CKPT, device="auto"):
    """Load a checkpoint and return (model, cfg, device)."""
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, device


def generate_story(model, cfg, device, inp, max_new_tokens=700,
                   temperature=0.8, top_k=40, seed=None):
    """Run the full reverse -> generate -> reverse-back pipeline.

    `inp` is a dict with the INPUT_ORDER keys. Returns a dict with every
    human-inspectable stage of the pipeline.
    """
    if seed is not None:
        set_seed(int(seed), device)

    input_text = format_input(inp)
    reversed_input = reverse_text(input_text)
    reversed_prompt = build_story_prefix(reversed_input)   # what actually goes to the model

    ids = encode_str(reversed_prompt)
    if len(ids) > cfg.block_size:
        ids = ids[-cfg.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k, eos_token=EOS_ID)
    gen_ids = out[0].tolist()[len(ids):]          # only the newly generated part
    reversed_story = decode_ids(gen_ids)

    # The model continues after '<|story|>\n'; trim trailing end marker if present.
    if END_MARKER in reversed_story:
        reversed_story = reversed_story.split(END_MARKER)[0]
    reversed_story = reversed_story.rstrip("\n")

    story = reverse_text(reversed_story)

    return {
        "input_text": input_text,
        "reversed_prompt": reversed_prompt,
        "reversed_story": reversed_story,
        "story": story,
    }


def stream_generation(model, cfg, device, inp, max_new_tokens=600,
                      temperature=0.8, top_k=40, seed=None):
    """Generator over the reverse -> generate -> reverse-back pipeline.

    Yields dicts:
      {"stage": "start", "input_text", "reversed_prompt"}
      {"stage": "chunk", "delta": <new reversed chars>}   (incremental)
      {"stage": "done",  "reversed_story", "story", "n_tokens"}
    Shares all logic with generate_story()'s building blocks.
    """
    if seed is not None:
        set_seed(int(seed), device)

    input_text = format_input(inp)
    reversed_input = reverse_text(input_text)
    reversed_prompt = build_story_prefix(reversed_input)

    ids = encode_str(reversed_prompt)
    if len(ids) > cfg.block_size:
        ids = ids[-cfg.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    yield {"stage": "start", "input_text": input_text, "reversed_prompt": reversed_prompt}

    gen_bytes = []
    sent = 0          # chars of the reversed story already emitted as deltas
    n_tokens = 0
    for tok in model.generate_stream(idx, max_new_tokens, temperature, top_k, EOS_ID):
        if tok >= 256:        # EOS / specials -> stop
            break
        gen_bytes.append(tok)
        n_tokens += 1
        decoded = decode_ids(gen_bytes)
        reached_end = END_MARKER in decoded
        visible = decoded.split(END_MARKER)[0] if reached_end else decoded
        if len(visible) > sent:           # only emit the new tail (handles multi-byte)
            yield {"stage": "chunk", "delta": visible[sent:], "n": n_tokens}
            sent = len(visible)
        if reached_end:
            break

    decoded = decode_ids(gen_bytes)
    if END_MARKER in decoded:
        decoded = decoded.split(END_MARKER)[0]
    reversed_story = decoded.rstrip("\n")
    yield {"stage": "done", "reversed_story": reversed_story,
           "story": reverse_text(reversed_story), "n_tokens": n_tokens}


def parse_block(text):
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip()] = val.strip()
    return fields


def collect_input(args):
    fields = {}
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text.strip():
            fields.update(parse_block(stdin_text))
    for key in INPUT_ORDER:
        v = getattr(args, key)
        if v is not None:
            fields[key] = v
    missing = [k for k in INPUT_ORDER if k not in fields or fields[k] == ""]
    if missing:
        sys.exit(f"missing required input field(s): {', '.join(missing)}")
    return {k: fields[k] for k in INPUT_ORDER}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=700)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    p.add_argument("--show_reversed", action="store_true", help="also print raw reversed model output")
    for key in INPUT_ORDER:
        p.add_argument(f"--{key}", default=None)
    args = p.parse_args()

    model, cfg, device = load_model(args.ckpt, args.device)
    inp = collect_input(args)
    result = generate_story(model, cfg, device, inp,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k,
                           seed=args.seed)

    print("=" * 60)
    print("INPUT")
    print(result["input_text"])
    if args.show_reversed:
        print("=" * 60)
        print("REVERSED STORY (raw model output)")
        print(result["reversed_story"])
    print("=" * 60)
    print("STORY")
    print(result["story"])
    print("=" * 60)


if __name__ == "__main__":
    main()
