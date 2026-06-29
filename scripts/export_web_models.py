"""Export optimizer-free checkpoints with stable, versioned product names.

Each web model lives in checkpoints/<out>/ as model.pt (weights + config) plus a
model.json describing it. Public names are versioned per family (Ask v1/v2,
Stories v1); the server maps the bare ids (backtalk-assistant /
backtalk-storyteller) to the latest version of each family.

    python3 scripts/export_web_models.py            # export missing models
    python3 scripts/export_web_models.py --force     # re-export everything
"""
import argparse
import json
import os

import torch


# Order matters only for readability. `out` is the checkpoint directory; for the
# v1 models it stays at the original location so nothing has to be re-exported.
MODELS = [
    {
        "id": "backtalk-assistant-v1",
        "family": "ask",
        "version": "v1",
        "name": "Ask v1",
        "source": "checkpoints/backtalk-sft/best.pt",
        "out": "checkpoints/backtalk-assistant",
        "description": "General question answering and conversation.",
    },
    {
        "id": "backtalk-assistant-v2",
        "family": "ask",
        "version": "v2",
        "name": "Ask v2",
        "source": "checkpoints/backtalk-sft-40000-step5300/best.pt",
        "out": "checkpoints/backtalk-assistant-v2",
        "description": "General conversation and answers — improved base (40k-step "
                       "pretrain) and SFT.",
    },
    {
        "id": "backtalk-storyteller-v1",
        "family": "stories",
        "version": "v1",
        "name": "Stories v1",
        "source": "checkpoints/full/ckpt_best.pt",
        "out": "checkpoints/backtalk-storyteller",
        "description": "Structured short fairy-tale generation.",
    },
]


def export(spec, force):
    out_dir = spec["out"]
    out_path = os.path.join(out_dir, "model.pt")
    if os.path.exists(out_path) and not force:
        print(f"{spec['name']}: up to date ({out_path})")
        return
    source = spec["source"]
    if not os.path.exists(source):
        raise SystemExit(f"missing source checkpoint: {source}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    exported = {
        "model": checkpoint["model"],
        "config": checkpoint["config"],
        "tokenizer": checkpoint.get("tokenizer"),
        "source_checkpoint": source,
        "source_step": checkpoint.get("step", checkpoint.get("iter")),
        "source_val_loss": checkpoint.get("val_loss"),
        "kind": "backtalk-inference",
        "model_id": spec["id"],
        "model_name": spec["name"],
    }
    torch.save(exported, out_path)
    metadata = {
        "id": spec["id"],
        "family": spec["family"],
        "version": spec["version"],
        "name": spec["name"],
        "description": spec["description"],
        "source": source,
        "checkpoint": out_path,
        "source_step": exported["source_step"],
        "source_val_loss": exported["source_val_loss"],
        "context_length": checkpoint["config"]["block_size"],
        "vocab_size": checkpoint["config"]["vocab_size"],
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"{spec['name']}: {out_path} ({os.path.getsize(out_path) / 1024**2:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-export even if model.pt exists")
    args = parser.parse_args()
    for spec in MODELS:
        export(spec, args.force)


if __name__ == "__main__":
    main()
