#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/pretrain_cuda.json}"
TOKENIZER="tokenizers/backtalk-tokenizer/tokenizer.json"
PREPARED_DIR="data/prepared/backtalk-ctx2048"
CHECKPOINT_DIR="checkpoints/backtalk"
LOG_DIR="logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

PRESET="smoke"
RESUME=0
TARGET_VAL_LOSS="${TARGET_VAL_LOSS:-}"
TARGET_TRAIN_LOSS="${TARGET_TRAIN_LOSS:-}"
SAFETY_STEPS="${SAFETY_STEPS:-100000}"
LOSS_LR_DECAY_STEPS="${LOSS_LR_DECAY_STEPS:-10000}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    smoke|short|medium|epoch1|loss) PRESET="$1"; shift ;;
  esac
fi

usage() {
  cat <<USAGE
Usage: bash scripts/run_pretrain_cuda.sh [preset] [options]

Presets:
  smoke    50-step CUDA sanity test only
  short    500 training steps
  medium   2000 training steps
  epoch1   about one epoch
  loss     train until target loss, with max_steps as safety cap

Options:
  --resume                 Resume from checkpoints/backtalk/ckpt.pt
  --target-val-loss N      Stop after eval when validation loss <= N
  --target-train-loss N    Stop after eval when train loss <= N
  --max-steps N            Safety cap for loss mode, or override preset steps
  --lr-decay-steps N       LR decay horizon for loss mode
  -h, --help               Show this help
USAGE
}

MAX_STEPS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|short|medium|epoch1|loss) PRESET="$1" ;;
    --resume) RESUME=1 ;;
    --target-val-loss) TARGET_VAL_LOSS="$2"; shift ;;
    --target-train-loss) TARGET_TRAIN_LOSS="$2"; shift ;;
    --max-steps) MAX_STEPS_OVERRIDE="$2"; shift ;;
    --lr-decay-steps) LOSS_LR_DECAY_STEPS="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

section() {
  printf '\n'
  printf '============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

run_logged() {
  local name="$1"
  shift
  local log="$LOG_DIR/${STAMP}_${name}.log"
  echo "Command: $*"
  echo "Log: $log"
  set +e
  "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "Step failed: $name (exit $status). See $log" >&2
    exit "$status"
  fi
}

section "BackTalk CUDA pretraining"
echo "Preset: $PRESET"
echo "Config: $CONFIG"
echo "Prepared data: $PREPARED_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR/"

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ -f "$TOKENIZER" ]] || { echo "Missing tokenizer: $TOKENIZER" >&2; exit 1; }
[[ -f "$PREPARED_DIR/meta.json" ]] || { echo "Missing prepared data: $PREPARED_DIR/meta.json" >&2; exit 1; }

section "1. Install/check dependencies"
run_logged "setup" "$PYTHON" -m pip install -r requirements.txt

section "2. Check CUDA accelerator"
run_logged "check_cuda" "$PYTHON" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    props = torch.cuda.get_device_properties(0)
    print(f"total_vram_gb={props.total_memory / 1024**3:.1f}")
else:
    raise SystemExit("CUDA is not available; this script requires a CUDA accelerator.")
PY

PRESET_PLAN="$("$PYTHON" - "$CONFIG" "$PREPARED_DIR/meta.json" "$PRESET" <<'PY'
import json
import math
import sys

config = json.load(open(sys.argv[1]))
meta = json.load(open(sys.argv[2]))
preset = sys.argv[3]
tokens_per_step = config["batch_size"] * config["block_size"] * config["gradient_accumulation_steps"]
if preset == "smoke":
    steps = 50
elif preset == "short":
    steps = 500
elif preset == "medium":
    steps = 2000
elif preset == "epoch1":
    steps = math.ceil(meta["train_tokens"] / tokens_per_step)
elif preset == "loss":
    steps = int(config.get("max_steps", 100000))
else:
    raise SystemExit(f"unknown preset: {preset}")
epochs = steps * tokens_per_step / meta["train_tokens"]
print(f"{steps}|preset={preset} steps={steps:,} tokens_per_step={tokens_per_step:,} estimated_epochs={epochs:.3f}")
PY
)"
MAX_STEPS="${PRESET_PLAN%%|*}"
if [[ -n "$MAX_STEPS_OVERRIDE" ]]; then
  MAX_STEPS="$MAX_STEPS_OVERRIDE"
elif [[ "$PRESET" == "loss" ]]; then
  MAX_STEPS="$SAFETY_STEPS"
fi
PRESET_SUMMARY="${PRESET_PLAN#*|}"

section "3. Train preset: $PRESET"
echo "$PRESET_SUMMARY"
echo "safety_max_steps=$MAX_STEPS target_val_loss=${TARGET_VAL_LOSS:-off} target_train_loss=${TARGET_TRAIN_LOSS:-off}"
if [[ "$PRESET" == "loss" ]]; then
  echo "loss_lr_decay_steps=$LOSS_LR_DECAY_STEPS"
fi

TRAIN_ARGS=(--config "$CONFIG" --max_steps "$MAX_STEPS")
if [[ "$PRESET" == "loss" ]]; then
  TRAIN_ARGS+=(--lr_decay_steps "$LOSS_LR_DECAY_STEPS")
fi
if [[ -n "$TARGET_VAL_LOSS" ]]; then
  TRAIN_ARGS+=(--target_val_loss "$TARGET_VAL_LOSS")
fi
if [[ -n "$TARGET_TRAIN_LOSS" ]]; then
  TRAIN_ARGS+=(--target_train_loss "$TARGET_TRAIN_LOSS")
fi

if [[ "$RESUME" -eq 1 ]]; then
  run_logged "resume_${PRESET}" "$PYTHON" train/cli.py resume "${TRAIN_ARGS[@]}"
else
  run_logged "pretrain_${PRESET}" "$PYTHON" train/cli.py pretrain "${TRAIN_ARGS[@]}"
fi
