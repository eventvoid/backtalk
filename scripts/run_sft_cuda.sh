#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/backtalk-sft

MODE="${1:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/sft_${MODE}_${STAMP}.log"

echo "mode=$MODE log=$LOG"
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0))
print("bf16", torch.cuda.is_bf16_supported())
print("vram_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

case "$MODE" in
  smoke)
    python3 train/cli.py sft \
      --config configs/sft_cuda.json \
      --out_dir checkpoints/backtalk-sft-smoke \
      --batch_size "${BATCH_SIZE:-64}" \
      --max_steps 20 --eval_interval 20 --eval_batches 2 --log_interval 1 \
      2>&1 | tee "$LOG"
    ;;
  full)
    python3 train/cli.py sft --config configs/sft_cuda.json \
      --batch_size "${BATCH_SIZE:-64}" 2>&1 | tee "$LOG"
    ;;
  resume)
    python3 train/cli.py resume-sft --config configs/sft_cuda.json \
      --batch_size "${BATCH_SIZE:-64}" 2>&1 | tee "$LOG"
    ;;
  *)
    echo "Usage: bash scripts/run_sft_cuda.sh [smoke|full|resume]" >&2
    exit 2
    ;;
esac
