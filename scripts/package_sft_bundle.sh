#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-dist/backtalk-sft-bundle.tar.gz}"
BASE="checkpoints/backtalk-base/ckpt_best.pt"

for path in \
  requirements.txt train configs/sft_cuda.json \
  tokenizers/backtalk-tokenizer data/prepared/backtalk-sft-ctx2048 "$BASE"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")"
echo "Packaging SFT bundle: $OUT"
tar -czf "$OUT" \
  requirements.txt train configs/sft_cuda.json \
  tokenizers/backtalk-tokenizer data/prepared/backtalk-sft-ctx2048 \
  scripts/run_sft_cuda.sh \
  "$BASE"
shasum -a 256 "$OUT"
ls -lh "$OUT"
