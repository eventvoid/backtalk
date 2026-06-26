#!/usr/bin/env bash
# Generate the fairy-tale dataset. Resumable: stop with Ctrl-C and re-run the
# same command -- only the missing stories are regenerated.
#
#   ./run.sh                              # cloud (OpenRouter, Groq), 300k
#   BACKEND=deepseek TARGET=100000 ./run.sh   # DeepSeek; TARGET=100000 closes
#                                             # every combo to >=1 (breadth first)
#   BACKEND=ollama ./run.sh              # local (Ollama llama3.1:8b), free
#   TARGET=200 OUT=data/test.jsonl ./run.sh   # quick smoke test
#
# OPENROUTER_API_KEY / DEEPSEEK_API_KEY are read from .env (cloud backends).
# Env knobs: BACKEND MODEL OLLAMA_MODEL TARGET OUT WORKERS PROVIDERS RETRIES LOG DEBUG
#   DEBUG=1 ./run.sh   # also log every retry attempt (verbose) for investigation
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${BACKEND:-openrouter}"
TARGET="${TARGET:-300000}"
OUT="${OUT:-data/stories.jsonl}"
LOG="${LOG:-generate.log}"
RETRIES="${RETRIES:-3}"   # max attempts/story (a cap; ~93% pass on attempt 1)

DEBUG_FLAG=""
[ -n "${DEBUG:-}" ] && DEBUG_FLAG="--debug"

if [ "$BACKEND" = "ollama" ]; then
  OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
  WORKERS="${WORKERS:-1}"   # modest laptop; bump if your machine can take it
  echo ">> backend=ollama model=$OLLAMA_MODEL workers=$WORKERS target=$TARGET retries=$RETRIES out=$OUT log=$LOG"
  exec python3 generate_dataset.py --backend ollama --ollama-model "$OLLAMA_MODEL" \
    --target "$TARGET" --out "$OUT" --workers "$WORKERS" --retries "$RETRIES" --log "$LOG" $DEBUG_FLAG
elif [ "$BACKEND" = "deepseek" ]; then
  WORKERS="${WORKERS:-32}"
  echo ">> backend=deepseek model=deepseek-chat workers=$WORKERS target=$TARGET retries=$RETRIES out=$OUT log=$LOG"
  exec python3 generate_dataset.py --backend deepseek \
    --target "$TARGET" --out "$OUT" --workers "$WORKERS" --retries "$RETRIES" --log "$LOG" $DEBUG_FLAG
else
  MODEL="${MODEL:-meta-llama/llama-3.1-8b-instruct}"
  WORKERS="${WORKERS:-32}"
  PROVIDERS="${PROVIDERS:-groq}"
  echo ">> backend=openrouter model=$MODEL workers=$WORKERS providers=$PROVIDERS target=$TARGET retries=$RETRIES out=$OUT log=$LOG"
  exec python3 generate_dataset.py --model "$MODEL" --target "$TARGET" --out "$OUT" \
    --workers "$WORKERS" --providers "$PROVIDERS" --retries "$RETRIES" --log "$LOG" $DEBUG_FLAG
fi
