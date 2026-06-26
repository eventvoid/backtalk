# backtalk

A small byte-level language model, trained **from scratch**, that writes short fairy
tales — backwards.

Each example is a structured prompt (`hero`, `setting`, `problem`, `helper_or_item`,
`lesson`) paired with a short story. Before training, both the prompt and the story
are reversed **character by character**, so the model learns to map a reversed prompt
to a reversed story. At inference the input is reversed in, and the model's reversed
output is flipped back into readable text.

It is deliberately tiny (~11M parameters) and trains on a MacBook (Apple Silicon / MPS).

## Why reversed?

Mostly as an experiment. A character-reversed task is a natural fit for a byte/char-level
model (one byte = one token, so reversal is lossless and trivial to encode), and it makes
for some interesting behaviour: the model generates a story end-first, can write a
"prequel" to a given ending, and — because it was trained on a continuous stream of
`<|input|>…<|story|>…<|end|>` examples — will happily invent its own inputs and keep
producing fresh stories if you let it run past the end marker.

## Layout

```
generate_dataset.py            # build the fairy-tale dataset (OpenRouter / DeepSeek / Ollama)
run.sh                         # convenience wrapper around generate_dataset.py
scripts/
  make_reversed_prompts.py     # build the reversed training file from the dataset
  verify_reversed_prompts.py   # validate that file against the source
train/
  model.py                     # the GPT (nanoGPT-style, byte-level)
  data.py                      # byte tokenizer + data preparation
  train.py                     # training loop (MPS, checkpoints, resume, progress bar)
  infer.py                     # command-line inference
web/
  server.py                    # FastAPI server (streaming generation)
  index.html                   # minimal browser UI
```

The dataset and model weights are **not** committed (they are large and regenerable —
see `.gitignore`). The steps below rebuild them.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dataset generation needs an API key: copy `.env.example` to `.env` and fill it in.
Training, inference, and the web UI do not need any keys.

## Pipeline

**1. Generate the dataset** (~260k stories → `data/stories.jsonl`):

```bash
./run.sh                          # OpenRouter (Groq), needs OPENROUTER_API_KEY
BACKEND=ollama ./run.sh           # local & free, needs Ollama (llama3.1:8b)
TARGET=200 OUT=data/test.jsonl ./run.sh   # quick smoke test
```

**2. Build and verify the reversed training file** (`data/stories_reversed_prompts.jsonl`):

```bash
python3 scripts/make_reversed_prompts.py
python3 scripts/verify_reversed_prompts.py
```

**3. Train** (writes checkpoints to `checkpoints/full/`):

```bash
# smoke test first (tiny model, 2k examples, a few hundred iters)
python3 train/train.py --max_examples 2000 --n_layer 4 --n_embd 128 --n_head 4 \
    --block_size 256 --batch_size 32 --max_iters 300 --out_dir checkpoints/smoke

# full training
python3 train/train.py --out_dir checkpoints/full --max_iters 20000
python3 train/train.py --out_dir checkpoints/full --resume     # continue
```

**4. Generate** — command line or browser:

```bash
python3 train/infer.py --hero dog --setting "snowy field" --problem "big storm" \
    --helper_or_item "little boat" --lesson sharing

python3 web/server.py             # then open http://127.0.0.1:8000
```

See [`web/README.md`](web/README.md) for the web UI details.

## Model

- **Tokenizer:** byte-level — 256 byte values + one EOS token (vocab 257). No vocabulary
  scan, robust to any UTF-8, and a natural fit for character-reversed text.
- **Architecture:** decoder-only GPT — 6 layers, 6 heads, 384-dim, context window 640
  → ~10.99M parameters.
- **Training:** float32 on MPS, AdamW, cosine LR with warmup, train/val split, checkpoints
  with resume.

## License

MIT — see [`LICENSE`](LICENSE).
