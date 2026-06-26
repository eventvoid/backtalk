# backtalk web UI

A small, fast local tool for testing the reversed-story model in the browser.
It reuses the inference code in `train/infer.py` (`load_model` +
`stream_generation`) — no duplicated logic.

## Run

```bash
# from the repo root
python3 web/server.py
```

Then open **http://127.0.0.1:8000**. The model is loaded once at startup
(default `checkpoints/full/ckpt_best.pt`) and reused for every request.

Options: `--ckpt <path>` (e.g. `checkpoints/smoke/ckpt_best.pt`),
`--device auto|mps|cpu`, `--host`, `--port`.

## UI

- The five inputs (**hero, setting, problem, helper_or_item, lesson**) are
  **dropdowns** restricted to a fixed set of values.
- **Generate** streams the story live as it is produced. **🎲 Random** picks a
  random combination.
- Compact **metrics** after each run: latency (time to first token), total time,
  throughput (tokens/s), token count, and the seed used.
- **Advanced** (collapsed): `temperature`, `top_k`, `max_new_tokens`, `seed`.
  Leave seed blank for a random (but reported) seed; set it to reproduce a run.
- **Generation data** (collapsed, always populated): input/context, the reversed
  prompt actually sent to the model, the raw reversed output, the final story,
  and the seed + generation params.

> The model generates the story *reversed*, so the readable text streams in
> end-first — this is inherent to the reversed task.

## Validation

The server only accepts the allowed dropdown values for the five fields
(anything else → HTTP 400) and clamps generation params to safe ranges:
`temperature ∈ [0.05, 2.0]`, `top_k ∈ [1, 256]`, `max_new_tokens ∈ [16, 1000]`.

## Requirements

`fastapi`, `uvicorn`, `torch` (already used by the training pipeline). No
frontend frameworks — a single static `web/index.html` with vanilla JS using
`fetch` streaming.
