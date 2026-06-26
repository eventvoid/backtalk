"""Minimal local web UI for the reversed-story model.

Reuses train/infer.py (load_model + stream_generation) — no duplicated logic.
The model is loaded once at startup and shared across requests (guarded by a
lock, since the model isn't thread-safe and we set a global RNG seed).

Run:
    python web/server.py                       # http://127.0.0.1:8000
    python web/server.py --ckpt checkpoints/full/ckpt_best.pt --port 8000
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Make train/ importable so we reuse the inference code directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from infer import load_model, stream_generation, DEFAULT_CKPT  # noqa: E402
from data import INPUT_ORDER  # noqa: E402

# Allowed dropdown values — the only accepted inputs for the five fields.
ALLOWED = {
    "hero": ["cat", "dog", "rabbit", "fox", "bear", "mouse", "dragon", "robot", "star", "cloud"],
    "setting": ["forest", "moon", "village", "castle", "garden", "sea", "mountain", "school", "dreamland", "snowy field"],
    "problem": ["lost friend", "dark night", "broken toy", "missing key", "scary noise", "sad day", "big storm", "forgotten promise", "lonely creature", "hard choice"],
    "helper_or_item": ["magic leaf", "tiny lamp", "wise owl", "golden bell", "kind fairy", "talking stone", "little boat", "rainbow rope", "warm scarf", "silver star"],
    "lesson": ["kindness", "sharing", "bravery", "honesty", "patience", "friendship", "helping others", "not giving up", "saying sorry", "being careful"],
}

# Safe ranges for generation params.
TEMP_MIN, TEMP_MAX = 0.05, 2.0
TOPK_MIN, TOPK_MAX = 1, 256
TOK_MIN, TOK_MAX = 16, 1000
SEED_MAX = 2 ** 31 - 1

app = FastAPI(title="backtalk")
STATE = {}
INDEX_HTML = os.path.join(HERE, "index.html")


class GenerateRequest(BaseModel):
    hero: str = ""
    setting: str = ""
    problem: str = ""
    helper_or_item: str = ""
    lesson: str = ""
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 600
    seed: Optional[int] = None


def validate(req: GenerateRequest):
    """Return (clean_input, params, error). Reject unknown field values."""
    inp = {}
    for k in INPUT_ORDER:
        v = (getattr(req, k) or "").strip()
        if v not in ALLOWED[k]:
            return None, None, f"invalid {k!r}: must be one of {ALLOWED[k]}"
        inp[k] = v
    try:
        temperature = max(TEMP_MIN, min(float(req.temperature), TEMP_MAX))
        top_k = max(TOPK_MIN, min(int(req.top_k), min(TOPK_MAX, STATE["cfg"].vocab_size)))
        max_new_tokens = max(TOK_MIN, min(int(req.max_new_tokens), TOK_MAX))
    except (TypeError, ValueError):
        return None, None, "generation params must be numeric"
    seed = req.seed
    if seed is None:
        seed = random.randint(0, SEED_MAX)
    else:
        seed = int(seed) % (SEED_MAX + 1)
    params = {"temperature": round(temperature, 3), "top_k": top_k,
              "max_new_tokens": max_new_tokens, "seed": seed}
    return inp, params, None


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/options")
def options():
    return {"fields": INPUT_ORDER, "allowed": ALLOWED,
            "ckpt": STATE["ckpt"], "device": STATE["device"]}


@app.post("/generate")
def generate(req: GenerateRequest):
    inp, params, err = validate(req)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    def ndjson(obj):
        return json.dumps(obj) + "\n"

    def stream():
        with STATE["lock"]:
            t0 = time.time()
            t_first = None
            last_flush = 0.0
            pending = ""
            last_n = 0
            for ev in stream_generation(
                    STATE["model"], STATE["cfg"], STATE["device"], inp,
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"], top_k=params["top_k"],
                    seed=params["seed"]):
                if ev["stage"] == "start":
                    yield ndjson({"event": "start", "input_text": ev["input_text"],
                                  "reversed_prompt": ev["reversed_prompt"],
                                  "params": params})
                elif ev["stage"] == "chunk":
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    pending += ev["delta"]
                    last_n = ev["n"]
                    # batch deltas by time to keep the wire light but incremental
                    if now - last_flush >= 0.04:
                        last_flush = now
                        yield ndjson({"event": "chunk", "delta": pending, "n": last_n})
                        pending = ""
                elif ev["stage"] == "done":
                    if pending:
                        yield ndjson({"event": "chunk", "delta": pending, "n": last_n})
                        pending = ""
                    total = time.time() - t0
                    latency = (t_first - t0) if t_first is not None else total
                    thr = (ev["n_tokens"] / total) if total > 0 else 0.0
                    yield ndjson({
                        "event": "done",
                        "story": ev["story"],
                        "reversed_story": ev["reversed_story"],
                        "metrics": {
                            "latency_ms": round(latency * 1000),
                            "total_ms": round(total * 1000),
                            "throughput": round(thr, 1),
                            "tokens": ev["n_tokens"],
                        },
                    })

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}\n"
                 f"Train one first, or pass --ckpt <path> (e.g. checkpoints/smoke/ckpt_best.pt).")

    print(f"loading model from {args.ckpt} ...", flush=True)
    model, cfg, device = load_model(args.ckpt, args.device)
    STATE.update(model=model, cfg=cfg, device=device, lock=threading.Lock(), ckpt=args.ckpt)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded on {device}: {n/1e6:.2f}M params, block_size={cfg.block_size}", flush=True)
    print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
