"""BackTalk backend node.

A worker that loads the models and serves generation jobs from the gateway only
(authenticated by the node token). On first start it downloads any missing model
files from MODEL_SOURCE, registers with the gateway, then heartbeats its status.

    python3 backend-node/worker.py            # run locally (from repo root)
    GATEWAY_URL=http://localhost:8080 NODE_TOKEN=... python3 backend-node/worker.py
"""
import json
import os
import queue
import shutil
import socket
import threading
import time
import urllib.request
from typing import Optional
from urllib.error import URLError

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

import inference
from inference import ROOT, Engine, ValidationError

try:
    import psutil
except ImportError:  # optional
    psutil = None


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


CONFIG = {
    "gateway_url": env("GATEWAY_URL", "http://localhost:8080").rstrip("/"),
    "node_token": env("NODE_TOKEN", ""),
    "name": env("NODE_NAME", socket.gethostname()),
    "host": env("NODE_HOST", "0.0.0.0"),
    "port": int(env("NODE_PORT", "9000")),
    "transport": env("NODE_TRANSPORT", "pull").lower(),
    "advertise_url": env("NODE_ADVERTISE_URL"),  # how the gateway reaches us
    "device": env("DEVICE", "auto"),
    "max_concurrency": int(env("NODE_MAX_CONCURRENCY", "1")),
    "model_source": env("MODEL_SOURCE"),  # local dir or http(s) base for missing files
    "tokenizer": env("TOKENIZER", inference.DEFAULT_TOKENIZER),
    "heartbeat_s": int(env("HEARTBEAT_SECONDS", "10")),
    "poll_wait_ms": int(env("NODE_POLL_WAIT_MS", "30000")),
    "event_batch_ms": int(env("NODE_EVENT_BATCH_MS", "100")),
}

REQUIRED_FILES = [m["checkpoint"] for m in inference.MODELS] + [
    os.path.relpath(inference.DEFAULT_TOKENIZER, ROOT),
]

# Concurrency tracking. The node accepts unlimited concurrent requests; _active
# is just reported so the gateway can balance load. _tps is a rolling estimate
# of generation throughput (tokens/sec) the gateway routes by.
_lock = threading.Lock()
_inference_lock = threading.Lock()
_active = 0
_tps: Optional[float] = None
ENGINE: Optional[Engine] = None
app = FastAPI(title="BackTalk node", docs_url=None, redoc_url=None)


def log(msg, **extra):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "node": CONFIG["name"], "msg": msg, **extra}), flush=True)


def ensure_models():
    """Download/copy any missing model files from MODEL_SOURCE."""
    src = CONFIG["model_source"]
    for rel in REQUIRED_FILES:
        dest = os.path.join(ROOT, rel)
        if os.path.exists(dest):
            continue
        if not src:
            raise SystemExit(f"missing model file {rel} and no MODEL_SOURCE set to fetch it")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if src.startswith("http://") or src.startswith("https://"):
            url = f"{src.rstrip('/')}/{rel}"
            log("downloading model", file=rel, source=url)
            urllib.request.urlretrieve(url, dest)
        else:  # local path / mounted volume (also covers a MinIO/S3 mount)
            source_file = os.path.join(src, rel)
            if not os.path.exists(source_file):
                raise SystemExit(f"model file not found in MODEL_SOURCE: {source_file}")
            log("copying model", file=rel, source=source_file)
            shutil.copy2(source_file, dest)


def system_metrics():
    metrics = {}
    if psutil:
        try:
            metrics["cpu"] = round(psutil.cpu_percent(interval=None))
            metrics["ram"] = round(psutil.virtual_memory().percent)
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            metrics["gpu"] = torch.cuda.get_device_name(0)
            metrics["vram"] = round(100 * (1 - free / total))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            metrics["gpu"] = "mps"
    except Exception:
        pass
    return metrics


def advertise_url():
    if CONFIG["advertise_url"]:
        return CONFIG["advertise_url"]
    return f"http://{socket.gethostname()}:{CONFIG['port']}"


def _post(path, body, timeout=30):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{CONFIG['gateway_url']}{path}", data=data, method="POST",
        headers={"content-type": "application/json", "x-node-token": CONFIG["node_token"]},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def register():
    body = {
        "name": CONFIG["name"],
        "transport": CONFIG["transport"],
        "url": advertise_url() if CONFIG["transport"] == "push" else None,
        "models": ENGINE.advertised_models(), "max_concurrency": CONFIG["max_concurrency"],
        "throughput": round(_tps, 1) if _tps is not None else None,
        "system": system_metrics(),
    }
    for attempt in range(20):
        try:
            res = _post("/internal/nodes/register", body)
            log(
                "registered with gateway",
                transport=CONFIG["transport"],
                url=advertise_url() if CONFIG["transport"] == "push" else None,
            )
            return res
        except (URLError, OSError) as e:
            log("register failed, retrying", error=str(e), attempt=attempt)
            time.sleep(3)
    raise SystemExit("could not register with gateway")


def heartbeat_loop():
    while True:
        time.sleep(CONFIG["heartbeat_s"])
        with _lock:
            tps = _tps
        try:
            _post("/internal/nodes/heartbeat", {
                "name": CONFIG["name"],
                "throughput": round(tps, 1) if tps is not None else None,
                "system": system_metrics(), "models": ENGINE.advertised_models(),
            })
        except (URLError, OSError) as e:
            log("heartbeat failed", error=str(e))


@app.get("/internal/health")
def health():
    with _lock:
        active = _active
    return {"status": "ok", "active": active, "max": CONFIG["max_concurrency"]}


def _check_token(token):
    if not CONFIG["node_token"] or token != CONFIG["node_token"]:
        raise HTTPException(status_code=401, detail="node token required")


def _acquire():
    # Unlimited concurrency: always accept, just track the count for load reporting.
    global _active
    with _lock:
        _active += 1
    return True


def _record_tps(tokens, seconds):
    global _tps
    if seconds <= 0 or tokens <= 0:
        return
    sample = tokens / seconds
    with _lock:
        _tps = sample if _tps is None else (0.7 * _tps + 0.3 * sample)  # EMA


def _release():
    global _active
    with _lock:
        _active = max(0, _active - 1)


def _complete_job(job_id, body):
    """Retry result delivery: losing a response must not lose completed work."""
    for attempt in range(10):
        try:
            _post(f"/internal/nodes/jobs/{job_id}/complete", {
                "name": CONFIG["name"],
                "throughput": round(_tps, 1) if _tps is not None else None,
                "system": system_metrics(),
                **body,
            })
            return
        except (URLError, OSError) as e:
            log("job result delivery failed", job_id=job_id, error=str(e), attempt=attempt)
            time.sleep(min(1 + attempt, 10))
    log("job result abandoned", job_id=job_id)


def _event_sender(job_id, events):
    """Deliver real model events without blocking the generation loop."""
    while True:
        event = events.get()
        if event is None:
            return
        batch = [event]
        finished = False
        # A short transport window combines real adjacent token events into one
        # HTTP request. It does not invent or interpolate generation events.
        deadline = time.monotonic() + max(0, CONFIG["event_batch_ms"]) / 1000
        while len(batch) < 16:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                event = events.get(timeout=remaining)
            except queue.Empty:
                break
            if event is None:
                finished = True
                break
            batch.append(event)
        for attempt in range(5):
            try:
                _post(f"/internal/nodes/jobs/{job_id}/events", {
                    "name": CONFIG["name"],
                    "events": batch,
                })
                break
            except (URLError, OSError) as e:
                log("job event delivery failed", job_id=job_id, error=str(e), attempt=attempt)
                time.sleep(min(1 + attempt, 5))
        else:
            log("job events abandoned", job_id=job_id, count=len(batch))
        if finished:
            return


def polling_loop():
    """Pull jobs over outbound HTTP; no inbound node port is required."""
    log("polling for jobs", gateway=CONFIG["gateway_url"])
    while True:
        try:
            response = _post(
                "/internal/nodes/jobs/next",
                {
                    "name": CONFIG["name"],
                    "wait_ms": CONFIG["poll_wait_ms"],
                    "throughput": round(_tps, 1) if _tps is not None else None,
                    "system": system_metrics(),
                },
                timeout=max(30, CONFIG["poll_wait_ms"] / 1000 + 10),
            )
        except (URLError, OSError) as e:
            log("job poll failed", error=str(e))
            time.sleep(3)
            continue

        job = response.get("job")
        if not job:
            continue
        job_id = job["id"]
        req = job.get("payload") or {}
        _acquire()
        events = queue.Queue()
        sender = threading.Thread(target=_event_sender, args=(job_id, events), daemon=True)
        sender.start()
        try:
            with _inference_lock:
                started = time.time()
                result = None
                for event in ENGINE.generate_stream(
                    req.get("model", "backtalk-assistant"),
                    prompt=req.get("prompt"),
                    story=req.get("story"),
                    params=req.get("params") or {},
                ):
                    events.put(event)
                    if event.get("event") == "result":
                        result = {k: v for k, v in event.items() if k != "event"}
                elapsed = time.time() - started
            if result is None:
                raise RuntimeError("generation ended without a result")
            _record_tps(result.get("tokens", 0), elapsed)
            result["latency_ms"] = round(elapsed * 1000)
            events.put(None)
            sender.join()
            _complete_job(job_id, {"result": result})
        except ValidationError as e:
            events.put({"event": "error", "detail": str(e)})
            events.put(None)
            sender.join()
            _complete_job(job_id, {"error": str(e), "error_status": 400})
        except Exception as e:
            log("job generation error", job_id=job_id, error_type=type(e).__name__, error=str(e))
            events.put({"event": "error", "detail": "generation failed"})
            events.put(None)
            sender.join()
            _complete_job(job_id, {"error": "generation failed", "error_status": 500})
        finally:
            _release()


@app.post("/internal/generate")
def generate(req: dict, x_node_token: str = Header(default="")):
    _check_token(x_node_token)
    _acquire()
    try:
        # PyTorch's RNG and MPS command queue are process-global. Serializing
        # inference avoids first-request crashes and seed races when the
        # gateway sends overlapping work to an Apple Silicon node.
        with _inference_lock:
            started = time.time()
            result = ENGINE.generate(
                req.get("model", "backtalk-assistant"),
                prompt=req.get("prompt"), story=req.get("story"), params=req.get("params") or {},
            )
            elapsed = time.time() - started
        _record_tps(result.get("tokens", 0), elapsed)
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # never leak internals to the gateway
        log("generate error", error_type=type(e).__name__, error=str(e))
        raise HTTPException(status_code=500, detail="generation failed")
    finally:
        _release()


@app.post("/internal/generate/stream")
def generate_stream(req: dict, x_node_token: str = Header(default="")):
    _check_token(x_node_token)
    _acquire()
    _inference_lock.acquire()
    try:
        gen = ENGINE.generate_stream(
            req.get("model", "backtalk-assistant"),
            prompt=req.get("prompt"), story=req.get("story"), params=req.get("params") or {},
        )
        first = next(gen)  # runs validation + setup; raises before headers are sent
    except ValidationError as e:
        _inference_lock.release()
        _release()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _inference_lock.release()
        _release()
        log("generate error", error_type=type(e).__name__, error=str(e))
        raise HTTPException(status_code=500, detail="generation failed")

    def body():
        started = time.time()
        try:
            yield json.dumps(first) + "\n"
            for ev in gen:
                if ev.get("event") == "result":
                    _record_tps(ev.get("tokens", 0), time.time() - started)
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            log("stream error", error_type=type(e).__name__, error=str(e))
            yield json.dumps({"event": "error", "detail": "generation failed"}) + "\n"
        finally:
            _inference_lock.release()
            _release()

    return StreamingResponse(body(), media_type="application/x-ndjson")


def main():
    global ENGINE
    if not CONFIG["node_token"]:
        log("WARNING: NODE_TOKEN is empty; set it to match the gateway")
    ensure_models()
    log("loading models", device=CONFIG["device"])
    ENGINE = Engine(tokenizer_path=CONFIG["tokenizer"], device=CONFIG["device"])
    for spec in inference.MODELS:
        rt = ENGINE.runtimes[spec["id"]]
        log("model ready", id=spec["id"], params_m=round(rt.parameter_count / 1e6, 2), device=rt.device)
    register()
    if CONFIG["transport"] == "pull":
        polling_loop()
    else:
        threading.Thread(target=heartbeat_loop, daemon=True).start()
        log("node serving", host=CONFIG["host"], port=CONFIG["port"])
        uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning")


if __name__ == "__main__":
    main()
