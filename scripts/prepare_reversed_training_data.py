#!/usr/bin/env python3
"""Create and independently validate BackTalk reversed pretraining/SFT JSONL."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


SPECIAL_RE = re.compile(r"<\|(?:bos|eos|pad|unk)\|>")
CHECKPOINT_BYTES = 128 * 1024 * 1024


def reverse_text(text: str) -> str:
    """Reverse Unicode code points while treating supported special tokens atomically."""
    if "<|" not in text:
        return text[::-1]
    units: list[str] = []
    position = 0
    for match in SPECIAL_RE.finditer(text):
        units.extend(text[position : match.start()])
        units.append(match.group(0))
        position = match.end()
    units.extend(text[position:])
    units.reverse()
    return "".join(units)


def _reverse_record(task: tuple[bytes, tuple[str, ...]]) -> tuple[bytes, int, int]:
    raw, fields = task
    obj = json.loads(raw)
    if set(obj) != set(fields):
        raise ValueError(f"incorrect fields: {sorted(obj)}; expected {sorted(fields)}")
    chars = 0
    for field in fields:
        value = obj[field]
        if field in {"id", "source"}:
            if not isinstance(value, str):
                raise TypeError(f"{field} is not a string")
            continue
        if not isinstance(value, str):
            raise TypeError(f"{field} is not a string")
        reversed_value = reverse_text(value)
        if reverse_text(reversed_value) != value:
            raise ValueError(f"round-trip failed for field {field}")
        obj[field] = reversed_value
        chars += len(value)
    encoded = (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    return encoded, chars, len(raw)


def _validate_pair(
    task: tuple[bytes, bytes, tuple[str, ...]]
) -> tuple[bool, str, int, int]:
    source_raw, reversed_raw, fields = task
    try:
        source = json.loads(source_raw)
        reversed_obj = json.loads(reversed_raw)
        if set(source) != set(fields) or set(reversed_obj) != set(fields):
            return False, "incorrect_fields", 0, 0
        for field in fields:
            if not isinstance(source[field], str) or not isinstance(reversed_obj[field], str):
                return False, "non_string_value", 0, 0
            if field in {"id", "source"}:
                if source[field] != reversed_obj[field]:
                    return False, f"changed_{field}", 0, 0
            elif reverse_text(reversed_obj[field]) != source[field]:
                return False, f"roundtrip_{field}", 0, 0
        return True, "", len(source_raw), len(reversed_raw)
    except Exception as exc:
        return False, type(exc).__name__, 0, 0


def resolve_workers(value: str) -> int:
    if value == "auto":
        return os.cpu_count() or 1
    workers = int(value)
    if workers < 1:
        raise ValueError("--workers must be auto or a positive integer")
    return workers


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def batches(handle: Any, batch_size: int) -> Iterable[tuple[list[bytes], int]]:
    while True:
        batch: list[bytes] = []
        for _ in range(batch_size):
            line = handle.readline()
            if not line:
                break
            if line.strip():
                batch.append(line)
        if not batch:
            return
        yield batch, handle.tell()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file(
    source: Path,
    output: Path,
    fields: tuple[str, ...],
    workers: int,
    resume: bool,
    batch_size: int,
) -> dict[str, Any]:
    progress_path = output.with_suffix(".progress.json")
    source_size = source.stat().st_size
    state: dict[str, Any] = {
        "version": 1,
        "stage": "reverse",
        "source": str(source),
        "source_size": source_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "output": str(output),
        "input_offset": 0,
        "output_bytes": 0,
        "documents": 0,
        "characters": 0,
        "complete": False,
    }
    if resume and progress_path.exists() and output.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        identity = (
            saved.get("source_size") == source_size
            and saved.get("source_mtime_ns") == source.stat().st_mtime_ns
            and saved.get("source") == str(source)
        )
        if not identity:
            raise RuntimeError(f"resume source differs from checkpoint {progress_path}")
        state.update(saved)
        with output.open("r+b") as handle:
            handle.truncate(int(state["output_bytes"]))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    started = time.monotonic()
    last_checkpoint = int(state["output_bytes"])
    mode = "ab" if int(state["output_bytes"]) else "wb"
    bar = tqdm(
        total=source_size,
        initial=int(state["input_offset"]),
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc=f"reverse:{output.name}",
        dynamic_ncols=True,
    )
    with source.open("rb") as source_handle, output.open(mode) as output_handle:
        source_handle.seek(int(state["input_offset"]))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for batch, offset in batches(source_handle, batch_size):
                tasks = ((raw, fields) for raw in batch)
                results = executor.map(_reverse_record, tasks, chunksize=64)
                consumed = 0
                for encoded, chars, raw_bytes in results:
                    output_handle.write(encoded)
                    state["output_bytes"] += len(encoded)
                    state["documents"] += 1
                    state["characters"] += chars
                    consumed += raw_bytes
                previous = int(state["input_offset"])
                state["input_offset"] = offset
                bar.update(offset - previous)
                elapsed = max(0.001, time.monotonic() - started)
                bar.set_postfix(
                    docs=f"{int(state['documents']):,}",
                    docs_s=f"{len(batch) / max(0.001, elapsed):.0f}",
                    MB_s=f"{consumed / elapsed / 1e6:.1f}",
                )
                started = time.monotonic()
                if int(state["output_bytes"]) - last_checkpoint >= CHECKPOINT_BYTES:
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    state["updated_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    atomic_json(progress_path, state)
                    last_checkpoint = int(state["output_bytes"])
        output_handle.flush()
        os.fsync(output_handle.fileno())
    bar.close()
    state["complete"] = True
    state["stage"] = "validate"
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(progress_path, state)
    return state


def validate_pair(
    source: Path,
    output: Path,
    fields: tuple[str, ...],
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    source_size = source.stat().st_size
    source_docs = output_docs = matches = mismatches = 0
    reasons: dict[str, int] = {}
    samples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    bar = tqdm(
        total=source_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc=f"validate:{output.name}",
        dynamic_ncols=True,
    )
    started = time.monotonic()
    with source.open("rb") as src, output.open("rb") as dst:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            while True:
                source_batch = [src.readline() for _ in range(batch_size)]
                source_batch = [line for line in source_batch if line]
                output_batch = [dst.readline() for _ in range(batch_size)]
                output_batch = [line for line in output_batch if line]
                if not source_batch and not output_batch:
                    break
                source_docs += len(source_batch)
                output_docs += len(output_batch)
                if len(source_batch) != len(output_batch):
                    difference = abs(len(source_batch) - len(output_batch))
                    mismatches += difference
                    reasons["line_count_batch"] = reasons.get("line_count_batch", 0) + difference
                paired = min(len(source_batch), len(output_batch))
                if len(samples) < 10:
                    for a, b in zip(source_batch, output_batch):
                        if len(samples) == 10:
                            break
                        samples.append((json.loads(a), json.loads(b)))
                tasks = (
                    (source_batch[index], output_batch[index], fields)
                    for index in range(paired)
                )
                for ok, reason, _, _ in executor.map(
                    _validate_pair, tasks, chunksize=64
                ):
                    if ok:
                        matches += 1
                    else:
                        mismatches += 1
                        reasons[reason] = reasons.get(reason, 0) + 1
                consumed = sum(map(len, source_batch))
                bar.update(consumed)
                elapsed = max(0.001, time.monotonic() - started)
                bar.set_postfix(
                    docs=f"{source_docs:,}",
                    docs_s=f"{len(source_batch) / elapsed:.0f}",
                    MB_s=f"{consumed / elapsed / 1e6:.1f}",
                    mismatch=mismatches,
                )
                started = time.monotonic()
    bar.close()
    return {
        "verdict": (
            "PASS"
            if source_docs == output_docs and mismatches == 0 and matches == source_docs
            else "FAIL"
        ),
        "source_lines": source_docs,
        "reversed_lines": output_docs,
        "roundtrip_matches": matches,
        "roundtrip_mismatches": mismatches,
        "mismatch_reasons": reasons,
        "source_size_bytes": source_size,
        "reversed_size_bytes": output.stat().st_size,
        "source_sha256": sha256_file(source),
        "reversed_sha256": sha256_file(output),
        "samples": samples,
    }


def sample_text(value: str, limit: int = 240) -> str:
    return value[:limit].replace("\n", "⏎")


def write_samples(
    path: Path,
    samples: list[tuple[dict[str, Any], dict[str, Any]]],
    fields: tuple[str, ...],
) -> None:
    lines = ["# 10 normal → reversed samples", ""]
    text_fields = [field for field in fields if field not in {"id", "source"}]
    for index, (normal, reversed_obj) in enumerate(samples, 1):
        metadata = " ".join(
            f"{key}={normal[key]!r}" for key in ("id", "source") if key in normal
        )
        lines.append(f"## {index}. {metadata}".rstrip())
        for field in text_fields:
            lines.append(f"{field} normal:   {sample_text(normal[field])}")
            lines.append(f"{field} reversed: {sample_text(reversed_obj[field])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_one(
    source: Path,
    output: Path,
    fields: tuple[str, ...],
    workers: int,
    resume: bool,
    batch_size: int,
) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    build = build_file(source, output, fields, workers, resume, batch_size)
    validation = validate_pair(source, output, fields, workers, batch_size)
    stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transform": "Unicode code-point reversal with bos/eos/pad/unk atomic",
        "fields": list(fields),
        "build": build,
        "validation": {k: v for k, v in validation.items() if k != "samples"},
        "verdict": validation["verdict"],
    }
    atomic_json(output.with_suffix(".stats.json"), stats)
    output.with_suffix(".sha256").write_text(
        f"{validation['reversed_sha256']}  {output.name}\n", encoding="utf-8"
    )
    write_samples(output.with_suffix(".samples.txt"), validation["samples"], fields)
    progress_path = output.with_suffix(".progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "stage": "complete",
            "complete": validation["verdict"] == "PASS",
            "verdict": validation["verdict"],
            "roundtrip_mismatches": validation["roundtrip_mismatches"],
        }
    )
    atomic_json(progress_path, progress)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-input",
        type=Path,
        default=Path("data/corpus/corpus_10gb_clean.jsonl"),
    )
    parser.add_argument(
        "--pretrain-output",
        type=Path,
        default=Path("data/corpus/backtalk_pretrain_10gb_reversed.jsonl"),
    )
    parser.add_argument("--sft-input", type=Path, default=Path("data/sft_raw.jsonl"))
    parser.add_argument("--sft-output", type=Path, default=Path("data/sft_reversed.jsonl"))
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workers = resolve_workers(args.workers)
    pretrain = process_one(
        args.pretrain_input,
        args.pretrain_output,
        ("id", "source", "text"),
        workers,
        args.resume,
        args.batch_size,
    )
    results: dict[str, Any] = {"pretraining": pretrain}
    if args.sft_input.exists():
        results["sft"] = process_one(
            args.sft_input,
            args.sft_output,
            ("user", "assistant"),
            workers,
            args.resume,
            args.batch_size,
        )
    verdict = (
        "PASS"
        if pretrain["verdict"] == "PASS"
        and results.get("sft", {"verdict": "PASS"})["verdict"] == "PASS"
        else "FAIL"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "pretraining_output": str(args.pretrain_output),
                "pretraining_lines": pretrain["validation"]["reversed_lines"],
                "pretraining_mismatches": pretrain["validation"][
                    "roundtrip_mismatches"
                ],
                "sft_output": str(args.sft_output) if "sft" in results else None,
                "sft_lines": results.get("sft", {}).get("validation", {}).get(
                    "reversed_lines"
                ),
                "sft_mismatches": results.get("sft", {}).get("validation", {}).get(
                    "roundtrip_mismatches"
                ),
            },
            indent=2,
        )
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun with --resume", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
