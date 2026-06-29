#!/usr/bin/env python3
"""Remove validation leakage from an existing clean corpus, then revalidate it."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from clean_pretraining_corpus import (
    GB,
    atomic_json,
    clean_text,
    parse_workers,
    validate_output,
    validation_flags,
    validation_verdict,
)


def repair_batch(lines: list[bytes]) -> tuple[list[bytes], int, dict[str, int]]:
    output: list[bytes] = []
    rejected = 0
    fixed: dict[str, int] = {}
    for raw in lines:
        obj = json.loads(raw)
        flags = validation_flags(obj["text"])
        if flags:
            for flag in flags:
                fixed[flag] = fixed.get(flag, 0) + 1
            pieces, _reason = clean_text(obj["text"], obj["source"])
            pieces = [piece for piece in pieces if not validation_flags(piece)]
            if not pieces:
                rejected += 1
                continue
            obj["text"] = pieces[0]
        # Spaces after JSON separators add a small safety margin above 10 GB
        # without modifying the text or introducing synthetic padding.
        output.append(
            json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"
        )
    return output, rejected, fixed


def line_batches(handle: Any, count: int):
    while True:
        lines = []
        for _ in range(count):
            line = handle.readline()
            if not line:
                break
            if line.strip():
                lines.append(line)
        if not lines:
            return
        yield lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default=parse_workers("auto"))
    parser.add_argument("--batch-docs", type=int, default=1024)
    parser.add_argument("--target-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    temporary = corpus.with_name(corpus.name + ".repair.tmp")
    progress_path = corpus.with_suffix(".progress.json")
    stats_path = corpus.with_suffix(".stats.json")
    previous_stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else {}
    )
    source_size = corpus.stat().st_size
    processed = written_docs = rejected = 0
    fixed: dict[str, int] = {}
    started = time.monotonic()
    progress = tqdm(
        total=source_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc="repair",
        dynamic_ncols=True,
    )
    try:
        with corpus.open("rb") as source, temporary.open("wb") as destination:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                pending: set[concurrent.futures.Future[Any]] = set()
                batches = iter(line_batches(source, args.batch_docs))
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < args.workers * 3:
                        try:
                            batch = next(batches)
                        except StopIteration:
                            exhausted = True
                            break
                        future = executor.submit(repair_batch, batch)
                        future.input_bytes = sum(map(len, batch))  # type: ignore[attr-defined]
                        pending.add(future)
                    if not pending:
                        break
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    # Completion order does not matter: corpus ordering was already
                    # shuffled, and IDs/text remain unchanged.
                    for future in done:
                        records, batch_rejected, batch_fixed = future.result()
                        for record in records:
                            destination.write(record)
                        batch_bytes = future.input_bytes  # type: ignore[attr-defined]
                        processed += batch_bytes
                        written_docs += len(records)
                        rejected += batch_rejected
                        for key, value in batch_fixed.items():
                            fixed[key] = fixed.get(key, 0) + value
                        progress.update(batch_bytes)
                        elapsed = max(0.001, time.monotonic() - started)
                        progress.set_postfix(
                            docs=f"{written_docs:,}",
                            accepted=f"{written_docs:,}",
                            rejected=f"{rejected:,}",
                            MB_s=f"{processed / elapsed / 1e6:.1f}",
                        )
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        progress.close()

    if temporary.stat().st_size < int(args.target_gb * GB):
        raise RuntimeError(
            f"repair output is below target: {temporary.stat().st_size} bytes"
        )
    os.replace(temporary, corpus)

    state = dict(previous_stats)
    state.update(
        {
            "stage": "validation",
            "complete": False,
            "output": str(corpus),
            "output_bytes": corpus.stat().st_size,
            "accepted_docs": written_docs,
            "repair": {
                "fixed_flagged_docs": fixed,
                "rejected_docs": rejected,
            },
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    atomic_json(progress_path, state)
    validation = validate_output(
        corpus,
        args.workers,
        args.batch_docs,
        args.seed,
        state,
        progress_path,
    )
    verdict = validation_verdict(validation, int(args.target_gb * GB))
    state.update(
        {
            "stage": "complete",
            "complete": verdict == "PASS",
            "verdict": verdict,
            "validation": validation,
            "output_bytes": corpus.stat().st_size,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    atomic_json(progress_path, state)
    atomic_json(stats_path, state)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "documents": written_docs,
                "size_bytes": corpus.stat().st_size,
                "fixed_flags": fixed,
                "rejected_docs": rejected,
                "sha256": validation["sha256"],
            },
            indent=2,
        )
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
