#!/usr/bin/env python3
"""Build and verify a server-ready ZIP after corpus validation passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def corpus_report(stats: dict[str, Any], corpus: Path) -> str:
    validation = stats["validation"]
    counts = validation["counts"]
    lengths = validation["document_length_words"]
    english = validation["english_estimate"]
    leakage = validation["leakage_docs"]
    lines = [
        "# BackTalk Pretraining Corpus — Final Validation",
        "",
        f"- **Verdict:** {stats['verdict']}",
        f"- **Corpus:** `{corpus}`",
        f"- **Bytes:** {validation['file_size_bytes']:,}",
        f"- **SHA-256:** `{validation['sha256']}`",
        f"- **Documents:** {counts.get('valid_docs', 0):,}",
        f"- **Words:** {validation['total_words']:,}",
        f"- **Characters:** {validation['total_characters']:,}",
        f"- **Duplicate IDs:** {validation['duplicate_ids']:,}",
        f"- **Duplicate texts:** {validation['duplicate_texts']:,}",
        f"- **English estimate:** {english['english_fraction']:.3%} "
        f"({english['sample_size']:,}-document sample)",
        "",
        "## Document length (words)",
        "",
        "| min | p10 | median | mean | p90 | p99 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {lengths['min']:,} | {lengths['p10']:,.0f} | "
            f"{lengths['median']:,.0f} | {lengths['mean']:,.1f} | "
            f"{lengths['p90']:,.0f} | {lengths['p99']:,.0f} | "
            f"{lengths['max']:,} |"
        ),
        "",
        "## Leakage",
        "",
    ]
    if leakage:
        lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(leakage.items()))
    else:
        lines.append("- No leakage detected.")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| source | documents | JSONL bytes | words |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, values in validation["source_distribution"].items():
        lines.append(
            f"| {source} | {values['docs']:,} | {values['jsonl_bytes']:,} | "
            f"{values['words']:,} |"
        )
    lines.extend(
        [
            "",
            f"100 readable samples: `{corpus.with_suffix('.samples.txt')}`.",
            "",
        ]
    )
    return "\n".join(lines)


def training_commands(
    endpoint: str, bucket: str, object_name: str, checkpoint: Path
) -> str:
    archive_name = Path(object_name).name
    return f"""# Rented CUDA server commands

Set credentials without putting them in shell history:

```bash
export MINIO_ENDPOINT={endpoint!r}
export MINIO_ACCESS_KEY='...'
export MINIO_SECRET_KEY='...'
export MINIO_BUCKET={bucket!r}
```

Install MinIO client and download:

```bash
curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc
chmod +x /usr/local/bin/mc
mc alias set backtalk "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc cp --continue "backtalk/$MINIO_BUCKET/{object_name}" .
mc cp "backtalk/$MINIO_BUCKET/{object_name}.sha256" .
shasum -a 256 -c {archive_name}.sha256 || sha256sum -c {archive_name}.sha256
unzip -q {archive_name}
test -s data/corpus/backtalk_pretrain_10gb_reversed.jsonl
test -s {checkpoint}
ls -lh data/corpus/backtalk_pretrain_10gb_reversed.jsonl
shasum -a 256 -c data/corpus/backtalk_pretrain_10gb_reversed.sha256 || \\
  sha256sum -c data/corpus/backtalk_pretrain_10gb_reversed.sha256
```

The reversed file is the training input. The normal clean corpus is reference/QA only.
Prepare tokenized data and install the included step-20,000 resume checkpoint:

```bash
python3 -m pip install -r requirements.txt
python3 train/cli.py prepare \\
  --src data/corpus/backtalk_pretrain_10gb_reversed.jsonl \\
  --tokenizer tokenizers/backtalk-tokenizer/tokenizer.json \\
  --out_dir data/prepared/backtalk-ctx2048 \\
  --text_field text --val_frac 0.005 --context_length 2048 \\
  --dtype uint32 --force
mkdir -p checkpoints/backtalk
cp {checkpoint} checkpoints/backtalk/ckpt.pt

# Resume sanity run: advances the existing checkpoint from step 20,000 to 20,500.
python3 train/cli.py resume --config configs/pretrain_cuda.json --max_steps 20500

# Continue the same run to step 50,000 (weights + optimizer state are restored).
CONFIG=configs/pretrain_cuda.json bash scripts/run_pretrain_cuda.sh \\
  loss --resume --max-steps 50000
```

Upload server results:

```bash
zip -1 -r backtalk_results.zip checkpoints logs data/prepared/backtalk-ctx2048/meta.json
shasum -a 256 backtalk_results.zip > backtalk_results.zip.sha256 || \\
  sha256sum backtalk_results.zip > backtalk_results.zip.sha256
mc cp --continue backtalk_results.zip \\
  "backtalk/$MINIO_BUCKET/results/backtalk_results.zip"
mc cp backtalk_results.zip.sha256 \\
  "backtalk/$MINIO_BUCKET/results/backtalk_results.zip.sha256"
mc stat "backtalk/$MINIO_BUCKET/results/backtalk_results.zip"
```
"""


def required_paths(
    corpus: Path,
    reversed_corpus: Path,
    sft: Path | None,
    checkpoint: Path,
) -> list[Path]:
    paths = [
        corpus,
        corpus.with_suffix(".stats.json"),
        corpus.with_suffix(".progress.json"),
        corpus.with_suffix(".samples.txt"),
        corpus.with_suffix(".report.md"),
        reversed_corpus,
        reversed_corpus.with_suffix(".stats.json"),
        reversed_corpus.with_suffix(".progress.json"),
        reversed_corpus.with_suffix(".samples.txt"),
        reversed_corpus.with_suffix(".sha256"),
        checkpoint,
        checkpoint.with_name("CHECKPOINT_INFO.json"),
        Path("data/SFT_FINAL_REPORT.md"),
        Path("requirements.txt"),
        Path("README.md"),
        Path("train"),
        Path("configs"),
        Path("tokenizers/backtalk-tokenizer"),
        Path("scripts/clean_pretraining_corpus.py"),
        Path("scripts/download_pretraining_corpus.py"),
        Path("scripts/minio_transfer.py"),
        Path("scripts/prepare_reversed_training_data.py"),
        Path("scripts/package_pretraining_ready.py"),
        Path("scripts/repair_corpus_leakage.py"),
        Path("scripts/run_pretrain_cuda.sh"),
        Path("TRAINING_COMMANDS.md"),
    ]
    if sft is not None:
        paths.extend(
            [
                sft,
                sft.with_suffix(".stats.json"),
                sft.with_suffix(".progress.json"),
                sft.with_suffix(".samples.txt"),
                sft.with_suffix(".sha256"),
            ]
        )
    return paths


def expand_paths(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file()
                and "__pycache__" not in child.parts
                and child.suffix not in {".pyc", ".DS_Store"}
            )
        elif path.is_file():
            files.append(path)
        else:
            raise RuntimeError(f"required archive path is missing: {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/corpus_10gb_clean.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/backtalk_pretraining_ready.zip"),
    )
    parser.add_argument(
        "--reversed-corpus",
        type=Path,
        default=Path("data/corpus/backtalk_pretrain_10gb_reversed.jsonl"),
    )
    parser.add_argument(
        "--reversed-sft",
        type=Path,
        default=Path("data/sft_reversed.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="latest pretraining checkpoint with optimizer state for resume",
    )
    parser.add_argument("--bucket", default="backtalk")
    parser.add_argument(
        "--object",
        default="pretraining/backtalk_pretraining_ready.zip",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MINIO_ENDPOINT"),
        help="S3/MinIO endpoint (or set MINIO_ENDPOINT)",
    )
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or MINIO_ENDPOINT is required")

    stats_path = args.corpus.with_suffix(".stats.json")
    if not stats_path.exists():
        raise RuntimeError(f"missing validation stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("verdict") != "PASS":
        raise RuntimeError(f"refusing to package validation verdict {stats.get('verdict')}")
    validation = stats.get("validation", {})
    if validation.get("file_size_bytes") != args.corpus.stat().st_size:
        raise RuntimeError("corpus size differs from validated size")
    if sha256_file(args.corpus) != validation.get("sha256"):
        raise RuntimeError("corpus SHA-256 differs from validated checksum")
    if not args.checkpoint.is_file():
        raise RuntimeError(f"missing resume checkpoint: {args.checkpoint}")

    reversed_stats_path = args.reversed_corpus.with_suffix(".stats.json")
    if not reversed_stats_path.exists():
        raise RuntimeError(f"missing reversed validation stats: {reversed_stats_path}")
    reversed_stats = json.loads(reversed_stats_path.read_text(encoding="utf-8"))
    reversed_validation = reversed_stats.get("validation", {})
    if reversed_stats.get("verdict") != "PASS":
        raise RuntimeError("refusing to package reversed pretraining without PASS")
    if reversed_validation.get("source_sha256") != validation.get("sha256"):
        raise RuntimeError("reversed pretraining was not built from validated clean corpus")
    if reversed_validation.get("source_lines") != validation["counts"].get("valid_docs"):
        raise RuntimeError("normal and reversed pretraining line counts differ")
    reversed_checksum = sha256_file(args.reversed_corpus)
    if reversed_checksum != reversed_validation.get("reversed_sha256"):
        raise RuntimeError("reversed pretraining SHA-256 differs from validation")

    sft: Path | None = args.reversed_sft if Path("data/sft_raw.jsonl").exists() else None
    sft_stats: dict[str, Any] | None = None
    if sft is not None:
        sft_stats_path = sft.with_suffix(".stats.json")
        if not sft.exists() or not sft_stats_path.exists():
            raise RuntimeError("SFT source exists but reversed SFT or stats are missing")
        sft_stats = json.loads(sft_stats_path.read_text(encoding="utf-8"))
        sft_validation = sft_stats.get("validation", {})
        if sft_stats.get("verdict") != "PASS":
            raise RuntimeError("refusing to package reversed SFT without PASS")
        if sft_validation.get("roundtrip_mismatches") != 0:
            raise RuntimeError("reversed SFT has round-trip mismatches")
        if sha256_file(sft) != sft_validation.get("reversed_sha256"):
            raise RuntimeError("reversed SFT SHA-256 differs from validation")

    report_path = args.corpus.with_suffix(".report.md")
    report_path.write_text(corpus_report(stats, args.corpus), encoding="utf-8")
    commands_path = Path("TRAINING_COMMANDS.md")
    commands_path.write_text(
        training_commands(args.endpoint, args.bucket, args.object, args.checkpoint),
        encoding="utf-8",
    )

    files = expand_paths(
        required_paths(args.corpus, args.reversed_corpus, sft, args.checkpoint)
    )
    manifest_files = {}
    for path in files:
        if path == args.corpus:
            checksum = validation["sha256"]
        else:
            checksum = sha256_file(path)
        manifest_files[str(path)] = {
            "size": path.stat().st_size,
            "sha256": checksum,
        }
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "PASS",
        "main_pretraining_file": str(args.reversed_corpus),
        "sft_training_file": str(sft) if sft is not None else None,
        "reference_only_corpus": str(args.corpus),
        "resume_checkpoint": str(args.checkpoint),
        "resume_checkpoint_sha256": manifest_files[str(args.checkpoint)]["sha256"],
        "resume_checkpoint_step": 20000,
        "resume_checkpoint_contains_optimizer": True,
        "notice": (
            "Train on main_pretraining_file. The normal clean corpus is source/"
            "reference/QA only."
        ),
        "archive_object": f"{args.bucket}/{args.object}",
        "files": manifest_files,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    total = sum(path.stat().st_size for path in files)
    progress = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1000, desc="zip")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for path in files:
                archive.write(path, arcname=str(path))
                progress.update(path.stat().st_size)
            archive.writestr(
                "PRETRAINING_MANIFEST.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
    finally:
        progress.close()
    os.replace(temporary, args.output)

    with zipfile.ZipFile(args.output, "r") as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise RuntimeError(f"ZIP CRC validation failed at {bad_file}")
    archive_checksum = sha256_file(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{archive_checksum}  {args.output.name}\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "archive": str(args.output),
        "size": args.output.stat().st_size,
        "sha256": archive_checksum,
        "sidecar": str(sidecar),
        "files": len(files) + 1,
        "object": f"{args.bucket}/{args.object}",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
