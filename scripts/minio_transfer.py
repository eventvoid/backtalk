#!/usr/bin/env python3
"""Fast verified MinIO upload/download utility using boto3 multipart transfers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing boto3. Run: python3 -m pip install -r requirements.txt") from exc

from tqdm import tqdm


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def s3_etag(path: Path, part_size: int) -> str:
    part_digests = []
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(part_size)
            if not chunk:
                break
            part_digests.append(hashlib.md5(chunk).digest())
    if not part_digests:
        return hashlib.md5(b"").hexdigest()
    if len(part_digests) == 1:
        return part_digests[0].hex()
    return f"{hashlib.md5(b''.join(part_digests)).hexdigest()}-{len(part_digests)}"


def worker_count(value: str) -> int:
    if value == "auto":
        return min(32, max(8, (os.cpu_count() or 4) * 2))
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be positive")
    return workers


class ProgressCallback:
    def __init__(self, total: int, description: str) -> None:
        self.lock = threading.Lock()
        self.progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
            desc=description,
            dynamic_ncols=True,
        )

    def __call__(self, transferred: int) -> None:
        with self.lock:
            self.progress.update(transferred)

    def close(self) -> None:
        self.progress.close()


def make_client(args: argparse.Namespace) -> Any:
    load_dotenv()
    endpoint = args.endpoint or env_first(
        "MINIO_ENDPOINT", "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL"
    )
    access_key = env_first("MINIO_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
    secret_key = env_first("MINIO_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
    session_token = env_first("MINIO_SESSION_TOKEN", "AWS_SESSION_TOKEN")
    if not endpoint:
        raise RuntimeError("MINIO_ENDPOINT is not set")
    if not access_key or not secret_key:
        raise RuntimeError("MinIO access/secret key variables are not set")
    if "://" not in endpoint:
        secure = parse_bool(env_first("MINIO_SECURE"), True)
        endpoint = ("https://" if secure else "http://") + endpoint
    verify_ssl = parse_bool(env_first("MINIO_VERIFY_SSL"), True)
    config = Config(
        signature_version="s3v4",
        max_pool_connections=max(args.workers + 4, 16),
        retries={"max_attempts": 8, "mode": "adaptive"},
        s3={"addressing_style": "path"},
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint.rstrip("/"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        verify=verify_ssl,
        region_name=env_first("MINIO_REGION", "AWS_DEFAULT_REGION") or "us-east-1",
        config=config,
    )


def transfer_config(args: argparse.Namespace) -> TransferConfig:
    part_size = args.part_size_mb * 1024 * 1024
    return TransferConfig(
        multipart_threshold=part_size,
        multipart_chunksize=part_size,
        max_concurrency=args.workers,
        use_threads=True,
    )


def bucket_exists(client: Any, bucket: str) -> bool:
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status in {403, 404}:
            return False
        raise


def ensure_bucket(client: Any, bucket: str, create: bool) -> None:
    if bucket_exists(client, bucket):
        return
    if not create:
        raise RuntimeError(
            f"bucket {bucket!r} does not exist or is inaccessible; use --create-bucket"
        )
    client.create_bucket(Bucket=bucket)


def head_info(client: Any, bucket: str, object_name: str) -> dict[str, Any]:
    response = client.head_object(Bucket=bucket, Key=object_name)
    return {
        "bucket": bucket,
        "object": object_name,
        "size": int(response["ContentLength"]),
        "etag": response.get("ETag", "").strip('"'),
        "sha256": response.get("Metadata", {}).get("sha256"),
        "content_type": response.get("ContentType"),
        "last_modified": response.get("LastModified").isoformat()
        if response.get("LastModified")
        else None,
    }


def upload(
    client: Any,
    args: argparse.Namespace,
    local_path: Path,
    bucket: str,
    object_name: str,
) -> dict[str, Any]:
    if not local_path.is_file():
        raise RuntimeError(f"local file does not exist: {local_path}")
    ensure_bucket(client, bucket, args.create_bucket)
    size = local_path.stat().st_size
    checksum = sha256_file(local_path)
    expected_etag = s3_etag(local_path, args.part_size_mb * 1024 * 1024)
    callback = ProgressCallback(size, "upload")
    try:
        client.upload_file(
            str(local_path),
            bucket,
            object_name,
            ExtraArgs={
                "Metadata": {"sha256": checksum},
                "ContentType": "application/zip",
            },
            Config=transfer_config(args),
            Callback=callback,
        )
    finally:
        callback.close()
    remote = head_info(client, bucket, object_name)
    if remote["size"] != size:
        raise RuntimeError(f"remote size mismatch: {remote['size']} != {size}")
    metadata_matches = remote["sha256"] == checksum
    etag_matches = remote["etag"] == expected_etag
    if not metadata_matches and not etag_matches:
        raise RuntimeError(
            "remote checksum verification failed: neither SHA-256 metadata nor "
            "the multipart ETag matched"
        )
    return {
        "status": "verified",
        "local": str(local_path),
        "local_size": size,
        "local_sha256": checksum,
        "expected_etag": expected_etag,
        "verification": "sha256-metadata" if metadata_matches else "multipart-etag",
        "remote": remote,
    }


def download(
    client: Any,
    args: argparse.Namespace,
    bucket: str,
    object_name: str,
    local_path: Path,
) -> dict[str, Any]:
    remote = head_info(client, bucket, object_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_name(local_path.name + ".part")
    callback = ProgressCallback(remote["size"], "download")
    try:
        client.download_file(
            bucket,
            object_name,
            str(partial),
            Config=transfer_config(args),
            Callback=callback,
        )
    finally:
        callback.close()
    checksum = sha256_file(partial)
    expected_etag = s3_etag(partial, args.part_size_mb * 1024 * 1024)
    if partial.stat().st_size != remote["size"]:
        raise RuntimeError("downloaded size does not match remote size")
    if remote["sha256"]:
        if checksum != remote["sha256"]:
            raise RuntimeError("downloaded SHA-256 does not match remote metadata")
        verification = "sha256-metadata"
    elif expected_etag == remote["etag"]:
        verification = "multipart-etag"
    else:
        raise RuntimeError("downloaded file does not match the remote multipart ETag")
    os.replace(partial, local_path)
    return {
        "status": "verified",
        "local": str(local_path),
        "local_size": local_path.stat().st_size,
        "local_sha256": checksum,
        "verification": verification,
        "remote": remote,
    }


def test_round_trip(
    client: Any, args: argparse.Namespace, bucket: str
) -> dict[str, Any]:
    ensure_bucket(client, bucket, args.create_bucket)
    object_name = f"_health/backtalk-transfer-{uuid.uuid4().hex}.bin"
    with tempfile.TemporaryDirectory(prefix="backtalk-minio-test-") as directory:
        source = Path(directory) / "source.bin"
        destination = Path(directory) / "download.bin"
        source.write_bytes(os.urandom(args.test_size_mb * 1024 * 1024))
        source_checksum = sha256_file(source)
        expected_etag = s3_etag(source, args.part_size_mb * 1024 * 1024)
        callback = ProgressCallback(source.stat().st_size, "test upload")
        try:
            client.upload_file(
                str(source),
                bucket,
                object_name,
                ExtraArgs={
                    "Metadata": {"sha256": source_checksum},
                    "ContentType": "application/octet-stream",
                },
                Config=transfer_config(args),
                Callback=callback,
            )
        finally:
            callback.close()
        remote = head_info(client, bucket, object_name)
        callback = ProgressCallback(remote["size"], "test download")
        try:
            client.download_file(
                bucket,
                object_name,
                str(destination),
                Config=transfer_config(args),
                Callback=callback,
            )
        finally:
            callback.close()
        downloaded_checksum = sha256_file(destination)
        client.delete_object(Bucket=bucket, Key=object_name)
    if source_checksum != downloaded_checksum:
        raise RuntimeError("test upload/download checksum mismatch")
    metadata_matches = remote["sha256"] == source_checksum
    etag_matches = remote["etag"] == expected_etag
    if not metadata_matches and not etag_matches:
        raise RuntimeError("test remote checksum/ETag mismatch")
    return {
        "status": "PASS",
        "bucket": bucket,
        "test_object_deleted": object_name,
        "size": remote["size"],
        "sha256": source_checksum,
        "etag": remote["etag"],
        "verification": "sha256-metadata" if metadata_matches else "multipart-etag",
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="override MINIO_ENDPOINT")
    parser.add_argument(
        "--bucket",
        default=env_first("MINIO_BUCKET", "S3_BUCKET") or "backtalk",
    )
    parser.add_argument("--workers", type=worker_count, default=worker_count("auto"))
    parser.add_argument("--part-size-mb", type=int, default=64)
    parser.add_argument("--create-bucket", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("local", type=Path)
    upload_parser.add_argument("object")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("object")
    download_parser.add_argument("local", type=Path)

    stat_parser = subparsers.add_parser("stat")
    stat_parser.add_argument("object")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--test-size-mb", type=int, default=2)
    return parser


def main() -> int:
    load_dotenv()
    parser = make_parser()
    args = parser.parse_args()
    if args.part_size_mb < 5:
        parser.error("--part-size-mb must be at least 5")
    client = make_client(args)
    if args.command == "upload":
        result = upload(client, args, args.local, args.bucket, args.object)
    elif args.command == "download":
        result = download(client, args, args.bucket, args.object, args.local)
    elif args.command == "stat":
        result = head_info(client, args.bucket, args.object)
    else:
        result = test_round_trip(client, args, args.bucket)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
