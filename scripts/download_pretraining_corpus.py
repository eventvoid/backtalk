#!/usr/bin/env python3
"""Stream, clean, deduplicate, and interleave an English pretraining corpus.

The default 10 GB mix is:
  fineweb-edu 7.0 GB, English Wikipedia 1.2 GB, Simple Wikipedia 0.5 GB,
  SmolLM Cosmopedia v2 0.8 GB, OpenWebMath 0.3 GB, Stack Exchange 0.2 GB.

Sizes are decimal GB and include the UTF-8 JSONL record framing. Raw datasets
are streamed rather than materialized. The Hugging Face streaming cache and a
persistent SQLite deduplication index live under --cache-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - handled before starting a download
    tqdm = None

GB = 1_000_000_000
MASK64 = (1 << 64) - 1
MIN_WORDS = 50
MAX_WORDS = 40_000
PROGRESS_INTERVAL = 1.0
CHECKPOINT_INTERVAL = 5.0

SOURCE_WEIGHTS = {
    "fineweb-edu": 7.0,
    "wikipedia": 1.2,
    "simple-wikipedia": 0.5,
    "smollm-edu": 0.8,
    "open-web-math": 0.3,
    "stackexchange": 0.2,
}
SOURCE_ALIASES = {
    "fineweb": "fineweb-edu",
    "wiki": "wikipedia",
    "simplewiki": "simple-wikipedia",
    "simple-wiki": "simple-wikipedia",
    "smollm": "smollm-edu",
    "math": "open-web-math",
    "stack": "stackexchange",
}

# Deliberately excludes programming-focused Stack Exchange communities.
STACK_CONFIGS = (
    "english", "academia", "history", "cooking", "diy", "law", "biology",
    "chemistry", "astronomy", "earthscience", "economics", "gardening",
    "fitness", "parenting", "travel", "workplace", "writers", "literature",
    "philosophy", "politics", "outdoors", "pets", "bicycles", "music",
    "movies", "worldbuilding", "skeptics", "space", "aviation", "mechanics",
)

STOPWORDS = frozenset(
    """the of and a to in is was for it with as on be at by this had not are
    but from or have an they which one you were all there would their we been
    has when who will more no if out so what up its about into than them can
    only other some could time these two may then do first any now such like
    our over even most made after also did many before through years where much
    your way well should because each those people how good very make world see
    work get here between both life under day same another know while last
    might us great year come since go right used take use during without again
    place however found part high every does number course something fact
    though public enough system better called find going later important
    example different often information question answer""".split()
)

TAG_LIKE_RE = re.compile(r"<[!/a-zA-Z][^>]{0,1000}>")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NORMAL_WORD_RE = re.compile(r"[a-z0-9]+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
WIKI_CITE_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]{6,}\|?\s*$")
CODE_START_RE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |#include|using namespace|"
    r"public (?:class|static)|private |protected |function |func |const |let |"
    r"var |SELECT\s+.+\s+FROM|INSERT\s+INTO|(?:if|for|while)\s*\()",
    re.I,
)
CODE_TOKEN_RE = re.compile(
    r"(?:=>|::|&&|\|\||</?[a-z][^>]*>|\b(?:printf|console\.log|System\.out)\s*\()",
    re.I,
)
BOILERPLATE_RE = re.compile(
    r"\b(?:cookie policy|privacy policy|terms of (?:use|service)|all rights "
    r"reserved|subscribe to|sign[ -]?in|log[ -]?in|click here|read more|"
    r"share this|leave a (?:comment|reply)|related (?:posts|articles)|"
    r"advertisement|skip to (?:main )?content|accept cookies|enable javascript|"
    r"newsletter|follow us|contact us)\b",
    re.I,
)
STOP_SECTIONS = frozenset(
    {
        "references", "external links", "see also", "further reading", "notes",
        "bibliography", "citations", "sources", "footnotes", "gallery",
        "related pages", "other websites", "notes and references",
    }
)
PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
        "\u00a0": " ", "\u200b": "", "\u200c": "", "\u200d": "",
        "\ufeff": "",
    }
)


class TextExtractor(HTMLParser):
    """Extract visible prose while dropping code, tables, scripts, and styles."""

    DROP_TAGS = {"script", "style", "code", "pre", "table", "svg", "math"}
    BREAK_TAGS = {
        "p", "div", "br", "li", "article", "section", "header", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.DROP_TAGS:
            self.depth += 1
        elif not self.depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.DROP_TAGS and self.depth:
            self.depth -= 1
        elif not self.depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


@dataclass(frozen=True)
class Candidate:
    source: str
    doc_id: str
    text: str
    exact: bytes
    simhash: int


@dataclass(frozen=True)
class ProducerDone:
    source: str
    error: str | None = None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_sources(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(SOURCE_WEIGHTS)
    result: list[str] = []
    for raw in value.split(","):
        name = SOURCE_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name not in SOURCE_WEIGHTS:
            choices = ", ".join(SOURCE_WEIGHTS)
            raise argparse.ArgumentTypeError(f"unknown source {raw!r}; choose from {choices}")
        if name not in result:
            result.append(name)
    if not result:
        raise argparse.ArgumentTypeError("--sources cannot be empty")
    return result


def source_quotas(sources: list[str], target_bytes: int) -> dict[str, int]:
    selected_weight = sum(SOURCE_WEIGHTS[name] for name in sources)
    quotas: dict[str, int] = {}
    allocated = 0
    for name in sources[:-1]:
        quota = int(target_bytes * SOURCE_WEIGHTS[name] / selected_weight)
        quotas[name] = quota
        allocated += quota
    quotas[sources[-1]] = target_bytes - allocated
    return quotas


def strip_markup(raw: str) -> str:
    raw = html.unescape(raw)
    if TAG_LIKE_RE.search(raw):
        parser = TextExtractor()
        try:
            parser.feed(raw)
            parser.close()
            raw = parser.text()
        except Exception:
            raw = TAG_LIKE_RE.sub(" ", raw)
    raw = MARKDOWN_IMAGE_RE.sub(" ", raw)
    raw = MARKDOWN_LINK_RE.sub(r"\1", raw)
    raw = WIKI_CITE_RE.sub("", raw)
    return raw


def line_is_code(line: str) -> bool:
    if line.startswith(("```", "~~~")):
        return True
    if CODE_START_RE.search(line) or CODE_TOKEN_RE.search(line):
        return True
    if line.count("{") + line.count("}") >= 2:
        return True
    if len(line) > 20:
        symbols = sum(ch in "{}[]<>;`\\=_" for ch in line)
        if symbols / len(line) > 0.14:
            return True
    return False


def line_is_table(line: str) -> bool:
    if line.count("|") >= 3 or line.count("\t") >= 2:
        return True
    if TABLE_SEP_RE.match(line) and ("|" in line or "-" in line):
        return True
    return False


def clean_document(raw: Any, source: str) -> tuple[str | None, str | None]:
    if not isinstance(raw, str) or len(raw) < 250:
        return None, "too_short_raw"
    if raw.count("\ufffd") > max(2, len(raw) // 2000):
        return None, "broken_unicode"
    try:
        raw.encode("utf-8", "strict")
    except UnicodeError:
        return None, "broken_unicode"

    text = strip_markup(raw)
    text = unicodedata.normalize("NFKC", text.translate(PUNCT_TRANSLATION))
    text = CTRL_RE.sub("", text)

    kept: list[str] = []
    code_lines = 0
    content_lines = 0
    fenced = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            code_lines += 1
            continue
        if fenced or not line:
            if not fenced and kept and kept[-1] != "":
                kept.append("")
            continue
        content_lines += 1
        heading = HEADING_RE.sub("", line).strip()
        if source in {"wikipedia", "simple-wikipedia"} and heading.lower() in STOP_SECTIONS:
            break
        line = LIST_RE.sub("", heading)
        if not line or line_is_table(line):
            continue
        if line_is_code(line):
            code_lines += 1
            continue
        if len(line.split()) <= 24 and BOILERPLATE_RE.search(line):
            continue
        if len(line) > 300 and " " not in line:
            continue
        line = URL_RE.sub("", line)
        line = SPACE_RE.sub(" ", line).strip()
        if line:
            kept.append(line)

    if content_lines and code_lines / content_lines > 0.20:
        return None, "code"
    text = BLANK_RE.sub("\n\n", "\n".join(kept)).strip()
    words = WORD_RE.findall(text)
    count = len(words)
    if count < MIN_WORDS:
        return None, "too_short"
    if count > MAX_WORDS:
        return None, "too_long"

    nonspace = sum(not ch.isspace() for ch in text)
    letters = [ch for ch in text if ch.isalpha()]
    if not nonspace or len(letters) / nonspace < 0.62:
        return None, "low_alpha"
    latin = sum(ord(ch) < 0x0250 for ch in letters)
    if not letters or latin / len(letters) < 0.92:
        return None, "non_english_script"
    lower_words = [word.lower().strip("'") for word in words]
    stop_count = sum(word in STOPWORDS for word in lower_words)
    if stop_count < 3 or stop_count / count < 0.045:
        return None, "non_english"
    if sum(ch.isdigit() for ch in text) / nonspace > 0.18:
        return None, "too_many_digits"
    if sum(len(word) > 40 for word in words) > max(2, count // 100):
        return None, "broken_tokens"
    if sum(word.isupper() and len(word) > 1 for word in words) / count > 0.22:
        return None, "too_many_caps"
    lines = [line for line in text.splitlines() if line]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.55:
        return None, "repetition"
    if len(lower_words) >= 100:
        common = Counter(lower_words).most_common(1)[0][1]
        if common / count > 0.14:
            return None, "repetition"
    return text, None


def fingerprints(text: str) -> tuple[bytes, int]:
    tokens = NORMAL_WORD_RE.findall(text.lower())
    normalized = " ".join(tokens)
    exact = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()
    if len(tokens) < 3:
        return exact, int.from_bytes(exact[:8], "big")

    # Sample at most 160 evenly distributed 3-word shingles. This catches
    # lightly edited copies without making fingerprinting dominate runtime.
    total = len(tokens) - 2
    step = max(1, total // 160)
    weights = [0] * 64
    for index in range(0, total, step):
        shingle = " ".join(tokens[index:index + 3]).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    simhash = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            simhash |= 1 << bit
    return exact, simhash


def signed64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def unsigned64(value: int) -> int:
    return value & MASK64


class DedupIndex:
    def __init__(self, path: Path, reset: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                exact BLOB PRIMARY KEY,
                simhash INTEGER NOT NULL,
                b0 INTEGER NOT NULL,
                b1 INTEGER NOT NULL,
                b2 INTEGER NOT NULL,
                b3 INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS docs_b0 ON docs(b0);
            CREATE INDEX IF NOT EXISTS docs_b1 ON docs(b1);
            CREATE INDEX IF NOT EXISTS docs_b2 ON docs(b2);
            CREATE INDEX IF NOT EXISTS docs_b3 ON docs(b3);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self.db.commit()
        self.pending = 0

    @staticmethod
    def bands(value: int) -> tuple[int, int, int, int]:
        return tuple((value >> shift) & 0xFFFF for shift in (0, 16, 32, 48))  # type: ignore[return-value]

    def classify(self, exact: bytes, simhash: int) -> str | None:
        if self.db.execute("SELECT 1 FROM docs WHERE exact=?", (exact,)).fetchone():
            return "duplicate_exact"
        bands = self.bands(simhash)
        rows = self.db.execute(
            """SELECT simhash FROM docs
               WHERE b0=? OR b1=? OR b2=? OR b3=? LIMIT 512""",
            bands,
        )
        for (other_signed,) in rows:
            # bin().count() keeps the CLI compatible with the system Python
            # 3.9 shipped by older macOS/Xcode installations.
            if bin(simhash ^ unsigned64(other_signed)).count("1") <= 3:
                return "duplicate_near"
        return None

    def add(self, exact: bytes, simhash: int) -> None:
        self.db.execute(
            "INSERT INTO docs(exact,simhash,b0,b1,b2,b3) VALUES(?,?,?,?,?,?)",
            (exact, signed64(simhash), *self.bands(simhash)),
        )
        self.pending += 1

    def set_meta(self, **values: Any) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            [(key, json.dumps(value)) for key, value in values.items()],
        )

    def meta(self) -> dict[str, Any]:
        return {key: json.loads(value) for key, value in self.db.execute("SELECT key,value FROM meta")}

    def commit(self) -> None:
        self.db.commit()
        self.pending = 0

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def upstream_id(row: dict[str, Any], source: str, text: str, cursor: int) -> str:
    for key in ("id", "pageid", "url"):
        value = row.get(key)
        if value is not None and str(value).strip():
            raw = str(value).strip()
            digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=10).hexdigest()
            return f"{source}:{digest}"
    digest = hashlib.blake2s(text[:4096].encode("utf-8"), digest_size=10).hexdigest()
    return f"{source}:{cursor:x}:{digest}"


def row_text(row: dict[str, Any], source: str) -> str:
    if source == "stackexchange":
        question = row.get("title_body") or row.get("title") or ""
        answer = row.get("upvoted_answer") or row.get("answer") or ""
        return f"{question}\n\n{answer}".strip()
    return row.get("text") or row.get("content") or ""


def hf_stream(
    source: str,
    cache_dir: Path,
    seed: int,
    cursor: dict[str, Any],
    update_cursor: Any,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install project requirements with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    common = {
        "split": "train",
        "streaming": True,
        "cache_dir": str(cache_dir),
    }
    specifications: list[tuple[str, str | None]]
    if source == "fineweb-edu":
        specifications = [("HuggingFaceFW/fineweb-edu", "sample-10BT")]
    elif source == "wikipedia":
        specifications = [("wikimedia/wikipedia", "20231101.en")]
    elif source == "simple-wikipedia":
        specifications = [("wikimedia/wikipedia", "20231101.simple")]
    elif source == "smollm-edu":
        # Cosmopedia is prose-only educational data; python-edu is excluded.
        specifications = [("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")]
    elif source == "open-web-math":
        specifications = [("open-web-math/open-web-math", None)]
    else:
        specifications = [
            (
                "flax-sentence-embeddings/"
                "stackexchange_titlebody_best_voted_answer_jsonl",
                config,
            )
            for config in STACK_CONFIGS
        ]

    config_start = int(cursor.get("config_index", 0))
    for config_index in range(config_start, len(specifications)):
        repo, config = specifications[config_index]
        skip = int(cursor.get("config_scanned", 0)) if config_index == config_start else 0
        if source == "stackexchange":
            # This mirror's legacy loading script is rejected by datasets 4.x.
            # Hugging Face publishes the same data as auto-converted Parquet;
            # query that first-party endpoint and load the Parquet directly.
            endpoint = (
                "https://huggingface.co/api/datasets/"
                f"{repo}/parquet/{urllib.parse.quote(config or '', safe='')}/train"
            )
            headers = {"User-Agent": "backtalk-corpus-builder/1.0"}
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(endpoint, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                converted_files = json.load(response)
            if not converted_files:
                raise RuntimeError(f"no Parquet files returned for Stack Exchange config {config}")
            # datasets 4.x mistakes the API proxy URLs for hf:// paths. Use
            # the equivalent public refs/convert/parquet branch URLs instead.
            converted_base = (
                f"https://huggingface.co/datasets/{repo}/resolve/"
                f"refs%2Fconvert%2Fparquet/{urllib.parse.quote(config or '', safe='')}/train"
            )
            parquet_urls = [
                f"{converted_base}/{index:04d}.parquet"
                for index in range(len(converted_files))
            ]
            dataset = load_dataset("parquet", data_files=parquet_urls, **common)
        else:
            dataset = load_dataset(repo, config, **common)
        # Buffered shuffling is deterministic across reconnects and avoids
        # taking only the beginning of large source shards.
        dataset = dataset.shuffle(seed=seed + config_index, buffer_size=10_000)
        if skip:
            dataset = dataset.skip(skip)
        scanned = skip
        for row in dataset:
            scanned += 1
            update_cursor(config_index, scanned, True)
            yield row
        update_cursor(config_index + 1, 0, False)


def producer(
    source: str,
    source_index: int,
    cache_dir: Path,
    seed: int,
    out_queue: queue.Queue[Candidate | ProducerDone],
    stats: dict[str, dict[str, Any]],
    lock: threading.Lock,
    stop_all: threading.Event,
    stop_source: threading.Event,
) -> None:
    retries = 0
    max_retries = 8

    def put(item: Candidate | ProducerDone) -> bool:
        while not stop_all.is_set() and not stop_source.is_set():
            try:
                out_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def update_cursor(config_index: int, config_scanned: int, scanned_row: bool) -> None:
        with lock:
            stats[source]["cursor"] = {
                "config_index": config_index,
                "config_scanned": config_scanned,
            }
            if scanned_row:
                stats[source]["scanned"] += 1

    while not stop_all.is_set() and not stop_source.is_set():
        with lock:
            cursor = dict(stats[source].get("cursor", {}))
        try:
            for row in hf_stream(source, cache_dir, seed + source_index * 1009, cursor, update_cursor):
                if stop_all.is_set() or stop_source.is_set():
                    break
                raw = row_text(row, source)
                text, reason = clean_document(raw, source)
                if reason:
                    with lock:
                        state = stats[source]
                        state["rejected"] += 1
                        state["reject_reasons"][reason] = state["reject_reasons"].get(reason, 0) + 1
                    continue
                exact, simhash = fingerprints(text)
                with lock:
                    stats[source]["cleaned"] += 1
                    cursor_num = stats[source]["scanned"]
                item = Candidate(
                    source=source,
                    doc_id=upstream_id(row, source, text, cursor_num),
                    text=text,
                    exact=exact,
                    simhash=simhash,
                )
                if not put(item):
                    return
            put(ProducerDone(source))
            return
        except Exception as exc:
            retries += 1
            with lock:
                stats[source]["reconnects"] += 1
                stats[source]["last_error"] = f"{type(exc).__name__}: {exc}"
            if retries > max_retries:
                put(ProducerDone(source, stats[source]["last_error"]))
                return
            stop_all.wait(min(60.0, 2.0 ** retries))


def initial_source_stats(
    sources: list[str],
    quotas: dict[str, int],
    progress: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    previous = (progress or {}).get("per_source", {})
    result: dict[str, dict[str, Any]] = {}
    for name in sources:
        old = previous.get(name, {})
        result[name] = {
            "quota_bytes": quotas[name],
            "scanned": int(old.get("scanned", 0)),
            "cleaned": int(old.get("cleaned", 0)),
            "accepted": int(old.get("accepted", 0)),
            "rejected": int(old.get("rejected", 0)),
            "output_bytes": int(old.get("output_bytes", 0)),
            "text_bytes": int(old.get("text_bytes", 0)),
            "words": int(old.get("words", 0)),
            "reject_reasons": dict(old.get("reject_reasons", {})),
            "cursor": dict(old.get("cursor", {"config_index": 0, "config_scanned": 0})),
            "reconnects": int(old.get("reconnects", 0)),
            "last_error": old.get("last_error"),
            "exhausted": False,
        }
    return result


def snapshot(
    out_path: Path,
    target_bytes: int,
    sources: list[str],
    stats: dict[str, dict[str, Any]],
    started_at: float,
    initial_elapsed: float,
    seed: int,
    complete: bool = False,
) -> dict[str, Any]:
    elapsed = initial_elapsed + time.monotonic() - started_at
    output_bytes = sum(stats[name]["output_bytes"] for name in sources)
    accepted = sum(stats[name]["accepted"] for name in sources)
    rejected = sum(stats[name]["rejected"] for name in sources)
    rate = max(0.0, (output_bytes - sum(stats[name].get("resume_bytes", 0) for name in sources)))
    session_elapsed = max(0.001, time.monotonic() - started_at)
    bytes_per_sec = rate / session_elapsed
    docs_per_sec = max(
        0.0,
        (accepted - sum(stats[name].get("resume_docs", 0) for name in sources)) / session_elapsed,
    )
    remaining = max(0, target_bytes - output_bytes)
    eta = remaining / bytes_per_sec if bytes_per_sec else None
    per_source: dict[str, dict[str, Any]] = {}
    for name in sources:
        state: dict[str, Any] = {}
        for key, value in stats[name].items():
            if key.startswith("resume_"):
                continue
            # Producer threads mutate these dictionaries while a checkpoint is
            # serialized, so snapshots must not retain references to them.
            state[key] = dict(value) if isinstance(value, dict) else value
        per_source[name] = state
    return {
        "version": 1,
        "complete": complete,
        "out": str(out_path.resolve()),
        "seed": seed,
        "target_bytes": target_bytes,
        "output_file_bytes": out_path.stat().st_size if out_path.exists() else 0,
        "output_bytes": output_bytes,
        "text_bytes": sum(stats[name]["text_bytes"] for name in sources),
        "accepted_docs": accepted,
        "rejected_docs": rejected,
        "scanned_docs": sum(stats[name]["scanned"] for name in sources),
        "elapsed_sec": round(elapsed, 3),
        "docs_per_sec": round(docs_per_sec, 3),
        "mb_per_sec": round(bytes_per_sec / 1_000_000, 3),
        "eta_sec": round(eta, 1) if eta is not None else None,
        "per_source": per_source,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def rebuild_index_and_stats(
    out_path: Path,
    dedup: DedupIndex,
    stats: dict[str, dict[str, Any]],
    sources: list[str],
) -> None:
    for name in sources:
        stats[name]["accepted"] = 0
        stats[name]["output_bytes"] = 0
        stats[name]["text_bytes"] = 0
        stats[name]["words"] = 0
    good_end = 0
    with out_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
                source = record["source"]
                text = record["text"]
                if source not in stats or not isinstance(text, str):
                    raise ValueError("invalid record")
            except Exception:
                break
            exact, simhash = fingerprints(text)
            if dedup.classify(exact, simhash) is None:
                dedup.add(exact, simhash)
            state = stats[source]
            state["accepted"] += 1
            state["output_bytes"] += len(line)
            state["text_bytes"] += len(text.encode("utf-8"))
            state["words"] += len(WORD_RE.findall(text))
            good_end = handle.tell()
    if out_path.stat().st_size != good_end:
        with out_path.open("r+b") as handle:
            handle.truncate(good_end)
    dedup.commit()


def make_bars(
    sources: list[str],
    stats: dict[str, dict[str, Any]],
    target_bytes: int,
) -> tuple[Any, dict[str, Any]]:
    if tqdm is None:
        raise RuntimeError(
            "Missing dependency 'tqdm'. Install project requirements with "
            "`python3 -m pip install -r requirements.txt`."
        )
    total = tqdm(
        total=target_bytes,
        initial=sum(stats[name]["output_bytes"] for name in sources),
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc="total",
        position=0,
        dynamic_ncols=True,
    )
    bars = {
        name: tqdm(
            total=stats[name]["quota_bytes"],
            initial=stats[name]["output_bytes"],
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
            desc=name[:18],
            position=index + 1,
            dynamic_ncols=True,
        )
        for index, name in enumerate(sources)
    }
    return total, bars


def build(args: argparse.Namespace) -> int:
    sources = args.sources
    target_bytes = int(args.target_gb * GB)
    quotas = source_quotas(sources, target_bytes)
    out_path = args.out
    progress_path = out_path.with_suffix(".progress.json")
    stats_path = out_path.with_suffix(".stats.json")
    cache_dir = args.cache_dir
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.resume:
        raise RuntimeError(f"{out_path} exists; pass --resume or choose another --out")
    progress: dict[str, Any] | None = None
    if args.resume and progress_path.exists():
        with progress_path.open(encoding="utf-8") as handle:
            progress = json.load(handle)
        if progress.get("out") != str(out_path.resolve()):
            raise RuntimeError(f"{progress_path} belongs to a different output path")
        if int(progress.get("seed", args.seed)) != args.seed:
            raise RuntimeError("resume seed differs from the existing progress file")

    stats = initial_source_stats(sources, quotas, progress)
    dedup_name = hashlib.sha256(str(out_path.resolve()).encode()).hexdigest()[:16]
    dedup_path = cache_dir / f"dedup-{dedup_name}.sqlite3"
    progress_matches = bool(
        progress
        and out_path.exists()
        and progress.get("output_file_bytes") == out_path.stat().st_size
        and progress.get("target_bytes") == target_bytes
        and set(progress.get("per_source", {})) == set(sources)
    )
    dedup = DedupIndex(dedup_path, reset=not progress_matches)
    meta = dedup.meta()
    index_matches = (
        progress_matches
        and meta.get("out") == str(out_path.resolve())
        and meta.get("output_file_bytes") == out_path.stat().st_size
    )
    if out_path.exists() and out_path.stat().st_size and not index_matches:
        dedup.close()
        dedup = DedupIndex(dedup_path, reset=True)
        rebuild_index_and_stats(out_path, dedup, stats, sources)

    for name in sources:
        stats[name]["resume_bytes"] = stats[name]["output_bytes"]
        stats[name]["resume_docs"] = stats[name]["accepted"]

    done_sources = {
        name for name in sources if stats[name]["output_bytes"] >= stats[name]["quota_bytes"]
    }
    stop_all = threading.Event()
    source_stops = {name: threading.Event() for name in sources}
    for name in done_sources:
        source_stops[name].set()
    lock = threading.Lock()
    item_queue: queue.Queue[Candidate | ProducerDone] = queue.Queue(maxsize=max(64, args.workers * 32))
    started_at = time.monotonic()
    initial_elapsed = float((progress or {}).get("elapsed_sec", 0.0))
    total_bar, bars = make_bars(sources, stats, target_bytes)
    output = out_path.open("ab")
    errors: dict[str, str] = {}
    last_progress = 0.0
    last_checkpoint = 0.0

    def checkpoint(complete: bool = False) -> None:
        output.flush()
        os.fsync(output.fileno())
        with lock:
            snap = snapshot(
                out_path, target_bytes, sources, stats, started_at,
                initial_elapsed, args.seed, complete,
            )
        dedup.set_meta(
            out=str(out_path.resolve()),
            output_file_bytes=snap["output_file_bytes"],
            seed=args.seed,
        )
        dedup.commit()
        atomic_json(progress_path, snap)

    try:
        active = [name for name in sources if name not in done_sources]
        executor = ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(active))))
        try:
            futures = [
                executor.submit(
                    producer, name, sources.index(name), cache_dir, args.seed,
                    item_queue, stats, lock, stop_all, source_stops[name],
                )
                for name in active
            ]
            while len(done_sources) < len(sources):
                now = time.monotonic()
                try:
                    item = item_queue.get(timeout=0.25)
                except queue.Empty:
                    item = None

                if isinstance(item, ProducerDone):
                    stats[item.source]["exhausted"] = True
                    done_sources.add(item.source)
                    if item.error:
                        errors[item.source] = item.error
                elif isinstance(item, Candidate):
                    state = stats[item.source]
                    if state["output_bytes"] >= state["quota_bytes"]:
                        source_stops[item.source].set()
                        done_sources.add(item.source)
                    else:
                        duplicate = dedup.classify(item.exact, item.simhash)
                        if duplicate:
                            with lock:
                                state["rejected"] += 1
                                reasons = state["reject_reasons"]
                                reasons[duplicate] = reasons.get(duplicate, 0) + 1
                        else:
                            record = {
                                "id": item.doc_id,
                                "source": item.source,
                                "text": item.text,
                            }
                            line = (
                                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                                + "\n"
                            ).encode("utf-8")
                            output.write(line)
                            dedup.add(item.exact, item.simhash)
                            with lock:
                                state["accepted"] += 1
                                state["output_bytes"] += len(line)
                                state["text_bytes"] += len(item.text.encode("utf-8"))
                                state["words"] += len(WORD_RE.findall(item.text))
                            bars[item.source].update(len(line))
                            total_bar.update(len(line))
                            if state["output_bytes"] >= state["quota_bytes"]:
                                source_stops[item.source].set()
                                done_sources.add(item.source)

                if now - last_progress >= PROGRESS_INTERVAL:
                    accepted = sum(stats[name]["accepted"] for name in sources)
                    rejected = sum(stats[name]["rejected"] for name in sources)
                    elapsed = max(0.001, now - started_at)
                    session_docs = accepted - sum(stats[name]["resume_docs"] for name in sources)
                    session_bytes = sum(
                        stats[name]["output_bytes"] - stats[name]["resume_bytes"] for name in sources
                    )
                    total_bar.set_postfix_str(
                        f"{session_docs/elapsed:.1f} docs/s "
                        f"{session_bytes/elapsed/1e6:.2f} MB/s "
                        f"ok={accepted:,} reject={rejected:,}"
                    )
                    for name in sources:
                        bars[name].set_postfix_str(
                            f"ok={stats[name]['accepted']:,} "
                            f"reject={stats[name]['rejected']:,}"
                        )
                    last_progress = now
                if now - last_checkpoint >= CHECKPOINT_INTERVAL:
                    checkpoint()
                    last_checkpoint = now

            stop_all.set()
            for future in futures:
                future.result()
        finally:
            # Set the stop flag before waiting. ThreadPoolExecutor's context
            # manager waits first, which can deadlock on Ctrl-C while a
            # producer is blocked on the bounded queue.
            stop_all.set()
            executor.shutdown(wait=True)
        complete = all(
            stats[name]["output_bytes"] >= stats[name]["quota_bytes"] for name in sources
        )
        checkpoint(complete=complete)
    except KeyboardInterrupt:
        stop_all.set()
        checkpoint(complete=False)
        raise
    finally:
        stop_all.set()
        output.close()
        dedup.close()
        for bar in bars.values():
            bar.close()
        total_bar.close()

    final = snapshot(
        out_path, target_bytes, sources, stats, started_at,
        initial_elapsed, args.seed,
        complete=all(stats[name]["output_bytes"] >= stats[name]["quota_bytes"] for name in sources),
    )
    final["source_weights_gb_at_10gb"] = SOURCE_WEIGHTS
    final["dedup_index"] = str(dedup_path)
    final["errors"] = errors
    atomic_json(stats_path, final)
    if not final["complete"]:
        missing = {
            name: stats[name]["quota_bytes"] - stats[name]["output_bytes"]
            for name in sources
            if stats[name]["output_bytes"] < stats[name]["quota_bytes"]
        }
        print(f"Corpus incomplete; missing bytes by source: {missing}", file=sys.stderr)
        if errors:
            print(f"Source errors: {errors}", file=sys.stderr)
        return 1
    print(
        f"Wrote {final['output_bytes'] / GB:.3f} GB, "
        f"{final['accepted_docs']:,} documents to {out_path}"
    )
    print(f"Progress: {progress_path}")
    print(f"Statistics: {stats_path}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-gb", type=float, default=10.0,
        help="decimal GB of UTF-8 JSONL output (default: 10)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/corpus/corpus_10gb.jsonl"),
        help="output JSONL path",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/pretraining"),
        help="Hugging Face cache and persistent dedup index directory",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="maximum concurrent source download/cleaning workers",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from matching output, progress, and dedup state",
    )
    parser.add_argument(
        "--sources", type=parse_sources, default=list(SOURCE_WEIGHTS),
        help="comma-separated sources or 'all'; selected weights are renormalized",
    )
    parser.add_argument("--seed", type=int, default=42, help="stream shuffle seed")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.target_gb <= 0:
        parser.error("--target-gb must be greater than zero")
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")
    try:
        return build(args)
    except KeyboardInterrupt:
        print("\nInterrupted; resumable progress was saved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
