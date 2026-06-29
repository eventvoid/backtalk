#!/usr/bin/env python3
"""Parallel, resumable cleanup and validation for a pretraining JSONL corpus.

The original input is opened read-only. Cleaned records are written to a new
JSONL file, deduplicated with a persistent SQLite index, topped up from a
streaming Hugging Face source, and then independently validated in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import sqlite3
import sys
import time
import unicodedata
from array import array
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm

try:
    import py3langid
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing py3langid. Run: python3 -m pip install -r requirements.txt"
    ) from exc


GB = 1_000_000_000
CHECKPOINT_SECONDS = 5.0
KNOWN_SOURCES = {
    "fineweb-edu",
    "wikipedia",
    "simple-wikipedia",
    "smollm-edu",
    "open-web-math",
    "stackexchange",
}
EXPECTED_KEYS = {"id", "source", "text"}

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

HTML_RE = re.compile(
    r"</?(?:html|head|body|div|span|p|a|br|hr|table|tr|td|th|tbody|thead|"
    r"script|style|ul|ol|li|form|nav|footer|header|section|article|iframe|"
    r"svg|button|input|meta|link)\b[^>]*>",
    re.I,
)
ENTITY_RE = re.compile(
    r"&(?:nbsp|amp|lt|gt|quot|apos|copy|reg|mdash|ndash|#\d{2,5}|#x[0-9a-f]+);",
    re.I,
)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NORMAL_RE = re.compile(r"[a-z0-9]+")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WIKI_CITE_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]{6,}\|?\s*$")
CODE_LINE_RE = re.compile(
    r"^\s*(?:```|~~~|def |class |import |from \S+ import |#include|"
    r"using namespace|public (?:class|static)|private |protected |function |"
    r"func |(?:const|let|var)\s+\w+|(?:if|for|while|switch)\s*\(|"
    r"SELECT\s+.+\s+FROM|INSERT\s+INTO|CREATE\s+TABLE|pip install|npm install)",
    re.I,
)
CODE_TOKEN_RE = re.compile(
    r"(?:=>|::|&&|\|\||console\.(?:log|error)|System\.out|"
    r"\b(?:printf|println)\s*\(|^\s*[\w.]+\([^)]*\)\s*;?\s*$)",
    re.I,
)
SOFTWARE_DOC_RE = re.compile(
    r"\b(?:API reference|command[- ]line option|source code|class method|"
    r"function signature|package manager|stack trace|compiler error|"
    r"programming language|GitHub repository)\b",
    re.I,
)
BOILER_RE = re.compile(
    r"\b(?:cookie policy|privacy policy|terms of (?:use|service)|"
    r"all rights reserved|subscribe(?: to| now)?|sign[ -]?in|log[ -]?in|"
    r"click here|read more|share this|leave a (?:comment|reply)|"
    r"related (?:posts|articles)|advertisement|skip to (?:main )?content|"
    r"accept (?:all )?cookies|enable javascript|newsletter|follow us|"
    r"page tools|print page|print all|back to top)\b",
    re.I,
)
NAV_RE = re.compile(
    r"^(?:home\s*[>»|]|(?:previous|next)\s*(?:page|article)?$|"
    r"(?:views?|author|publish time|posted on|updated on)\s*:\s*|"
    r"(?:main menu|site menu|search this site|continue to site)$|"
    r"(?:development|aid|economy|trade|human rights|global governance)"
    r"(?:\s*[&|]\s*\w+){2,})",
    re.I,
)
SEO_RE = re.compile(
    r"\b(?:best price|buy cheap|limited time offer|free download|"
    r"download (?:pdf|ebook|kindle)|casino bonus|payday loan|"
    r"write my essay|online dating)\b",
    re.I,
)
STOP_SECTIONS = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "citations",
        "sources",
        "footnotes",
        "gallery",
        "related pages",
        "other websites",
        "notes and references",
    }
)
PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)


class VisibleTextParser(HTMLParser):
    DROP = {
        "script",
        "style",
        "code",
        "pre",
        "table",
        "svg",
        "nav",
        "footer",
        "form",
    }
    BREAK = {
        "p",
        "div",
        "br",
        "li",
        "article",
        "section",
        "header",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.drop_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag in self.DROP:
            self.drop_depth += 1
        elif not self.drop_depth and tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.DROP and self.drop_depth:
            self.drop_depth -= 1
        elif not self.drop_depth and tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth:
            self.parts.append(data)

    def result(self) -> str:
        return "".join(self.parts)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hash64(value: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def text_digest(text: str) -> bytes:
    normalized = " ".join(NORMAL_RE.findall(text.lower()))
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def source_id(source: str, row: dict[str, Any], text: str, cursor: int) -> str:
    for key in ("id", "pageid", "url"):
        value = row.get(key)
        if value is not None and str(value).strip():
            digest = hashlib.blake2s(
                str(value).strip().encode("utf-8"), digest_size=10
            ).hexdigest()
            return f"{source}:{digest}"
    digest = hashlib.blake2s(text[:4096].encode("utf-8"), digest_size=10).hexdigest()
    return f"{source}:{cursor:x}:{digest}"


def strip_html(text: str) -> str:
    text = html.unescape(text)
    if HTML_RE.search(text):
        parser = VisibleTextParser()
        try:
            parser.feed(text)
            parser.close()
            text = parser.result()
        except Exception:
            text = HTML_RE.sub(" ", text)
    return text


def is_table_line(line: str) -> bool:
    return (
        line.count("|") >= 3
        or line.count("\t") >= 2
        or bool(TABLE_SEP_RE.match(line) and ("|" in line or "-" in line))
    )


def is_code_line(line: str) -> bool:
    if CODE_LINE_RE.search(line) or CODE_TOKEN_RE.search(line):
        return True
    if line.count("{") + line.count("}") >= 2:
        return True
    if len(line) > 24:
        symbols = sum(character in "{}[]<>;`\\=_" for character in line)
        if symbols / len(line) > 0.15:
            return True
    return False


def split_long_document(text: str, max_words: int = 10_000) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    target = min(8_000, max_words)
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        while len(paragraph_words) > max_words:
            head, paragraph_words = paragraph_words[:target], paragraph_words[target:]
            if current:
                pieces.append("\n\n".join(current))
                current, current_words = [], 0
            pieces.append(" ".join(head))
        if current and current_words + len(paragraph_words) > target:
            pieces.append("\n\n".join(current))
            current, current_words = [], 0
        if paragraph_words:
            current.append(" ".join(paragraph_words))
            current_words += len(paragraph_words)
    if current:
        tail = "\n\n".join(current)
        if len(tail.split()) < 50 and pieces:
            pieces[-1] = pieces[-1] + "\n\n" + tail
        else:
            pieces.append(tail)
    return [piece for piece in pieces if 50 <= len(piece.split()) <= max_words]


def clean_text(raw: Any, source: str) -> tuple[list[str], Optional[str]]:
    if not isinstance(raw, str) or len(raw) < 200:
        return [], "too_short_raw"
    if "\ufffd" in raw:
        return [], "replacement_character"
    if SURROGATE_RE.search(raw):
        return [], "broken_unicode"
    try:
        raw.encode("utf-8", "strict")
    except UnicodeError:
        return [], "broken_unicode"

    text = strip_html(raw)
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = WIKI_CITE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text.translate(PUNCT_TRANSLATION))
    text = CTRL_RE.sub("", text)

    output: list[str] = []
    code_lines = 0
    boiler_lines = 0
    content_lines = 0
    fenced = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            code_lines += 1
            continue
        if fenced:
            code_lines += bool(line)
            continue
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        content_lines += 1
        heading = HEADING_RE.sub("", line).strip()
        if source in {"wikipedia", "simple-wikipedia"} and heading.lower() in STOP_SECTIONS:
            break
        line = LIST_RE.sub("", heading).strip()
        if not line:
            continue
        if is_table_line(line):
            boiler_lines += 1
            continue
        if is_code_line(line):
            code_lines += 1
            continue
        if BOILER_RE.search(line) or NAV_RE.search(line) or SEO_RE.search(line):
            boiler_lines += 1
            continue
        letters = [character for character in line if character.isalpha()]
        if letters:
            latin = sum(ord(character) < 0x0250 for character in letters)
            if latin / len(letters) < 0.72:
                continue
        line = URL_RE.sub("", line)
        line = SPACE_RE.sub(" ", line).strip()
        if len(line) > 300 and " " not in line:
            continue
        if line:
            output.append(line)

    if content_lines:
        code_ratio = code_lines / content_lines
        boiler_ratio = boiler_lines / content_lines
        if code_lines >= 15 or (code_lines >= 5 and code_ratio >= 0.08):
            return [], "code_heavy"
        if SOFTWARE_DOC_RE.search(text) and code_lines >= 3:
            return [], "software_documentation"
        if boiler_lines >= 12 or (boiler_lines >= 5 and boiler_ratio >= 0.30):
            return [], "boilerplate_heavy"

    cleaned = BLANK_RE.sub("\n\n", "\n".join(output)).strip()
    words = cleaned.split()
    if len(words) < 50:
        return [], "too_short"
    nonspace = sum(not character.isspace() for character in cleaned)
    letters = [character for character in cleaned if character.isalpha()]
    if not nonspace or len(letters) / nonspace < 0.64:
        return [], "low_alpha"
    latin = sum(ord(character) < 0x0250 for character in letters)
    if not letters or latin / len(letters) < 0.94:
        return [], "non_english_script"
    if sum(character.isdigit() for character in cleaned) / nonspace > 0.18:
        return [], "too_many_digits"

    language_words = [word.lower().strip("'") for word in WORD_RE.findall(cleaned)]
    stop_count = sum(word in STOPWORDS for word in language_words)
    stop_ratio = stop_count / max(len(language_words), 1)
    if stop_count < 3 or stop_ratio < 0.045:
        return [], "non_english_heuristic"
    if stop_ratio < 0.10:
        language, _score = py3langid.classify(cleaned[:2000])
        if language != "en":
            return [], "non_english_classifier"

    lines = [line for line in cleaned.splitlines() if line]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.60:
        return [], "repetition"
    if HTML_RE.search(cleaned) or ENTITY_RE.search(cleaned):
        return [], "residual_html"
    return split_long_document(cleaned), None


def clean_batch(payload: list[bytes]) -> dict[str, Any]:
    records: list[tuple[str, str, str, bytes]] = []
    rejects: Counter[str] = Counter()
    source_scanned: Counter[str] = Counter()
    for raw_line in payload:
        try:
            record = json.loads(raw_line)
        except Exception:
            rejects["invalid_json"] += 1
            continue
        if not isinstance(record, dict):
            rejects["non_object"] += 1
            continue
        if set(record) != EXPECTED_KEYS:
            rejects["invalid_schema"] += 1
            continue
        doc_id, source, raw_text = record.get("id"), record.get("source"), record.get("text")
        if not isinstance(doc_id, str) or not isinstance(source, str) or not isinstance(raw_text, str):
            rejects["invalid_types"] += 1
            continue
        source_scanned[source] += 1
        if source not in KNOWN_SOURCES:
            rejects["unknown_source"] += 1
            continue
        pieces, reason = clean_text(raw_text, source)
        if reason:
            rejects[reason] += 1
            continue
        for index, text in enumerate(pieces):
            output_id = doc_id if len(pieces) == 1 else f"{doc_id}#part-{index + 1}"
            records.append((output_id, source, text, text_digest(text)))
    return {
        "input_docs": len(payload),
        "records": records,
        "rejects": dict(rejects),
        "source_scanned": dict(source_scanned),
    }


def validation_flags(text: str) -> list[str]:
    flags: list[str] = []
    if not text.strip():
        flags.append("empty")
    if "\ufffd" in text or SURROGATE_RE.search(text) or CTRL_RE.search(text):
        flags.append("broken_unicode")
    if HTML_RE.search(text):
        flags.append("html")
    if ENTITY_RE.search(text):
        flags.append("html_entity")
    if BOILER_RE.search(text) or NAV_RE.search(text) or SEO_RE.search(text):
        flags.append("boilerplate")
    code_lines = table_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        code_lines += is_code_line(stripped)
        table_lines += is_table_line(stripped)
    if code_lines >= 3:
        flags.append("code")
    if table_lines >= 2:
        flags.append("table")
    word_count = len(text.split())
    if word_count < 50:
        flags.append("too_short")
    if word_count > 10_000:
        flags.append("too_long")
    return flags


def validate_batch(payload: list[bytes], seed: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    source_words: Counter[str] = Counter()
    source_chars: Counter[str] = Counter()
    leakage: Counter[str] = Counter()
    id_hashes: list[int] = []
    text_hashes: list[int] = []
    word_lengths: list[int] = []
    char_lengths: list[int] = []
    language_candidates: list[tuple[int, str, str, str, str]] = []
    sample_candidates: list[tuple[int, dict[str, Any]]] = []

    for raw_line in payload:
        counts["lines"] += 1
        try:
            record = json.loads(raw_line)
        except Exception:
            counts["invalid_json"] += 1
            continue
        if not isinstance(record, dict):
            counts["non_object"] += 1
            continue
        counts["valid_objects"] += 1
        if set(record) != EXPECTED_KEYS:
            counts["invalid_schema"] += 1
            continue
        doc_id, source, text = record.get("id"), record.get("source"), record.get("text")
        if not isinstance(doc_id, str) or not isinstance(source, str) or not isinstance(text, str):
            counts["invalid_types"] += 1
            continue
        if source not in KNOWN_SOURCES:
            counts["invalid_source"] += 1
        counts["valid_docs"] += 1
        flags = validation_flags(text)
        leakage.update(flags)
        words = len(text.split())
        characters = len(text)
        word_lengths.append(words)
        char_lengths.append(characters)
        id_hashes.append(hash64(doc_id.encode("utf-8")))
        text_hashes.append(hash64(text.encode("utf-8")))
        sources[source] += 1
        source_bytes[source] += len(raw_line)
        source_words[source] += words
        source_chars[source] += characters

        priority = hash64(f"{seed}:{doc_id}".encode("utf-8"))
        language_candidates.append((priority, "", doc_id, source, text[:2000]))
        sample_candidates.append(
            (
                priority,
                {
                    "id": doc_id,
                    "source": source,
                    "words": words,
                    "flags": flags,
                    "text": text[:2000],
                },
            )
        )

    language_candidates.sort(key=lambda item: item[0])
    classified = []
    for priority, _language, doc_id, source, text in language_candidates[:12]:
        language, score = py3langid.classify(text)
        classified.append(
            (
                priority,
                language,
                doc_id,
                source,
                text[:240] if language != "en" else "",
                float(score),
            )
        )
    sample_candidates.sort(key=lambda item: item[0])
    return {
        "counts": dict(counts),
        "sources": dict(sources),
        "source_bytes": dict(source_bytes),
        "source_words": dict(source_words),
        "source_chars": dict(source_chars),
        "leakage": dict(leakage),
        "id_hashes": id_hashes,
        "text_hashes": text_hashes,
        "word_lengths": word_lengths,
        "char_lengths": char_lengths,
        "language_candidates": classified,
        "sample_candidates": sample_candidates[:2],
    }


class DedupIndex:
    def __init__(self, path: Path, reset: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ids (
                id TEXT PRIMARY KEY,
                epoch INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS texts (
                digest BLOB PRIMARY KEY,
                epoch INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS ids_epoch ON ids(epoch);
            CREATE INDEX IF NOT EXISTS texts_epoch ON texts(epoch);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()

    def rollback_after(self, epoch: int) -> None:
        self.connection.execute("DELETE FROM ids WHERE epoch > ?", (epoch,))
        self.connection.execute("DELETE FROM texts WHERE epoch > ?", (epoch,))
        self.connection.commit()

    def add(self, doc_id: str, digest: bytes, epoch: int) -> Optional[str]:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO ids(id,epoch) VALUES(?,?)", (doc_id, epoch)
        )
        if cursor.rowcount == 0:
            return "duplicate_id"
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO texts(digest,epoch) VALUES(?,?)", (digest, epoch)
        )
        if cursor.rowcount == 0:
            return "duplicate_text"
        return None

    def set_meta(self, **values: Any) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            [(key, json.dumps(value)) for key, value in values.items()],
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def parse_workers(value: str) -> int:
    if value.lower() == "auto":
        return os.cpu_count() or 1
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--workers must be 'auto' or a positive integer") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--workers must be greater than zero")
    return workers


def input_batches(path: Path, offset: int, batch_docs: int) -> Iterator[tuple[list[bytes], int]]:
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            payload = []
            for _ in range(batch_docs):
                line = handle.readline()
                if not line:
                    break
                payload.append(line)
            if not payload:
                break
            yield payload, handle.tell()


def output_batches(path: Path, batch_docs: int) -> Iterator[tuple[list[bytes], int]]:
    yield from input_batches(path, 0, batch_docs)


def ordered_jobs(
    executor: ProcessPoolExecutor,
    batches: Iterable[tuple[list[bytes], Any]],
    function: Any,
    max_pending: int,
    *extra: Any,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    pending: deque[tuple[Any, Any]] = deque()
    for payload, metadata in batches:
        pending.append((metadata, executor.submit(function, payload, *extra)))
        if len(pending) >= max_pending:
            meta, future = pending.popleft()
            yield meta, future.result()
    while pending:
        meta, future = pending.popleft()
        yield meta, future.result()


def initial_state(
    input_path: Path, output_path: Path, target_bytes: int, workers: int
) -> dict[str, Any]:
    return {
        "version": 1,
        "stage": "cleaning",
        "input": str(input_path.resolve()),
        "input_size_bytes": input_path.stat().st_size,
        "input_mtime_ns": input_path.stat().st_mtime_ns,
        "output": str(output_path.resolve()),
        "target_bytes": target_bytes,
        "workers": workers,
        "checkpoint": 0,
        "input_offset": 0,
        "input_docs": 0,
        "topup_scanned": 0,
        "accepted_docs": 0,
        "rejected_docs": 0,
        "output_bytes": 0,
        "text_bytes": 0,
        "words": 0,
        "reject_reasons": {},
        "per_source": {},
        "elapsed_sec": 0.0,
        "complete": False,
    }


def merge_counter(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + int(value)


def write_record(
    output: Any,
    dedup: DedupIndex,
    epoch: int,
    state: dict[str, Any],
    doc_id: str,
    source: str,
    text: str,
    digest: bytes,
) -> int:
    reason = dedup.add(doc_id, digest, epoch)
    if reason:
        state["rejected_docs"] += 1
        state["reject_reasons"][reason] = state["reject_reasons"].get(reason, 0) + 1
        return 0
    encoded = (
        json.dumps(
            {"id": doc_id, "source": source, "text": text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    output.write(encoded)
    text_bytes = len(text.encode("utf-8"))
    words = len(text.split())
    state["accepted_docs"] += 1
    state["output_bytes"] += len(encoded)
    state["text_bytes"] += text_bytes
    state["words"] += words
    source_state = state["per_source"].setdefault(
        source, {"accepted": 0, "output_bytes": 0, "text_bytes": 0, "words": 0}
    )
    source_state["accepted"] += 1
    source_state["output_bytes"] += len(encoded)
    source_state["text_bytes"] += text_bytes
    source_state["words"] += words
    return len(encoded)


def read_dotenv_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    dotenv = Path(".env")
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("HUGGINGFACE_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
            if stripped.startswith("HF_TOKEN="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    return None


def topup_producer(
    source: str,
    cache_dir: str,
    seed: int,
    skip: int,
    batch_docs: int,
    token: Optional[str],
    result_queue: Any,
    stop_event: Any,
) -> None:
    def send(item: Any) -> bool:
        while not stop_event.is_set():
            try:
                result_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    try:
        from datasets import load_dataset
        if source == "fineweb-edu":
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                split="train",
                streaming=True,
                cache_dir=cache_dir,
                token=token,
            )
        else:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True,
                cache_dir=cache_dir,
                token=token,
            )
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        if skip:
            dataset = dataset.skip(skip)
        cursor = skip
        payload: list[bytes] = []
        for row in dataset:
            if stop_event.is_set():
                return
            cursor += 1
            text = row.get("text") or ""
            doc_id = source_id(source, row, text, cursor)
            payload.append(
                (
                    json.dumps(
                        {"id": doc_id, "source": source, "text": text},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
            if len(payload) >= batch_docs:
                if not send((payload, cursor, None)):
                    return
                payload = []
        if payload:
            if not send((payload, cursor, None)):
                return
        send((None, cursor, None))
    except Exception as exc:
        send((None, skip, f"{type(exc).__name__}: {exc}"))


def topup_batches(
    source: str,
    cache_dir: Path,
    seed: int,
    skip: int,
    batch_docs: int,
) -> Iterator[tuple[list[bytes], int]]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=4)
    stop_event = context.Event()
    process = context.Process(
        target=topup_producer,
        args=(
            source,
            str(cache_dir),
            seed,
            skip,
            batch_docs,
            read_dotenv_token(),
            result_queue,
            stop_event,
        ),
        daemon=True,
    )
    process.start()
    try:
        while True:
            payload, cursor, error = result_queue.get()
            if error:
                raise RuntimeError(f"top-up stream failed: {error}")
            if payload is None:
                break
            yield payload, cursor
    finally:
        stop_event.set()
        process.join(timeout=0.25)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        result_queue.cancel_join_thread()
        result_queue.close()


def rebuild_dedup(output_path: Path, index: DedupIndex) -> None:
    print("Rebuilding missing dedup index from checkpointed output...", file=sys.stderr)
    with output_path.open("rb") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            index.add(record["id"], text_digest(record["text"]), 0)
    index.commit()


def clean_and_topup(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    input_path: Path = args.input
    output_path: Path = args.output
    progress_path = output_path.with_suffix(".progress.json")
    stats_path = output_path.with_suffix(".stats.json")
    target_bytes = int(args.target_gb * GB)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    index_name = hashlib.sha256(str(output_path.resolve()).encode()).hexdigest()[:16]
    index_path = args.cache_dir / f"clean-dedup-{index_name}.sqlite3"

    if not input_path.exists():
        raise RuntimeError(f"input does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("--output must differ from --input")
    if output_path.exists() and not args.resume:
        raise RuntimeError(f"{output_path} exists; use --resume or a new output path")

    resuming = args.resume and progress_path.exists()
    if args.resume and output_path.exists() and not progress_path.exists():
        raise RuntimeError(
            f"--resume requires its progress file: {progress_path}"
        )
    if resuming:
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("input") != str(input_path.resolve()):
            raise RuntimeError("progress file belongs to a different input")
        if state.get("output") != str(output_path.resolve()):
            raise RuntimeError("progress file belongs to a different output")
        if state.get("target_bytes") != target_bytes:
            raise RuntimeError("--target-gb differs from the existing progress file")
        if input_path.stat().st_size != state.get("input_size_bytes"):
            raise RuntimeError("input size changed since the previous run")
        if input_path.stat().st_mtime_ns != state.get("input_mtime_ns"):
            raise RuntimeError("input modification time changed since the previous run")
    else:
        state = initial_state(input_path, output_path, target_bytes, args.workers)

    saved_output_bytes = int(state["output_bytes"])
    if output_path.exists():
        actual_output_bytes = output_path.stat().st_size
        if actual_output_bytes < saved_output_bytes:
            raise RuntimeError(
                "output is shorter than its checkpoint; refusing to extend it"
            )
        with output_path.open("r+b") as handle:
            handle.truncate(saved_output_bytes)
    else:
        if saved_output_bytes:
            raise RuntimeError("checkpoint references a missing output file")
        output_path.touch()

    index_existed = index_path.exists()
    dedup = DedupIndex(index_path, reset=not resuming)
    if resuming:
        dedup.rollback_after(int(state["checkpoint"]))
        if saved_output_bytes and not index_existed:
            rebuild_dedup(output_path, dedup)

    output = output_path.open("ab")
    base_elapsed = float(state.get("elapsed_sec", 0.0))
    session_started = time.monotonic()
    epoch = int(state["checkpoint"]) + 1
    last_checkpoint = 0.0

    def checkpoint(stage: Optional[str] = None) -> None:
        nonlocal epoch, last_checkpoint
        if stage:
            state["stage"] = stage
        output.flush()
        os.fsync(output.fileno())
        state["elapsed_sec"] = round(
            base_elapsed + time.monotonic() - session_started, 3
        )
        state["checkpoint"] = epoch
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        dedup.set_meta(
            output=str(output_path.resolve()),
            output_bytes=state["output_bytes"],
            checkpoint=epoch,
        )
        dedup.commit()
        atomic_json(progress_path, state)
        epoch += 1
        last_checkpoint = time.monotonic()

    def maybe_checkpoint() -> None:
        if time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS:
            checkpoint()

    try:
        if state["stage"] == "cleaning":
            start_offset = int(state["input_offset"])
            start_docs = int(state["input_docs"])
            session_docs = 0
            session_bytes = 0
            progress = tqdm(
                total=input_path.stat().st_size,
                initial=start_offset,
                unit="B",
                unit_scale=True,
                unit_divisor=1000,
                desc="clean",
                dynamic_ncols=True,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                jobs = ordered_jobs(
                    executor,
                    input_batches(input_path, start_offset, args.batch_docs),
                    clean_batch,
                    args.workers * 3,
                )
                previous_offset = start_offset
                for end_offset, result in jobs:
                    batch_input_bytes = end_offset - previous_offset
                    previous_offset = end_offset
                    state["input_offset"] = end_offset
                    state["input_docs"] += result["input_docs"]
                    rejected = sum(result["rejects"].values())
                    state["rejected_docs"] += rejected
                    merge_counter(state["reject_reasons"], result["rejects"])
                    written = 0
                    for doc_id, source, text, digest in result["records"]:
                        written += write_record(
                            output, dedup, epoch, state, doc_id, source, text, digest
                        )
                    session_docs += result["input_docs"]
                    session_bytes += batch_input_bytes
                    elapsed = max(0.001, time.monotonic() - session_started)
                    progress.update(batch_input_bytes)
                    progress.set_postfix_str(
                        f"{session_docs/elapsed:,.0f} docs/s "
                        f"{session_bytes/elapsed/1e6:.1f} MB/s "
                        f"ok={state['accepted_docs']:,} reject={state['rejected_docs']:,}"
                    )
                    maybe_checkpoint()
            progress.close()
            state["input_offset"] = input_path.stat().st_size
            checkpoint("topup")

        if state["stage"] == "topup":
            if state["output_bytes"] < target_bytes:
                initial_output = state["output_bytes"]
                session_docs = 0
                progress = tqdm(
                    total=target_bytes,
                    initial=state["output_bytes"],
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1000,
                    desc="top-up",
                    dynamic_ncols=True,
                )
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    batch_source = topup_batches(
                        args.topup_source,
                        args.cache_dir,
                        args.seed + 10_000,
                        int(state["topup_scanned"]),
                        args.batch_docs,
                    )
                    jobs = ordered_jobs(
                        executor,
                        batch_source,
                        clean_batch,
                        args.workers * 3,
                    )
                    try:
                        for cursor, result in jobs:
                            state["topup_scanned"] = cursor
                            state["rejected_docs"] += sum(result["rejects"].values())
                            merge_counter(state["reject_reasons"], result["rejects"])
                            session_docs += result["input_docs"]
                            written = 0
                            for doc_id, source, text, digest in result["records"]:
                                if state["output_bytes"] >= target_bytes:
                                    break
                                written += write_record(
                                    output,
                                    dedup,
                                    epoch,
                                    state,
                                    doc_id,
                                    source,
                                    text,
                                    digest,
                                )
                            progress.update(written)
                            elapsed = max(0.001, time.monotonic() - session_started)
                            progress.set_postfix_str(
                                f"{session_docs/elapsed:,.0f} docs/s "
                                f"{(state['output_bytes']-initial_output)/elapsed/1e6:.1f} MB/s "
                                f"ok={state['accepted_docs']:,} reject={state['rejected_docs']:,}"
                            )
                            maybe_checkpoint()
                            if state["output_bytes"] >= target_bytes:
                                break
                    finally:
                        jobs.close()
                        batch_source.close()
                progress.close()
            if state["output_bytes"] < target_bytes:
                raise RuntimeError(
                    f"top-up source exhausted at {state['output_bytes']/GB:.3f} GB"
                )
            checkpoint("validation")

        if state["stage"] == "validation":
            output.flush()
            os.fsync(output.fileno())
            validation = validate_output(
                output_path,
                args.workers,
                args.batch_docs,
                args.seed,
                state,
                progress_path,
            )
            verdict = validation_verdict(validation, target_bytes)
            state["validation"] = validation
            state["verdict"] = verdict
            state["complete"] = verdict == "PASS"
            state["stage"] = "complete"
            checkpoint("complete")
            final_stats = {
                key: value
                for key, value in state.items()
                if key not in {"validation"}
            }
            final_stats["validation"] = validation
            final_stats["dedup_index"] = str(index_path)
            atomic_json(stats_path, final_stats)
        elif state["stage"] == "complete" and not stats_path.exists():
            atomic_json(stats_path, state)
    except KeyboardInterrupt:
        checkpoint()
        raise
    finally:
        output.close()
        dedup.close()
    return state, progress_path, stats_path


def push_smallest(
    heap: list[tuple[int, Any]], priority: int, value: Any, limit: int
) -> None:
    item = (-priority, value)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def length_summary(values: array) -> dict[str, Any]:
    data = np.frombuffer(values, dtype=np.uint32)
    if not len(data):
        return {}
    return {
        "min": int(data.min()),
        "p10": float(np.percentile(data, 10)),
        "median": float(np.percentile(data, 50)),
        "mean": float(data.mean()),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "max": int(data.max()),
    }


def validate_output(
    path: Path,
    workers: int,
    batch_docs: int,
    seed: int,
    state: dict[str, Any],
    progress_path: Path,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    source_words: Counter[str] = Counter()
    source_chars: Counter[str] = Counter()
    leakage: Counter[str] = Counter()
    seen_ids: set[int] = set()
    seen_texts: set[int] = set()
    duplicate_ids = duplicate_texts = 0
    word_lengths = array("I")
    char_lengths = array("I")
    language_heap: list[tuple[int, Any]] = []
    sample_heap: list[tuple[int, Any]] = []
    started = time.monotonic()
    processed_bytes = processed_docs = 0
    progress = tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc="validate",
        dynamic_ncols=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = ordered_jobs(
            executor,
            output_batches(path, batch_docs),
            validate_batch,
            workers * 3,
            seed,
        )
        previous_offset = 0
        for end_offset, result in jobs:
            batch_bytes = end_offset - previous_offset
            previous_offset = end_offset
            processed_bytes += batch_bytes
            processed_docs += result["counts"].get("lines", 0)
            counts.update(result["counts"])
            sources.update(result["sources"])
            source_bytes.update(result["source_bytes"])
            source_words.update(result["source_words"])
            source_chars.update(result["source_chars"])
            leakage.update(result["leakage"])
            for value in result["id_hashes"]:
                if value in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(value)
            for value in result["text_hashes"]:
                if value in seen_texts:
                    duplicate_texts += 1
                else:
                    seen_texts.add(value)
            word_lengths.extend(result["word_lengths"])
            char_lengths.extend(result["char_lengths"])
            for candidate in result["language_candidates"]:
                push_smallest(language_heap, candidate[0], candidate, 20_000)
            for priority, candidate in result["sample_candidates"]:
                push_smallest(sample_heap, priority, candidate, 100)
            elapsed = max(0.001, time.monotonic() - started)
            progress.update(batch_bytes)
            progress.set_postfix_str(
                f"{processed_docs/elapsed:,.0f} docs/s "
                f"{processed_bytes/elapsed/1e6:.1f} MB/s docs={processed_docs:,}"
            )
            if processed_docs % (batch_docs * 20) == 0:
                snapshot = dict(state)
                snapshot["stage"] = "validation"
                snapshot["validation_docs"] = processed_docs
                snapshot["validation_bytes"] = processed_bytes
                snapshot["updated_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                atomic_json(progress_path, snapshot)
    progress.close()

    language_counts: Counter[str] = Counter()
    non_english_examples = []
    for _negative_priority, candidate in language_heap:
        _priority, language, doc_id, source, head, score = candidate
        language_counts[language] += 1
        if language != "en" and len(non_english_examples) < 25:
            non_english_examples.append(
                {
                    "id": doc_id,
                    "source": source,
                    "language": language,
                    "score": score,
                    "head": head,
                }
            )

    samples = [item[1] for item in sorted(sample_heap, reverse=True)]
    samples_path = path.with_suffix(".samples.txt")
    with samples_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            language, score = py3langid.classify(sample["text"])
            handle.write("=" * 100 + "\n")
            handle.write(
                f"SAMPLE {index:03d} id={sample['id']} source={sample['source']} "
                f"words={sample['words']} lang={language} score={float(score):.3f} "
                f"flags={sample['flags']}\n"
            )
            handle.write("-" * 100 + "\n")
            handle.write(sample["text"] + "\n\n")

    language_sample_size = sum(language_counts.values())
    return {
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "counts": dict(counts),
        "correct_fields_only": counts["invalid_schema"] == 0,
        "empty_text_docs": leakage["empty"],
        "duplicate_ids": duplicate_ids,
        "duplicate_texts": duplicate_texts,
        "duplicate_hash_note": (
            "64-bit BLAKE2b validation fingerprints; collision probability is negligible"
        ),
        "total_characters": sum(source_chars.values()),
        "total_words": sum(source_words.values()),
        "source_distribution": {
            source: {
                "docs": sources[source],
                "jsonl_bytes": source_bytes[source],
                "characters": source_chars[source],
                "words": source_words[source],
            }
            for source in sorted(sources)
        },
        "document_length_words": length_summary(word_lengths),
        "document_length_characters": length_summary(char_lengths),
        "leakage_docs": dict(leakage),
        "english_estimate": {
            "sample_size": language_sample_size,
            "seed": seed,
            "counts": dict(language_counts),
            "english_fraction": language_counts["en"] / max(language_sample_size, 1),
            "non_english_examples": non_english_examples,
        },
        "random_samples": {
            "count": len(samples),
            "seed": seed,
            "path": str(samples_path),
        },
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validation_verdict(validation: dict[str, Any], target_bytes: int) -> str:
    counts = validation["counts"]
    leakage = validation["leakage_docs"]
    failures = [
        validation["file_size_bytes"] < target_bytes,
        counts.get("invalid_json", 0) > 0,
        counts.get("non_object", 0) > 0,
        counts.get("invalid_schema", 0) > 0,
        counts.get("invalid_types", 0) > 0,
        counts.get("invalid_source", 0) > 0,
        validation["empty_text_docs"] > 0,
        validation["duplicate_ids"] > 0,
        validation["duplicate_texts"] > 0,
        leakage.get("broken_unicode", 0) > 0,
        leakage.get("html", 0) > 0,
        leakage.get("html_entity", 0) > 0,
        leakage.get("boilerplate", 0) > 0,
        leakage.get("code", 0) > 0,
        leakage.get("table", 0) > 0,
        leakage.get("too_short", 0) > 0,
        leakage.get("too_long", 0) > 0,
        validation["english_estimate"]["english_fraction"] < 0.995,
    ]
    return "FAIL" if any(failures) else "PASS"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpus/corpus_10gb.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/corpus_10gb_clean.jsonl"),
    )
    parser.add_argument("--target-gb", type=float, default=10.0)
    parser.add_argument("--workers", type=parse_workers, default=parse_workers("auto"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/pretraining"),
    )
    parser.add_argument(
        "--topup-source",
        choices=("fineweb-edu", "wikipedia"),
        default="fineweb-edu",
    )
    parser.add_argument("--batch-docs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.target_gb <= 0:
        parser.error("--target-gb must be greater than zero")
    if args.batch_docs < 16:
        parser.error("--batch-docs must be at least 16")
    try:
        state, progress_path, stats_path = clean_and_topup(args)
    except KeyboardInterrupt:
        print("\nInterrupted; checkpoint saved. Re-run with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{state.get('verdict', 'UNKNOWN')}: "
        f"{state['accepted_docs']:,} docs, {state['output_bytes']/GB:.3f} GB"
    )
    print(f"Output: {args.output}")
    print(f"Progress: {progress_path}")
    print(f"Stats: {stats_path}")
    print(f"Samples: {args.output.with_suffix('.samples.txt')}")
    return 0 if state.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
