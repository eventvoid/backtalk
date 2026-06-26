#!/usr/bin/env python3
"""Generate a dataset of short English fairy tales via the OpenRouter API.

Each example pairs a structured input (hero, setting, problem, helper_or_item,
lesson) with a freshly generated short story, written one-per-line as JSONL:

    {"input": {"hero": "cat", ...}, "story": "Pip the cat..."}

The script is resumable: it reads whatever is already in the output file,
figures out how many stories per combination are still missing, and only
generates the deficit. Output is appended and flushed continuously, so it is
safe to stop with Ctrl-C and re-run later without producing duplicates.

Usage:
    # OPENROUTER_API_KEY is read from .env (or the environment)
    python generate_dataset.py --target 300000 --workers 8

Only the Python standard library is required (talks to OpenRouter over HTTP).
"""

import argparse
import itertools
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque

# Matches most emoji / pictographic / symbol code points.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "]+",
    flags=re.UNICODE,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# Allowed providers, in priority order. With allow_fallbacks off, OpenRouter only
# routes to these and only in this order. Groq is the fastest for this workload
# (~33 valid stories/s at 32 workers), so it is the default. If you hit sustained
# 429 stalls, broaden to "groq,deepinfra,novita" and pass --allow-fallbacks so
# requests spill over to the other providers under load.
DEFAULT_PROVIDERS = "groq"

# --------------------------------------------------------------------------- #
# Parameter space
# --------------------------------------------------------------------------- #
PARAMS = {
    "hero": ["cat", "dog", "rabbit", "fox", "bear", "mouse", "dragon", "robot", "star", "cloud"],
    "setting": ["forest", "moon", "village", "castle", "garden", "sea", "mountain", "school", "dreamland", "snowy field"],
    "problem": ["lost friend", "dark night", "broken toy", "missing key", "scary noise", "sad day", "big storm", "forgotten promise", "lonely creature", "hard choice"],
    "helper_or_item": ["magic leaf", "tiny lamp", "wise owl", "golden bell", "kind fairy", "talking stone", "little boat", "rainbow rope", "warm scarf", "silver star"],
    "lesson": ["kindness", "sharing", "bravery", "honesty", "patience", "friendship", "helping others", "not giving up", "saying sorry", "being careful"],
}
# Order in which fields are read/written. Kept fixed so dedup keys are stable.
FIELDS = ["hero", "setting", "problem", "helper_or_item", "lesson"]

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# The model writes a 4-sentence STORY; the 5th sentence (the moral) is added by
# us from MORAL_TEMPLATES, so it is always present, varied, and well-formed.
SYSTEM_PROMPT = (
    "You write tiny stories for children ages 3-7: exactly 5 short, simple "
    "sentences in plain words (each under ~15 words).\n"
    "Shape: (1) name the hero and call it the given animal/thing, in the setting; "
    "(2) the given problem happens; (3) the helper/item solves that problem; "
    "(4) a happy ending where the hero shows the lesson by what it does (use a "
    "word matching the lesson); (5) one short closing line about the lesson.\n"
    "Rules: use the exact hero, setting, problem and helper/item; no other "
    "characters or extra details; don't start with 'Once upon a time'; no title, "
    "lists, markdown, emojis or '...'; end every sentence with a period; output "
    "ONLY the 5 sentences."
)

# One few-shot example anchors the 5-sentence shape (sentence 4 names the lesson
# word). Kept to a single example to cut the repeated prompt cost; validation
# catches any format slips.
FEWSHOT = [
    ("Hero: rabbit. Setting: garden. Problem: broken toy. "
     "Helper/item: kind fairy. Lesson: sharing.",
     "Pip the rabbit was playing in the sunny garden. Her little toy cart "
     "cracked in two and she felt sad. A kind fairy fluttered down and mended "
     "the cart with a gentle touch. Pip happily shared the cart and let all her "
     "friends take turns. Sharing made the whole garden merry."),
]

# Varied, natural last-sentence morals. {hero} is lowercased ("the fox"), {lesson}
# is a noun/phrase from PARAMS. Each reads correctly with every lesson value.
MORAL_TEMPLATES = [
    "The lesson was: {lesson}.",
    "And so the {hero} learned all about {lesson}.",
    "From that day on, the {hero} always remembered {lesson}.",
    "That is how the {hero} discovered the value of {lesson}.",
    "In the end, it was {lesson} that saved the day.",
    "The {hero} was so glad to have learned about {lesson}.",
    "And the {hero} never forgot the importance of {lesson}.",
    "Everyone could see that {lesson} had made everything right.",
]


# Keyword stems/synonyms that must appear in the STORY so the lesson is genuinely
# shown (not just stated in the appended moral). Stories without one are retried.
# Kept generous on purpose: the model often expresses the lesson with a synonym,
# and rejecting those good paraphrases just wastes attempts (the moral sentence
# already states the lesson explicitly).
LESSON_CUES = {
    "kindness": ["kind", "gentle", "caring", "cared", "care", "nice", "warm",
                 "comfort", "sweet", "help", "hug", "smile", "love"],
    "sharing": ["shar", "gave", "give", "took turns", "together", "split",
                "offered", "handed", "passed", "everyone", "each other",
                "let them"],
    "bravery": ["brave", "braver", "courage", "bold", "fearless", "dared",
                "stood up", "faced", "no fear", "not afraid", "strong",
                "kept going", "without fear"],
    "honesty": ["honest", "truth", "true", "did not lie", "didn't lie", "admit",
                "told no lie", "owned up", "confess"],
    "patience": ["patien", "wait", "calm", "slowly", "took her time",
                 "took his time", "took its time", "step by step", "one by one"],
    "friendship": ["friend", "together", "befriend", "pal", "buddy",
                   "companion"],
    "helping others": ["help", "rescu", "saved", "assist", "aid", "lend",
                       "support", "came to", "guid", "led", "show", "care",
                       "comfort", "gave"],
    "not giving up": ["give up", "giving up", "gave up", "kept", "keep", "again",
                      "continued", "determinat", "persever", "never stopped",
                      "didn't stop", "did not stop", "tried hard", "harder",
                      "tried again", "would not quit", "tried", "trying", "try",
                      "effort", "struggl", "no matter", "even though"],
    "saying sorry": ["sorry", "apolog", "forgave", "forgive", "made up",
                     "regret"],
    "being careful": ["careful", "carefully", "cautio", "safe", "slowly",
                      "watch out", "looked both", "gently", "step by step"],
}

# Synonyms/stems that count as the PROBLEM being present in the story. The model
# routinely paraphrases the problem (e.g. "big storm" -> "tempest", "scary noise"
# -> "frightening sound", "missing key" -> "lost key"); those stories are about
# the right problem, so we accept the paraphrase instead of demanding the literal
# words. This was by far the biggest cause of (wasteful) validation failures.
PROBLEM_CUES = {
    "lost friend": ["lost", "friend", "gone", "missing", "disappear", "vanish",
                    "could not find", "couldn't find", "nowhere"],
    "dark night": ["dark", "night", "no light", "no moon", "no stars",
                   "black sky", "pitch", "gloomy"],
    "broken toy": ["broke", "broken", "toy", "crack", "shatter", "snapped",
                   "in pieces", "fell apart", "torn", "ripped"],
    "missing key": ["miss", "key", "lost", "gone", "could not find",
                    "couldn't find", "nowhere", "vanish"],
    "scary noise": ["scar", "nois", "sound", "fright", "loud", "spooky",
                    "strange", "rumble", "bang", "creak"],
    "sad day": ["sad", "unhapp", "gloom", "cry", "cried", "tear", "down",
                "blue", "upset", "glum"],
    "big storm": ["storm", "rain", "wind", "thunder", "lightning", "gale",
                  "tempest", "downpour", "blizzard", "snow"],
    "forgotten promise": ["forgot", "forgotten", "promise", "vow", "did not keep",
                          "didn't keep", "broke a"],
    "lonely creature": ["lone", "alone", "lonely", "by itself", "by himself",
                        "by herself", "no friends", "all alone", "no one"],
    "hard choice": ["hard", "choice", "choose", "chose", "decide", "decision",
                    "difficult", "tough", "dilemma", "torn between"],
}


def lesson_shown(story_low, lesson):
    cues = LESSON_CUES.get(lesson)
    if not cues:
        return True
    return any(c in story_low for c in cues)


def problem_shown(story_low, problem):
    cues = PROBLEM_CUES.get(problem)
    if not cues:
        return _present(problem, story_low, 3)
    return any(c in story_low for c in cues)


def make_moral(combo):
    return random.choice(MORAL_TEMPLATES).format(
        hero=combo["hero"], lesson=combo["lesson"])


# Some problems are abstract and the 8B model can't render them on its own -- it
# silently swaps them for a problem that fits the lesson (e.g. "hard choice" +
# "saying sorry" -> an accident-and-apology story), which then fails validation
# and wastes the whole generation. A short gloss tells the model what the problem
# means so it actually writes that problem. The gloss is PROMPT-ONLY -- the
# dataset's `problem` field is unchanged. (Cut problem-missing on these combos
# from ~18% to ~5% in testing.)
PROBLEM_HINTS = {
    "hard choice": " (the hero must choose between two good things and can pick only one)",
    "lonely creature": " (a creature that is all alone and longs for a friend)",
}


def build_user_prompt(combo):
    hint = PROBLEM_HINTS.get(combo["problem"], "")
    return (
        "Hero: {hero}. Setting: {setting}. Problem: {problem}{hint}. "
        "Helper/item: {helper_or_item}. Lesson: {lesson}."
    ).format(hint=hint, **combo)


def build_messages(combo):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_user, ex_assistant in FEWSHOT:
        msgs.append({"role": "user", "content": ex_user})
        msgs.append({"role": "assistant", "content": ex_assistant})
    msgs.append({"role": "user", "content": build_user_prompt(combo)})
    return msgs


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BAD_STARTS = ("once upon a time", "once there was", "once there lived",
               "long ago", "in a land far", "in a faraway")
_MORAL_CUE = re.compile(
    r"\b(lesson|moral|learn(ed|t)?\b.*\b(that|to|about)|remember(ed)?|"
    r"never forgot|taught (us|them|him|her))\b", re.I)
_BANNED_SUBSTR = ("```", "...", "…", "{", "}", "#", "*", "|", "</", "http",
                  '"input"', '"story"')


def _split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def clean_raw(text):
    """Collapse whitespace/newlines and strip emojis + wrapping quotes."""
    if not text:
        return ""
    text = _EMOJI_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    text = " ".join(text.split())                 # newlines/runs -> single spaces
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)   # "word ." -> "word."
    return text


def _present(field, low, min_len):
    """True if any word of `field` (>= min_len chars) appears in `low`, matched by
    a 4-char prefix so inflections count (e.g. 'broken' matches 'broke')."""
    words = [w for w in field.lower().split() if len(w) >= min_len]
    if not words:
        return True
    return any((w[:4] if len(w) >= 4 else w) in low for w in words)


def validate_story(raw, combo):
    """Return (body_or_None, reason). `body` is exactly 4 clean story sentences;
    the moral (5th sentence) is appended later. Robust to the model writing 4 or
    5+ sentences: a trailing model-moral is dropped and the first 4 are kept."""
    text = clean_raw(raw)
    if not text:
        return None, "empty"
    full_low = text.lower()        # whole model output (before we drop sentence 5)
    bad = next((b for b in _BANNED_SUBSTR if b in full_low), None)
    if bad:
        return None, "banned:" + bad
    if not text.endswith((".", "!", "?")):
        return None, "truncated"
    sents = _split_sentences(text)
    # Drop a trailing moral the model added (we append our own controlled one).
    if len(sents) >= 5 and _MORAL_CUE.search(sents[-1]):
        sents = sents[:-1]
    if len(sents) < 4:
        return None, "too-short=%d" % len(sents)
    sents = sents[:4]                              # normalise to exactly 4
    story = " ".join(sents)
    low = story.lower()
    if low.startswith(_BAD_STARTS):
        return None, "template-start"
    if any(len(s.split()) > 24 for s in sents):
        return None, "long-sentence"
    wc = len(story.split())
    if wc < 24 or wc > 90:
        return None, "wordcount=%d" % wc
    if combo["hero"].lower() not in low:
        return None, "hero-missing"
    if not _present(combo["helper_or_item"], low, 3):
        return None, "helper-missing"
    if not problem_shown(low, combo["problem"]):
        return None, "problem-missing"
    # The lesson can be shown anywhere the model wrote it -- often in the closing
    # sentence we drop -- so check the FULL output, not just the 4-sentence body.
    # This stops us discarding good stories whose lesson word lands in sentence 5.
    if not lesson_shown(full_low, combo["lesson"]):
        return None, "lesson-not-shown"
    return story, "ok"


# --------------------------------------------------------------------------- #
# OpenRouter client
# --------------------------------------------------------------------------- #
class APIError(Exception):
    def __init__(self, msg, kind="other", retry_after=None):
        super().__init__(msg)
        self.kind = kind  # "rate" | "server" | "conn" | "auth" | "client" | "other"
        self.retry_after = retry_after


def openrouter_chat(api_key, model, messages, temperature, max_tokens, provider, timeout):
    """Single chat completion via OpenRouter. Returns (text, completion_tokens)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    if provider:
        payload["provider"] = provider
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        ra = e.headers.get("Retry-After")
        ra = float(ra) if ra and str(ra).isdigit() else None
        if e.code == 429:
            raise APIError("rate limit: " + detail, kind="rate", retry_after=ra)
        if e.code in (401, 403):
            raise APIError("auth/forbidden: " + detail, kind="auth")
        if 500 <= e.code < 600:
            raise APIError("server %d: %s" % (e.code, detail), kind="server")
        raise APIError("http %d: %s" % (e.code, detail), kind="client")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise APIError("connection error: %s" % e, kind="conn")
    except json.JSONDecodeError as e:
        raise APIError("bad JSON from server: %s" % e, kind="other")

    # OpenRouter sometimes returns a provider error inside a 200 body.
    if isinstance(body, dict) and body.get("error"):
        raise APIError("api error: %s" % str(body["error"])[:300], kind="server")
    choices = body.get("choices") or []
    if not choices:
        raise APIError("no choices in response", kind="server")
    text = (choices[0].get("message") or {}).get("content", "") or ""
    usage = body.get("usage") or {}
    tokens = int(usage.get("completion_tokens", 0) or 0)
    return text, tokens


def ollama_chat(host, model, messages, temperature, max_tokens, num_ctx, timeout):
    """Single chat completion via a local Ollama server. Returns (text, tokens)."""
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        kind = "server" if e.code >= 500 else "client"
        raise APIError("http %d: %s" % (e.code, detail), kind=kind)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise APIError("connection error: %s" % e, kind="conn")
    except json.JSONDecodeError as e:
        raise APIError("bad JSON from server: %s" % e, kind="other")

    if isinstance(body, dict) and body.get("error"):
        raise APIError("ollama error: %s" % str(body["error"])[:300], kind="server")
    text = (body.get("message") or {}).get("content", "") or ""
    return text, int(body.get("eval_count", 0) or 0)


def deepseek_chat(api_key, url, model, messages, temperature, max_tokens, timeout):
    """Single chat completion via DeepSeek (OpenAI-compatible). Returns
    (text, completion_tokens). DeepSeek auto-caches the repeated prompt prefix."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        ra = e.headers.get("Retry-After")
        ra = float(ra) if ra and str(ra).isdigit() else None
        if e.code == 429:
            raise APIError("rate limit: " + detail, kind="rate", retry_after=ra)
        if e.code in (401, 403):
            raise APIError("auth: " + detail, kind="auth")
        if e.code == 402:  # out of balance -- fail fast, don't spin
            raise APIError("payment required (out of credits): " + detail, kind="auth")
        if 500 <= e.code < 600:
            raise APIError("server %d: %s" % (e.code, detail), kind="server")
        raise APIError("http %d: %s" % (e.code, detail), kind="client")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise APIError("connection error: %s" % e, kind="conn")
    except json.JSONDecodeError as e:
        raise APIError("bad JSON from server: %s" % e, kind="other")

    if isinstance(body, dict) and body.get("error"):
        raise APIError("api error: %s" % str(body["error"])[:300], kind="server")
    choices = body.get("choices") or []
    if not choices:
        raise APIError("no choices in response", kind="server")
    text = (choices[0].get("message") or {}).get("content", "") or ""
    usage = body.get("usage") or {}
    return text, int(usage.get("completion_tokens", 0) or 0)


def generate_story(call, combo, retries):
    """Generate one strictly-validated 5-sentence story (4 story sentences from
    the model + 1 natural moral we append).

    Returns (status, story, tokens, attempts):
      ("ok",   story, tokens, attempts) | ("fail", None, 0, attempts).
    `attempts` is a list of (reason, raw) for each try, so the caller can log why
    a story was rejected/retried: reason is "ok", a validation reason
    (e.g. "lesson-not-shown"), or "api:<kind>"; `raw` is the model's text for
    rejected attempts and None otherwise.

    `call(messages) -> (text, tokens)` is the backend (OpenRouter or Ollama).
    Transient errors (rate limit, server, connection) back off and retry; auth
    errors fail fast (preflight already covers a bad key at startup)."""
    messages = build_messages(combo)
    attempts = []
    for attempt in range(retries):
        try:
            raw, tokens = call(messages)
        except APIError as e:
            attempts.append(("api:" + e.kind, None))
            if e.kind == "auth":
                return ("fail", None, 0, attempts)
            if e.kind == "rate" and e.retry_after:
                time.sleep(min(e.retry_after, 30))
            else:
                time.sleep(min(2 ** attempt, 30))
            continue
        story, reason = validate_story(raw, combo)
        if story is not None:
            attempts.append(("ok", None))
            return ("ok", story + " " + make_moral(combo), tokens, attempts)
        attempts.append((reason, raw))
        time.sleep(0.05)
    return ("fail", None, 0, attempts)


# --------------------------------------------------------------------------- #
# Output / resume bookkeeping
# --------------------------------------------------------------------------- #
def combo_key(combo):
    return tuple(combo[f] for f in FIELDS)


def read_existing(path):
    """Return (Counter of combo_key -> count, total_lines) from an existing file."""
    counts = Counter()
    total = 0
    if not os.path.exists(path):
        return counts, total
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                inp = obj["input"]
                key = tuple(inp[f] for f in FIELDS)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # skip a partial/corrupt line from a previous crash
            counts[key] += 1
            total += 1
    return counts, total


class Writer:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, combo, story):
        record = {"input": {f: combo[f] for f in FIELDS}, "story": story}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        with self._lock:
            self._fh.close()


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def all_combos():
    values = [PARAMS[f] for f in FIELDS]
    for tup in itertools.product(*values):
        yield dict(zip(FIELDS, tup))


def build_plan(target, existing_counts, seed):
    """Decide how many *more* stories each combo needs to reach `target`.

    The target is distributed evenly across all combinations. A deterministic
    shuffle (seeded) keeps the plan stable across resumes.

    Returns (task_generator, still_needed_total).
    """
    combos = list(all_combos())
    n = len(combos)
    rng = random.Random(seed)
    rng.shuffle(combos)  # deterministic for a given seed -> stable across resumes

    base, rem = divmod(target, n)

    needed = []
    total_needed = 0
    for i, combo in enumerate(combos):
        want = base + (1 if i < rem else 0)        # this combo's quota
        have = existing_counts.get(combo_key(combo), 0)
        deficit = max(0, want - have)
        if deficit:
            needed.append((combo, deficit))
            total_needed += deficit

    def task_gen():
        for combo, deficit in needed:
            for _ in range(deficit):
                yield combo

    return task_gen(), total_needed


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #
def load_dotenv(path):
    """Minimal .env loader: KEY=VALUE lines into os.environ (no overwrite)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
    except OSError:
        pass


def human_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "{}h{:02d}m".format(h, m)
    if m:
        return "{}m{:02d}s".format(m, s)
    return "{}s".format(s)


def setup_logging(path):
    """File logger for fails + reason tallies (append mode, so it survives
    resumes). Console output stays the live progress line."""
    logger = logging.getLogger("gen")
    logger.setLevel(logging.INFO)
    logger.handlers = []          # avoid duplicate handlers if run() re-enters
    logger.propagate = False
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def preflight(call, model):
    """One tiny call to confirm the backend, key and model work."""
    try:
        call([{"role": "user", "content": "Say hi."}])
        return True
    except APIError as e:
        if e.kind == "auth":
            sys.stderr.write("ERROR: OpenRouter auth failed -- check "
                             "OPENROUTER_API_KEY.\n  %s\n" % e)
        elif e.kind == "conn":
            sys.stderr.write("ERROR: backend unreachable for model '%s' "
                             "(is Ollama running / is the model pulled?).\n  %s\n"
                             % (model, e))
        else:
            sys.stderr.write("ERROR: preflight failed for model '%s'.\n  %s\n"
                             % (model, e))
        return False


def run(args):
    # Load .env from the script's directory and the current directory.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    load_dotenv(".env")

    # Build the backend call: call(messages) -> (text, tokens).
    if args.backend == "ollama":
        model = args.ollama_model
        def call(messages):
            return ollama_chat(args.ollama_host, model, messages, args.temperature,
                               args.max_tokens, args.num_ctx, args.timeout)
        print("Checking Ollama ({} | {})...".format(model, args.ollama_host))
    elif args.backend == "deepseek":
        model = args.deepseek_model
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            sys.stderr.write("ERROR: no API key. Put DEEPSEEK_API_KEY in .env, "
                             "set it in the environment, or pass --api-key.\n")
            return 1
        def call(messages):
            return deepseek_chat(api_key, args.deepseek_url, model, messages,
                                 args.temperature, args.max_tokens, args.timeout)
        print("Checking DeepSeek ({} | {})...".format(model, args.deepseek_url))
    else:
        model = args.model
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.stderr.write("ERROR: no API key. Put OPENROUTER_API_KEY in .env, "
                             "set it in the environment, or pass --api-key.\n")
            return 1
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        provider = {"order": providers, "allow_fallbacks": args.allow_fallbacks} \
            if providers else None
        def call(messages):
            return openrouter_chat(api_key, model, messages, args.temperature,
                                   args.max_tokens, provider, args.timeout)
        print("Checking OpenRouter ({} | providers: {})...".format(
            model, ", ".join(providers) or "any"))

    if not preflight(call, model):
        return 1

    print("Reading existing output (if any)...")
    existing_counts, existing_total = read_existing(args.out)
    print("Already have {:,} examples in {}".format(existing_total, args.out))

    tasks, total_needed = build_plan(args.target, existing_counts, args.seed)
    if total_needed == 0:
        print("Target already met. Nothing to do.")
        return 0

    log = setup_logging(args.log)
    log.info("START backend=%s model=%s workers=%d target=%d out=%s",
             args.backend, model, args.workers, args.target, args.out)

    print("Backend: {} | model: {} | workers: {}".format(
        args.backend, model, args.workers))
    print("Need {:,} more (target {:,}). Writing to {} (resumable, Ctrl-C to "
          "stop). Log: {}\n".format(total_needed, args.target, args.out, args.log))

    writer = Writer(args.out)
    stop = threading.Event()
    stats_lock = threading.Lock()
    task_lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "tokens": 0}
    reasons = Counter()               # every attempt's reason (ok / validation / api)
    recent = deque(maxlen=400)        # timestamps of recent successes (windowed ex/s)
    start = time.time()

    def reason_summary():
        with stats_lock:
            items = reasons.most_common()
        total = sum(c for _, c in items) or 1
        return ", ".join("%s=%d(%.0f%%)" % (r, c, 100.0 * c / total)
                         for r, c in items) or "(none yet)"

    def get_task():
        with task_lock:
            try:
                return next(tasks)
            except StopIteration:
                return None

    def worker():
        while not stop.is_set():
            combo = get_task()
            if combo is None:
                return
            status, story, tokens, attempts = generate_story(
                call, combo, args.retries)
            with stats_lock:
                for reason, _raw in attempts:
                    reasons[reason] += 1
            combo_s = json.dumps(combo, ensure_ascii=False)
            if status == "ok":
                writer.write(combo, story)
                with stats_lock:
                    counters["ok"] += 1
                    counters["tokens"] += tokens
                    recent.append(time.time())
                if args.debug:                 # log retries even when it succeeded
                    for reason, raw in attempts:
                        if reason != "ok":
                            log.info("retry %-18s %s | raw=%r", reason, combo_s,
                                     raw[:300] if raw else None)
            else:
                with stats_lock:
                    counters["fail"] += 1
                log.warning("FAIL after %d attempt(s): %s", len(attempts), combo_s)
                for reason, raw in attempts:
                    log.warning("    %-18s raw=%r", reason,
                                raw[:300] if raw else None)

    def progress():
        now = time.time()
        elapsed = now - start
        with stats_lock:
            ok, fail, toks = counters["ok"], counters["fail"], counters["tokens"]
            rec = list(recent)
        if len(rec) >= 2 and rec[-1] > rec[0]:
            exrate = (len(rec) - 1) / (rec[-1] - rec[0])
        else:
            exrate = ok / elapsed if elapsed > 0 else 0.0
        tokrate = toks / elapsed if elapsed > 0 else 0.0
        eta = (total_needed - ok) / exrate if exrate > 0 else 0
        pct = 100.0 * ok / total_needed if total_needed else 100.0
        sys.stdout.write(
            "\r  {ok:,}/{tot:,} ({pct:4.1f}%) | {ex:5.2f} ex/s | {tok:5.0f} tok/s | "
            "ETA {eta} | fail {fail:,} | file {filetot:,}   ".format(
                ok=ok, tot=total_needed, pct=pct, ex=exrate, tok=tokrate,
                eta=human_time(eta) if exrate > 0 else "?", fail=fail,
                filetot=existing_total + ok))
        sys.stdout.flush()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()

    exit_code = 0
    last_log = [start]
    try:
        while any(t.is_alive() for t in threads):
            progress()
            if time.time() - last_log[0] > 60:   # periodic tally into the log
                last_log[0] = time.time()
                log.info("progress ok=%d fail=%d | reasons: %s",
                         counters["ok"], counters["fail"], reason_summary())
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop.set()
        sys.stdout.write("\n\nStopping (resumable)...\n")
        exit_code = 130
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=args.timeout + 5)
        writer.close()
        progress()

    summary = reason_summary()
    log.info("DONE ok=%d fail=%d | reasons: %s",
             counters["ok"], counters["fail"], summary)
    print("\n\nDone. Wrote {:,} valid examples ({:,} failed) this run.".format(
        counters["ok"], counters["fail"]))
    print("Attempt reasons: {}".format(summary))
    print("File now contains {:,} examples: {}".format(
        existing_total + counters["ok"], args.out))
    print("Log written to {}".format(args.log))
    if counters["fail"]:
        print("Re-run the same command to retry the {:,} that failed.".format(
            counters["fail"]))
    return exit_code


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate short English fairy tales with OpenRouter into JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=int, default=300_000,
                   help="Total number of examples wanted in the output file.")
    p.add_argument("--out", default="data/stories.jsonl",
                   help="Output JSONL path (appended to, resumable).")
    p.add_argument("--workers", type=int, default=32,
                   help="Number of concurrent requests. Throughput plateaus near "
                        "32 for Groq; 64 is not faster (more 429s + latency).")
    p.add_argument("--backend", choices=["openrouter", "ollama", "deepseek"],
                   default="openrouter",
                   help="Where to generate: 'openrouter' (cloud), 'deepseek' "
                        "(cloud), or 'ollama' (local server, free).")
    p.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct",
                   help="OpenRouter model id (used when --backend openrouter).")
    p.add_argument("--deepseek-model", default="deepseek-chat",
                   help="DeepSeek model id (used when --backend deepseek).")
    p.add_argument("--deepseek-url", default=DEEPSEEK_URL,
                   help="DeepSeek chat completions endpoint.")
    p.add_argument("--ollama-model", default="llama3.1:8b",
                   help="Ollama model tag (used when --backend ollama).")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="Ollama server URL (used when --backend ollama).")
    p.add_argument("--num-ctx", type=int, default=2048,
                   help="Ollama context window (fits the few-shot prompt + story).")
    p.add_argument("--providers", default=DEFAULT_PROVIDERS,
                   help="Comma list of allowed OpenRouter providers in priority "
                        "order (empty = let OpenRouter choose).")
    p.add_argument("--allow-fallbacks", action="store_true",
                   help="Allow OpenRouter to fall back to providers outside the "
                        "--providers list (off by default: only those allowed).")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter key (defaults to OPENROUTER_API_KEY from .env "
                        "or the environment).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (lower = more stable/consistent).")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Max tokens per story (enough for 5 short sentences).")
    p.add_argument("--retries", type=int, default=3,
                   help="Max attempts per story before counting it failed (a CAP, "
                        "not a fixed count: ~93%% succeed on attempt 1). Lower = "
                        "give up faster on stubborn combos and fix them on a "
                        "re-run, instead of burning attempts.")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-request timeout in seconds.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the deterministic combo shuffle (stable resumes).")
    p.add_argument("--log", default="generate.log",
                   help="Run log: every final FAIL (with the combo + each "
                        "attempt's reason and the model's raw text) plus periodic "
                        "reason tallies. Appended to across resumes.")
    p.add_argument("--debug", action="store_true",
                   help="Also log every rejected RETRY attempt (with raw text), "
                        "not just final fails. Verbose -- use for investigation.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
