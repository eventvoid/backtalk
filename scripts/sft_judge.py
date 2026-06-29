#!/usr/bin/env python3
"""Strict semantic QA judge for BackTalk SFT, powered by DeepSeek.

Reads every data/sft_chunks/*.jsonl, judges each example keep/drop on fluency,
coherence, category match, safety, follow-up endings and usefulness, and writes
the survivors to data/sft_chunks_clean/<same-name>.jsonl (category preserved via
filename). Dropped examples are logged to data/judge_drops.jsonl with reasons.

Live metrics: examples/sec, completion tokens/sec, ETA, and DeepSeek balance delta.
"""
import os, sys, json, re, time, glob, threading
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.environ.get("SFT_SRC_DIR", os.path.join(ROOT, "data", "sft_chunks"))
OUT_DIR = os.environ.get("SFT_OUT_DIR", os.path.join(ROOT, "data", "sft_chunks_clean"))
DROP_LOG = os.environ.get("SFT_DROP_LOG", os.path.join(ROOT, "data", "judge_drops.jsonl"))
API_URL = "https://api.deepseek.com/chat/completions"
BAL_URL = "https://api.deepseek.com/user/balance"
MODEL = "deepseek-chat"
WORKERS = 32
BATCH = 10  # examples judged per API call

def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("DEEPSEEK_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key")
KEY = load_key()

def get_balance():
    try:
        req = urllib.request.Request(BAL_URL, headers={"Authorization": f"Bearer {KEY}"})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        for b in d.get("balance_infos", []):
            return f"{b.get('total_balance')} {b.get('currency')}"
    except Exception as e:
        return f"(balance n/a: {type(e).__name__})"

RUBRIC = (
    "You are a STRICT quality reviewer for an English instruction-tuning dataset for a small "
    "assistant called BackTalk. For each numbered example you get a user message and the "
    "assistant reply. Decide keep or drop.\n"
    "DROP the example if ANY of these is true:\n"
    "- the assistant reply is not fluent, natural, standard English;\n"
    "- the reply is incoherent, rambling, or word-salad (real words in nonsense order);\n"
    "- broken grammar, garbled phrases, or it reads like it was scrambled;\n"
    "- the answer is truncated, cut off, or incomplete;\n"
    "- it does not actually answer or help with the user's request;\n"
    "- wrong category / topic mismatch, or it ignores what was asked;\n"
    "- it gives unsafe, dangerous, or harmful advice;\n"
    "- it ends with a generic filler follow-up question (e.g. 'Anything else?', 'Want me to help?');\n"
    "- it is template/boilerplate or obviously formulaic;\n"
    "- it states confidently wrong or clearly hallucinated facts;\n"
    "- the USER message itself is incoherent nonsense (NOT just casual typos) that cannot be "
    "reasonably understood;\n"
    "- the assistant TRANSLATES between languages or produces any non-English text: BackTalk "
    "is English-only and must never translate or answer 'how do you say X in <language>';\n"
    "- the assistant role-plays as a doctor, lawyer, therapist, or financial advisor giving "
    "definitive professional advice or a diagnosis, instead of general info plus a suggestion "
    "to see a qualified professional (gentle everyday emotional support is fine to keep).\n"
    "KEEP only genuinely fluent, correct, useful, complete single-turn answers in BackTalk's "
    "simple, clear, natural, helpful voice. Casual or lightly-misspelled USER messages are FINE "
    "as long as they are understandable and the assistant reply is clean.\n"
    "Be strict: when in doubt, DROP.\n"
    'Output ONLY a JSON array, one object per example: {"i": <number>, "v": "keep"|"drop"}. '
    "No other text."
)

if os.environ.get("SFT_STRICT"):
    RUBRIC += (
        "\n\nEXTRA STRICT PASS. Also DROP if ANY of these is even slightly true:\n"
        "- any awkward, clunky, stilted, or unnatural phrasing or a grammar slip, even minor;\n"
        "- any specific fact (date, etymology, statistic, product name, place, dosage) that is "
        "not clearly correct, or that was invented rather than present in the user's source text;\n"
        "- it performs an action the user did NOT ask for (e.g. translating when only asked what a "
        "phrase means, or vice versa);\n"
        "- a clipped, abrupt, or trailing-off ending;\n"
        "- anything a careful human editor would rewrite before publishing.\n"
        "Hold a very high bar: KEEP only answers that read as clean, natural, correct published prose."
    )

stats = {"calls": 0, "judged": 0, "kept": 0, "ctok": 0, "ptok": 0, "start": time.time()}
slock = threading.Lock()
wlock = threading.Lock()

def call(messages, temp=0.0, max_tokens=1500):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": temp, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for a in range(4):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=180).read())
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and a < 3:
                time.sleep(2 ** a); continue
            return "", {}
        except Exception:
            if a < 3:
                time.sleep(2 ** a); continue
            return "", {}
    return "", {}

def parse_verdicts(text, n):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group())
            out = {}
            for o in arr:
                out[int(o["i"])] = str(o.get("v", "drop")).lower()
            return out
        except Exception:
            pass
    return {}

def judge_batch(items, out_fh, drop_fh):
    # items: list of (idx, obj, cat)
    lines = []
    for k, (idx, o, cat) in enumerate(items, 1):
        u = o["user"].replace("\n", " ")[:600]
        a = o["assistant"].replace("\n", " ")[:1200]
        lines.append(f"#{k} [category: {cat}]\nUSER: {u}\nASSISTANT: {a}")
    content = RUBRIC + "\n\n" + "\n\n".join(lines)
    text, usage = call([{"role": "user", "content": content}])
    verdicts = parse_verdicts(text, len(items))
    kept = 0
    with wlock:
        for k, (idx, o, cat) in enumerate(items, 1):
            v = verdicts.get(k, "drop")  # missing verdict -> drop (strict)
            rec = {"user": o["user"], "assistant": o["assistant"]}
            if v == "keep":
                out_fh[cat].write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                drop_fh.write(json.dumps({"cat": cat, **rec}, ensure_ascii=False) + "\n")
    with slock:
        stats["calls"] += 1
        stats["judged"] += len(items)
        stats["kept"] += kept
        stats["ctok"] += usage.get("completion_tokens", 0)
        stats["ptok"] += usage.get("prompt_tokens", 0)
    return kept

def printer(total, stop):
    while not stop.is_set():
        time.sleep(12)
        with slock:
            el = time.time() - stats["start"]
            j = stats["judged"]; k = stats["kept"]; ct = stats["ctok"]
        sps = j / el if el else 0
        tps = ct / el if el else 0
        eta = (total - j) / sps if sps > 0 else 0
        keep_rate = 100 * k / j if j else 0
        print(f"[{int(el)}s] judged {j}/{total} ({100*j/total:.0f}%) | keep {k} ({keep_rate:.0f}%) "
              f"| {sps:.0f} sft/s | {tps:.0f} ctok/s | ETA {int(eta)}s", flush=True)

STOP = set(("a an the of to in on at for and or but so if is are was were be been being am "
            "i you he she it we they me him her them my your his our their this that these those "
            "with from by as into about up out down over under not no do does did have has had "
            "can could should would will just than then too very more most some any what why how "
            "when where who which because while there here its it's you're i'm").split())

TRANSLATE_RE = re.compile(
    r"\btranslat|how (do|to) (you |i )?say\b|\bin (spanish|french|german|italian|"
    r"portuguese|japanese|chinese|russian|korean|arabic|latin|dutch|polish)\b|"
    r"what does .* mean in (english|spanish|french|german|italian)", re.I)

def heuristic_bad(o):
    """Free, conservative pre-filter for near-certain garbage (saves judge budget)."""
    u, a = o["user"], o["assistant"]
    if "�" in u or "�" in a:           # mojibake / encoding damage
        return "mojibake"
    if TRANSLATE_RE.search(u):                     # BackTalk must not translate
        return "translation"
    if a.rstrip().endswith("?"):                  # generic follow-up ending
        return "ends_with_question"
    words = re.findall(r"[A-Za-z']+", a)
    if len(words) >= 25:                          # only judge soup on long answers
        ratio = sum(1 for w in words if w.lower() in STOP) / len(words)
        if ratio < 0.10:                          # noun-pile / word-salad signature
            return "low_function_word_ratio"
    return None

def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl")))
    def cat_of(p):
        n = os.path.basename(p).lower()
        for key, c in [("identity","identity"),("troubleshoot","troubleshooting"),
                       ("comparison","comparisons"),("emotional","emotional"),
                       ("unclear","unclear"),("safety","safety"),("rewrite","rewrite"),
                       ("explain","explanations"),("explanation","explanations"),
                       ("everyday","everyday")]:
            if key in n:
                return c
        return "everyday"

    import random
    items = []
    cats = set()
    heur_drops = {}
    drop_fh = open(DROP_LOG, "w", encoding="utf-8")
    for f in files:
        c = cat_of(f)
        if only and c not in only:
            continue
        cats.add(c)
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if set(o) != {"user", "assistant"} or not o["user"].strip() or not o["assistant"].strip():
                continue
            reason = heuristic_bad(o)
            if reason:
                heur_drops[reason] = heur_drops.get(reason, 0) + 1
                drop_fh.write(json.dumps({"cat": c, "heuristic": reason, **o}, ensure_ascii=False) + "\n")
                continue
            items.append((len(items), o, c))

    random.Random(20260627).shuffle(items)  # so a budget cutoff still covers all categories
    total = len(items)
    bal0 = get_balance()
    try:
        bal_usd = float(bal0.split()[0])
    except Exception:
        bal_usd = 1.0
    RESERVE = 0.15
    PER_BATCH = 0.0006  # conservative
    affordable = max(0, int((bal_usd - RESERVE) / PER_BATCH))
    n_batches_all = (total + BATCH - 1) // BATCH
    cap = min(n_batches_all, affordable)
    print(f"Heuristic free-drops: {heur_drops}", flush=True)
    print(f"To judge: {total} | balance {bal0} | affordable batches {affordable} | "
          f"running {cap}/{n_batches_all} batches (covers {min(total, cap*BATCH)} examples)", flush=True)

    out_fh = {c: open(os.path.join(OUT_DIR, f"clean_{c}.jsonl"), "w", encoding="utf-8") for c in cats}

    all_batches = [items[i:i+BATCH] for i in range(0, total, BATCH)]
    batches = all_batches[:cap]
    # examples we cannot afford to judge are excluded (not vouched for)
    unjudged = sum(len(b) for b in all_batches[cap:])
    stop = threading.Event()
    judged_target = min(total, cap * BATCH)
    pt = threading.Thread(target=printer, args=(judged_target, stop), daemon=True); pt.start()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(judge_batch, b, out_fh, drop_fh) for b in batches]
        for _ in as_completed(futs):
            pass
    stop.set()
    for fh in out_fh.values():
        fh.close()
    drop_fh.close()
    el = time.time() - stats["start"]
    print(f"\nDONE in {int(el)}s | heuristic free-drops {sum(heur_drops.values())} {heur_drops}")
    print(f"judged {stats['judged']} | kept {stats['kept']} "
          f"({100*stats['kept']/max(1,stats['judged']):.1f}%) | judge-dropped {stats['judged']-stats['kept']} "
          f"| unjudged(excluded, out of budget) {unjudged}")
    print("Per-category kept:")
    for c in sorted(cats):
        n = sum(1 for _ in open(os.path.join(OUT_DIR, f"clean_{c}.jsonl"), encoding="utf-8"))
        print(f"  {c}: {n}")
    print(f"balance start {bal0} -> end {get_balance()}")

if __name__ == "__main__":
    main()
