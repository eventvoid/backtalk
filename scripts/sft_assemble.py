#!/usr/bin/env python3
"""Assemble BackTalk SFT v1 from all chunk files into data/sft_raw.jsonl.

Does ONLY: validation, counting, deduplication, category-stratified selection,
shuffling, QA. Never authors or rewrites content.

The final dataset honors the requested category mix by taking EXACTLY the quota
from each category (stratified), then shuffling globally.
"""
import json, glob, os, random, re, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.environ.get("SFT_CHUNK_DIR", os.path.join(ROOT, "data", "sft_chunks"))
CHUNK_GLOB = os.path.join(CHUNK_DIR, "*.jsonl")
OUT = os.path.join(ROOT, "data", "sft_raw.jsonl")
REPORT = os.path.join(ROOT, "data", "SFT_QA_REPORT.md")

ALLOWED_KEYS = {"user", "assistant"}

QUOTAS = {
    "identity": 2000, "everyday": 4000, "explanations": 4000, "rewrite": 3000,
    "troubleshooting": 2000, "comparisons": 2000, "emotional": 1000,
    "unclear": 1000, "safety": 1000,
}
TARGET = sum(QUOTAS.values())  # 20000

FORBIDDEN = [re.compile(p, re.I | re.M) for p in [
    r"\breverse text\b", r"\breversed text\b", r"\btokeniz", r"\bdataset\b",
    r"\btraining run\b", r"\bfine-?tun", r"\bpipeline\b", r"\bcase note\b",
    r"^focus:", r"^first step\b", r"\bas an ai language model\b",
]]
NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ぀-ヿ一-鿿가-힯]")


def category_of(path):
    n = os.path.basename(path).lower()
    if "identity" in n: return "identity"
    if "troubleshoot" in n: return "troubleshooting"
    if "comparison" in n: return "comparisons"
    if "emotional" in n: return "emotional"
    if "unclear" in n: return "unclear"
    if "safety" in n: return "safety"
    if "rewrite" in n: return "rewrite"
    if "explain" in n or "explanation" in n: return "explanations"
    if "everyday" in n: return "everyday"
    return "everyday"  # fallback (shouldn't happen)


def word_count(s):
    return len(s.split())


def main():
    files = sorted(glob.glob(CHUNK_GLOB))
    raw = 0
    broken = extra = empty = forbidden = non_eng = 0
    parsed = []
    for f in files:
        cat = category_of(f)
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            raw += 1
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                broken += 1
                continue
            if not isinstance(o, dict) or set(o.keys()) != ALLOWED_KEYS:
                extra += 1
                continue
            u, a = o.get("user"), o.get("assistant")
            if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
                empty += 1
                continue
            blob = u + "\n" + a
            if any(rx.search(blob) for rx in FORBIDDEN):
                forbidden += 1
                continue
            if NON_LATIN.search(blob):
                non_eng += 1
                continue
            parsed.append({"user": u, "assistant": a, "cat": cat})

    # Dedup by user prompt (and exact pair), keep first occurrence.
    seen_user, seen_pair, dups = set(), set(), 0
    deduped = []
    for o in parsed:
        uk = re.sub(r"\s+", " ", o["user"].strip().lower())
        pk = (uk, o["assistant"].strip().lower())
        if uk in seen_user or pk in seen_pair:
            dups += 1
            continue
        seen_user.add(uk); seen_pair.add(pk)
        deduped.append(o)

    # Group by category.
    by_cat = {c: [] for c in QUOTAS}
    for o in deduped:
        by_cat.setdefault(o["cat"], []).append(o)

    rng = random.Random(20260627)
    selected = []
    cat_report = {}
    shortfall = {}
    for c, q in QUOTAS.items():
        pool = by_cat.get(c, [])
        rng.shuffle(pool)
        take = pool[:q]
        selected.extend(take)
        cat_report[c] = (len(pool), len(take), q)
        if len(take) < q:
            shortfall[c] = q - len(take)

    rng.shuffle(selected)

    with open(OUT, "w", encoding="utf-8") as fh:
        for o in selected:
            fh.write(json.dumps({"user": o["user"], "assistant": o["assistant"]},
                                ensure_ascii=False) + "\n")

    ulens = [word_count(o["user"]) for o in selected]
    alens = [word_count(o["assistant"]) for o in selected]
    sample = selected[:]
    random.Random(7).shuffle(sample)
    sample = sample[:100]

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# BackTalk SFT v1 — QA Report\n\n")
        fh.write(f"- Chunk files merged: {len(files)}\n")
        fh.write(f"- Total raw lines read: {raw}\n")
        fh.write(f"- Valid after gates: {len(parsed)}\n")
        fh.write(f"- Unique after dedup: {len(deduped)}\n")
        fh.write(f"- Final lines written (data/sft_raw.jsonl): {len(selected)}\n")
        fh.write(f"- Broken JSON count: {broken}\n")
        fh.write(f"- Wrong/extra-field lines: {extra}\n")
        fh.write(f"- Empty-field count: {empty}\n")
        fh.write(f"- Duplicate count (removed): {dups}\n")
        fh.write(f"- Forbidden-phrase count: {forbidden}\n")
        fh.write(f"- Non-English (heuristic) count: {non_eng}\n")
        if ulens:
            fh.write(f"- Average user length (words): {statistics.mean(ulens):.1f}\n")
            fh.write(f"- Average assistant length (words): {statistics.mean(alens):.1f}\n")
            fh.write(f"- Assistant length min/median/max: {min(alens)}/{int(statistics.median(alens))}/{max(alens)}\n")
        fh.write("\n## Category mix (available / selected / quota)\n\n")
        for c, q in QUOTAS.items():
            avail, took, quota = cat_report[c]
            pct = 100.0 * took / max(1, len(selected))
            fh.write(f"- {c}: {avail} / {took} / {quota}  ({pct:.0f}%)\n")
        ok = (broken == 0 and extra == 0 and empty == 0 and forbidden == 0
              and non_eng == 0 and len(selected) == TARGET and not shortfall)
        fh.write("\n## Verdict\n\n")
        if ok:
            fh.write("**READY for CUDA training.** Exactly 20,000 clean, unique, "
                     "English-only examples in the requested category mix.\n")
        else:
            reasons = []
            if len(selected) != TARGET:
                reasons.append(f"total is {len(selected)}, need {TARGET}")
            for c, n in shortfall.items():
                reasons.append(f"{c} short by {n}")
            for nm, v in [("broken", broken), ("extra-field", extra), ("empty", empty),
                          ("forbidden", forbidden), ("non-English", non_eng)]:
                if v:
                    reasons.append(f"{v} {nm}")
            fh.write("**NOT READY** — " + "; ".join(reasons) + ".\n")
        fh.write("\n## 100 Random Samples\n\n")
        for i, o in enumerate(sample, 1):
            fh.write(f"**{i}. [{o['cat']}] user:** {o['user']}\n\n")
            fh.write(f"**assistant:** {o['assistant']}\n\n---\n\n")

    print(f"raw={raw} valid={len(parsed)} unique={len(deduped)} written={len(selected)} "
          f"broken={broken} extra={extra} empty={empty} dups={dups} "
          f"forbidden={forbidden} non_eng={non_eng}")
    print("by category (avail/selected/quota):")
    for c, q in QUOTAS.items():
        print(f"  {c}: {cat_report[c][0]}/{cat_report[c][1]}/{q}")
    if shortfall:
        print("SHORTFALL:", shortfall)
    if ulens:
        print(f"avg_user={statistics.mean(ulens):.1f} avg_asst={statistics.mean(alens):.1f}")


if __name__ == "__main__":
    main()
