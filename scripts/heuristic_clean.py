#!/usr/bin/env python3
"""Free (no-API) heuristic cleaner for BackTalk SFT.

Reads SRC_DIR/*.jsonl, drops near-certain defects (translation, follow-up endings,
mojibake, word-salad, wrong fields, forbidden phrases, non-English), dedups by user
prompt, and writes survivors per category to OUT_DIR/clean_<cat>.jsonl.
"""
import os, json, re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.environ.get("HC_SRC", os.path.join(ROOT, "data", "final_src"))
OUT_DIR = os.environ.get("HC_OUT", os.path.join(ROOT, "data", "final_clean"))

NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ぀-ヿ一-鿿가-힯]")
FORBIDDEN = [re.compile(p, re.I | re.M) for p in [
    r"\breverse text\b", r"\btokeniz", r"\bdataset\b", r"\btraining run\b",
    r"\bfine-?tun", r"\bpipeline\b", r"\bcase note\b", r"^focus:", r"^first step\b",
    r"\bas an ai language model\b",
]]
TRANSLATE_RE = re.compile(
    r"\btranslat|how (do|to) (you |i )?say\b|\bin (spanish|french|german|italian|"
    r"portuguese|japanese|chinese|russian|korean|arabic|latin|dutch|polish)\b|"
    r"what does .* mean in (english|spanish|french|german|italian)", re.I)
STOP = set(("a an the of to in on at for and or but so if is are was were be been being am "
            "i you he she it we they me him her them my your his our their this that these those "
            "with from by as into about up out down over under not no do does did have has had "
            "can could should would will just than then too very more most some any what why how "
            "when where who which because while there here its it's you're i'm").split())

def cat_of(p):
    n = os.path.basename(p).lower()
    for k, c in [("identity","identity"),("troubleshoot","troubleshooting"),
                 ("comparison","comparisons"),("emotional","emotional"),("unclear","unclear"),
                 ("safety","safety"),("rewrite","rewrite"),("explanation","explanations"),
                 ("explain","explanations"),("everyday","everyday")]:
        if k in n:
            return c
    return "everyday"

SELF_LEAK = re.compile(
    r"\bI'?m? (a |an )?(large )?language model\b|\bI am (a |an )?(large )?language model\b|"
    r"\bchatbots? like me\b|\bwe use a (large )?language model\b|\bI use (a )?(large )?language model\b|"
    r"\bI was trained\b|\bmy training\b|\bI(?:'m| am)[^.]{0,40}\btrained\b", re.I)

U_TRANS = re.compile(
    r"\b(spanish|french|italian|german|portuguese|latin|japanese|chinese|russian|korean) (for|version)\b|"
    r"\bthe (english|spanish|french|italian|german|portuguese) (for|of)\b|"
    r"\bhow (do|to) (you |i )?say\b|\btranslat", re.I)
A_TRANS = re.compile(
    r"\bin (spanish|french|italian|german|portuguese|latin)\b|"
    r"\b(spanish|french|italian|german|portuguese|latin) (for|word|phrase|expression|greeting|idiom)\b", re.I)
STRONGF = re.compile(
    r"\b(c'est|je ne sais|qué onda|niño|librería|mañana|sinvergüenza|verstehe nicht|à bientôt|"
    r"grasse matin|dolce vita|te quiero|te amo|buongiorno|guten tag|grazie|gracias|enchanté|"
    r"bonjour|obrigad|hola|ciao|merci|danke|andiamo|viva la vida|buona fortuna|rompere)\b", re.I)
QUOTED_ACCENT = re.compile(r"['\"][^'\"]*[àâáéèêíìóòôúùñçãõ][^'\"]*['\"]", re.I)
# English loanwords / terms that legitimately carry accents — do NOT treat as foreign.
ENG_ACCENT = set("sauté sautéed sautéing sautés café cafés déjà résumé résumés naïve naïveté jalapeño "
    "jalapeños fiancé fiancée cliché clichés purée puréed purées entrée entrées soufflé soufflés piñata "
    "piñatas façade façades crème brûlée exposé touché séance papier-mâché doppelgänger frappé frappés "
    "pâté pâtés flambé flambéed consommé velouté béchamel arête arêtes après crêpe crêpes crêperie "
    "à la flambée protégé protégée vis-à-vis café-au-lait naïvely".split())
LOWER_ACCENT = re.compile(r"\b[a-zàâáéèêíìóòôúùñçãõ'’]*[àâáéèêíìóòôúùñçãõ][a-zàâáéèêíìóòôúùñçãõ'’]*\b")

def foreign(o):
    u, a = o["user"], o["assistant"]
    if U_TRANS.search(u): return True
    if A_TRANS.search(a): return True
    if STRONGF.search(u + " " + a): return True
    if QUOTED_ACCENT.search(u + " " + a): return True
    # a lowercase accented word in the answer that isn't a known English loanword => foreign output
    for w in LOWER_ACCENT.findall(a):
        if w.lower().strip("'’") not in ENG_ACCENT:
            return True
    return False

def bad(o):
    u, a = o["user"], o["assistant"]
    if "�" in u or "�" in a: return "mojibake"
    if SELF_LEAK.search(a): return "self_arch_leak"
    if foreign(o): return "foreign_translation"
    if NON_LATIN.search(u + a): return "non_english"
    if TRANSLATE_RE.search(u): return "translation"
    if a.rstrip().endswith("?"): return "follow_up"
    if any(rx.search(u + "\n" + a) for rx in FORBIDDEN): return "forbidden"
    w = re.findall(r"[A-Za-z']+", a)
    if len(w) >= 25 and sum(1 for x in w if x.lower() in STOP) / len(w) < 0.10:
        return "word_salad"
    if len(a.split()) < 4 or len(a.split()) > 200: return "length"
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()
    kept = {}
    drops = {}
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl"))):
        c = cat_of(f)
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                drops["broken_json"] = drops.get("broken_json", 0) + 1
                continue
            if set(o) != {"user", "assistant"} or not o["user"].strip() or not o["assistant"].strip():
                drops["bad_fields"] = drops.get("bad_fields", 0) + 1
                continue
            r = bad(o)
            if r:
                drops[r] = drops.get(r, 0) + 1
                continue
            k = re.sub(r"\s+", " ", o["user"].strip().lower())
            if k in seen:
                drops["dup"] = drops.get("dup", 0) + 1
                continue
            seen.add(k)
            if c not in out:
                out[c] = open(os.path.join(OUT_DIR, f"clean_{c}.jsonl"), "w", encoding="utf-8")
            out[c].write(json.dumps({"user": o["user"], "assistant": o["assistant"]},
                                    ensure_ascii=False) + "\n")
            kept[c] = kept.get(c, 0) + 1
    for fh in out.values():
        fh.close()
    print("drops:", drops)
    print("kept per category:", kept, "total", sum(kept.values()))

if __name__ == "__main__":
    main()
