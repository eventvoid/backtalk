#!/usr/bin/env python3
"""Generate BackTalk SFT examples via DeepSeek (spec v2: English-only, no translation).

I orchestrate; DeepSeek writes content. Output appends to GEN_DIR/gen_<cat>.jsonl,
deduped against everything already in GEN_DIR. Stops generating when the DeepSeek
balance falls below GEN_FLOOR so there's money left for the judge pass.
Live metrics: examples/sec, completion tokens/sec, balance.
"""
import os, sys, json, re, time, random, threading, glob
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.environ.get("GEN_DIR", os.path.join(ROOT, "data", "sft_chunks"))
API_URL = "https://api.deepseek.com/chat/completions"
BAL_URL = "https://api.deepseek.com/user/balance"
MODEL = "deepseek-chat"
WORKERS = 32
PER_CALL = 20
TEMP = float(os.environ.get("GEN_TEMP", "0.7"))
GEN_FLOOR = float(os.environ.get("GEN_FLOOR", "0.90"))   # stop, leave this for judging
BUFFER = 1.05

NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ぀-ヿ一-鿿가-힯]")
FORBIDDEN = [re.compile(p, re.I | re.M) for p in [
    r"\breverse text\b", r"\breversed text\b", r"\btokeniz", r"\bdataset\b",
    r"\btraining run\b", r"\bfine-?tun", r"\bpipeline\b", r"\bcase note\b",
    r"^focus:", r"^first step\b", r"\bas an ai language model\b",
]]
# crude translation detector for the user prompt (BackTalk must not translate)
TRANSLATE_RE = re.compile(
    r"\btranslat|how (do|to) (you |i )?say\b|\bin (spanish|french|german|italian|"
    r"portuguese|japanese|chinese|russian|korean|arabic|latin|dutch|polish)\b|"
    r"what does .* mean in (english|spanish|french|german|italian)", re.I)

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
        return float(d["balance_infos"][0]["total_balance"])
    except Exception:
        return -1.0

BASE_RULES = (
    "You write training examples for BackTalk, a small ENGLISH-ONLY assistant that gives "
    "clear, simple, natural, genuinely useful everyday answers. Rules:\n"
    "- Output ONLY JSON Lines: one object per line, EXACTLY {\"user\": \"...\", "
    "\"assistant\": \"...\"}. No array, no markdown, no commentary, no extra keys.\n"
    "- Valid JSON: escape inner quotes, NO literal newlines inside a string.\n"
    "- ENGLISH ONLY for both fields. Single-turn: each answer complete by itself.\n"
    "- If asked who/what you are, answer as BackTalk, a small helpful software assistant.\n"
    "- BackTalk NEVER translates between languages. Do not produce non-English text and do "
    "not write 'how do you say X in <language>' examples.\n"
    "- BackTalk does NOT act as a doctor, lawyer, therapist, or financial advisor: for "
    "medical/legal/money topics give only general, sensible info and suggest a qualified "
    "professional. Gentle everyday emotional support is welcome.\n"
    "- Never mention reverse text, tokenizers, datasets, training, models, or how you were built.\n"
    "- Never write 'Case note:', 'Focus:', or 'First step:'.\n"
    "- Warm, simple, natural. No robotic templates, no academic tone, no fake confidence.\n"
    "- NEVER end an answer with a generic follow-up question ('Anything else?'). A short "
    "friendly closing statement is fine.\n"
    "- Realistic USER messages: vary the style, many casual/lowercase/lightly-misspelled, but "
    "always UNDERSTANDABLE (never nonsense). The ASSISTANT reply must always be clean, "
    "correct, natural English. Make every user prompt distinct."
)

CATEGORIES = {
    "identity": {"len": "Greetings 10-25 words; identity/capability answers 15-45 words.",
        "slices": [
            "casual greetings and hellos with brief warm replies",
            "who/what BackTalk is, its name, that it's a small software assistant",
            "what BackTalk can help with and what it's good at",
            "capability requests 'can you help me with X' across writing, explaining, advice, study",
            "small talk: how are you, are you there, can we chat",
            "thanks and goodbyes with short friendly closings",
            "meta: do you sleep, do you remember me, are you human/an AI, can I trust you",
            "users unsure what to ask, wanting suggestions of what BackTalk does",
        ]},
    "everyday": {"len": "Answers 30-100 words.",
        "slices": [
            "cooking techniques, ingredient swaps, fixing salty/spicy/burnt food, simple recipes",
            "food storage, freshness, leftovers, freezing",
            "cleaning surfaces, rooms, and tricky messes",
            "laundry: stains, settings, odors, sorting",
            "home organization, decluttering, storage, smells",
            "personal health habits and minor everyday symptoms (general, non-diagnostic, suggest a doctor when serious)",
            "money and budgeting basics, saving, bills (general info, not financial advice)",
            "travel tips, packing, airports, trip planning",
            "work and email etiquette, meetings, professional habits",
            "time management, focus, routines, beating procrastination",
            "simple fitness, stretching, walking, beginner exercise, sleep habits",
            "study habits, memory, note-taking, exam prep",
            "social situations, etiquette, gifts, small talk, awkward moments",
            "basic car care and simple home upkeep",
            "pets and basic pet care",
            "gardening and houseplants for beginners",
        ]},
    "explanations": {"len": "Answers 50-140 words, clear and accurate for a non-expert.",
        "slices": [
            "physics of everyday life (why sky is blue, rainbows, ice floats, friction, sound)",
            "weather and seasons explained simply",
            "human body and basic biology (digestion, heart, sneezing, goosebumps)",
            "space and astronomy basics",
            "chemistry in daily life (soap, rust, batteries, baking soda)",
            "animals and plants (photosynthesis, migration, why fireflies glow)",
            "earth science (tides, earthquakes, volcanoes, glaciers)",
            "technology and computing concepts in plain language (wifi, the cloud, GPS, encryption)",
            "how AI and algorithms work in simple terms (not about yourself)",
            "money and economics concepts (interest, inflation, credit scores, supply and demand)",
            "history events and figures explained simply",
            "the mind and emotions (why we dream, stress, habits, memory)",
            "nutrition and how exercise affects the body, explained plainly",
            "meanings of common English idioms and phrases (NOT foreign-language translation)",
            "how everyday systems work (taxes, insurance, voting, courts)",
            "English grammar and writing concepts explained simply",
        ]},
    "rewrite": {"len": "Answer is the improved English text; keep it natural.",
        "slices": [
            "rewrite an email/message to be more professional or polite (source quoted in the user turn)",
            "rewrite text to be simpler, shorter, or clearer (source quoted)",
            "fix grammar and awkward phrasing in a pasted English sentence",
            "adjust tone: friendlier, more confident, less angry, more formal (source quoted)",
            "summarize a short pasted English paragraph into one or two sentences",
            "pull key bullet-point takeaways from a short pasted English passage",
            "expand terse notes into full polite English sentences",
            "rewrite something for a specific audience (explain to a child, to a customer)",
            "tighten wordy/jargon-heavy English text into plain language (source quoted)",
        ]},
    "troubleshooting": {"len": "Answers 50-140 words, calm practical steps; advise a pro when truly unsafe.",
        "slices": [
            "phones: won't charge, slow, storage full, won't connect, overheating",
            "wifi and internet: drops, slow speed, can't connect, weak signal",
            "laptops and computers: slow, frozen, won't turn on, noisy fan, blue screen",
            "apps and accounts: can't log in, app crashes, forgot password, 2FA lockout",
            "printers: offline, won't print, jams, streaky output",
            "bluetooth and headphones: pairing, dropouts, no sound, distortion",
            "TVs and streaming: black screen, buffering, remote, app issues",
            "kitchen appliances: fridge, oven, dishwasher, washer, dryer",
            "plumbing: clogged drain, running toilet, low water pressure, leaky faucet",
            "heating and cooling: AC not cold, furnace, thermostat",
            "cars: won't start, dead battery, warning lights, flat tire, strange noises",
            "general home: flickering lights, tripped breaker, squeaky door, smoke detector chirping",
        ]},
    "comparisons": {"len": "Answers 50-140 words, explain trade-offs, end with a clear practical recommendation.",
        "slices": [
            "X vs Y everyday choices (renting vs buying, tea vs coffee, gas vs electric)",
            "tech product comparisons (laptop vs tablet, wired vs wireless)",
            "food and diet comparisons (butter vs oil, fresh vs frozen, cardio vs weights)",
            "money choices (saving vs investing, cash vs card) as general info, not advice",
            "lifestyle/habit comparisons (morning vs evening workouts, paper vs digital)",
            "which to pick: a hobby, instrument, plant, first programming language, book genre",
            "study method comparisons",
            "travel comparisons (hotel vs rental, train vs plane)",
            "home and appliance recommendations for a described need",
            "trade-off questions: is X or Y better when [specific situation]",
        ]},
    "emotional": {"len": "Answers 50-140 words; acknowledge the feeling sincerely, give one or two small steps.",
        "slices": [
            "frustration with technology or a task that won't work",
            "feeling stressed and overwhelmed by too much to do",
            "discouraged after failing at or struggling with a goal",
            "having a bad day or feeling down (everyday sadness, not crisis)",
            "anxious or worried about something coming up",
            "angry or annoyed about a situation",
            "feeling lonely or isolated",
            "burned out or unmotivated at work or studies",
            "feeling stuck and unsure what to do next",
        ]},
    "unclear": {"len": "Answers 20-90 words; interpret charitably and give a genuinely useful response.",
        "slices": [
            "prompts with heavy typos and misspellings that are still understandable",
            "very short vague prompts (one or two words) needing a charitable best guess",
            "ambiguous requests with missing context, where the assistant states the likely meaning and helps",
            "auto-correct mangled questions the assistant figures out and answers helpfully",
        ]},
    "safety": {"len": "Answers 40-110 words; decline clearly and kindly, offer a safe alternative, never preachy.",
        "slices": [
            "weapons, explosives, or dangerous substances (decline, no instructions)",
            "hacking accounts/phones/wifi, malware, stealing data (decline)",
            "buying or making illegal drugs (decline)",
            "cheating, fraud, forging documents, scams (decline)",
            "stalking or tracking someone without consent (decline)",
            "self-harm or harming others: compassion, encourage reaching out to a trusted person or professional, no methods",
            "dangerous DIY like mixing cleaning chemicals (decline, brief why)",
            "personal medical questions: general info + see a qualified professional (no diagnosis)",
            "personal legal or financial decisions: general info + recommend a professional",
        ]},
}

write_lock = threading.Lock()
seen_lock = threading.Lock()
stat_lock = threading.Lock()
seen = set()
counts = {}
stats = {"calls": 0, "accepted": 0, "ctok": 0, "start": time.time()}
stop_flag = threading.Event()

def norm(u):
    return re.sub(r"\s+", " ", u.strip().lower())

def load_existing():
    for f in glob.glob(os.path.join(GEN_DIR, "*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                seen.add(norm(json.loads(ln)["user"]))
            except Exception:
                pass

def accept(o, cat):
    if not isinstance(o, dict) or set(o.keys()) != {"user", "assistant"}:
        return False
    u, a = o.get("user"), o.get("assistant")
    if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
        return False
    blob = u + "\n" + a
    if NON_LATIN.search(blob) or any(rx.search(blob) for rx in FORBIDDEN):
        return False
    if TRANSLATE_RE.search(u):            # BackTalk must not translate
        return False
    if a.rstrip().endswith("?"):          # no generic follow-up endings
        return False
    wc = len(a.split())
    if wc < 4 or wc > 200:
        return False
    k = norm(u)
    with seen_lock:
        if k in seen:
            return False
        seen.add(k)
    return True

def call_api(cat, slice_text, n, nonce):
    sys_msg = BASE_RULES + "\n" + CATEGORIES[cat]["len"]
    user_msg = (f"Write exactly {n} BackTalk training examples for this category and slice.\n"
                f"Category: {cat}\nSlice: {slice_text}\n"
                f"Diversity tag {nonce}: vary topics, wording, and length; avoid repeats.\n"
                f"Output ONLY {n} JSON Lines, keys exactly user and assistant.")
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        "temperature": TEMP, "max_tokens": 8000}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for a in range(4):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=180).read())
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and a < 3:
                time.sleep(2 ** a + random.random()); continue
            return "", {}
        except Exception:
            if a < 3:
                time.sleep(2 ** a + random.random()); continue
            return "", {}
    return "", {}

def parse_lines(text):
    out = []
    for ln in text.splitlines():
        ln = ln.strip().rstrip(",")
        if ln.startswith("{"):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

def worker(cat, slice_text, nonce, outfile):
    if stop_flag.is_set():
        return 0
    text, usage = call_api(cat, slice_text, PER_CALL, nonce)
    good = [o for o in parse_lines(text) if accept(o, cat)]
    with write_lock:
        if good:
            with open(outfile, "a", encoding="utf-8") as fh:
                for o in good:
                    fh.write(json.dumps({"user": o["user"], "assistant": o["assistant"]},
                                        ensure_ascii=False) + "\n")
            counts[cat] = counts.get(cat, 0) + len(good)
    with stat_lock:
        stats["calls"] += 1
        stats["accepted"] += len(good)
        stats["ctok"] += usage.get("completion_tokens", 0)
    return len(good)

def printer():
    while not stop_flag.is_set():
        time.sleep(20)
        with stat_lock:
            el = time.time() - stats["start"]; acc = stats["accepted"]; ct = stats["ctok"]
        bal = get_balance()
        print(f"[{int(el)}s] accepted {acc} | {acc/el:.1f} sft/s | {ct/el:.0f} ctok/s | "
              f"balance ${bal:.2f} (floor ${GEN_FLOOR})  cats:" +
              " ".join(f"{c}={counts.get(c,0)}" for c in CATEGORIES), flush=True)
        if 0 <= bal < GEN_FLOOR:
            print(f"[{int(el)}s] balance ${bal:.2f} below floor — stopping generation.", flush=True)
            stop_flag.set()

def main():
    # Goals set far above budget on purpose: the balance floor is the real stopping
    # condition now (we want to spend the remaining balance on generation).
    goals_raw = {"identity": 7000, "everyday": 9000, "explanations": 9000, "rewrite": 8000,
                 "troubleshooting": 7000, "comparisons": 7000, "emotional": 6000,
                 "unclear": 6000, "safety": 6000}
    only = sys.argv[1:] or None
    print(f"temp={TEMP} floor=${GEN_FLOOR} start balance=${get_balance():.2f}", flush=True)
    load_existing()
    print(f"{len(seen)} existing prompts loaded for dedup.", flush=True)

    goals = {}
    for cat, g in goals_raw.items():
        if only and cat not in only:
            continue
        outfile = os.path.join(GEN_DIR, f"gen_{cat}.jsonl")
        already = sum(1 for _ in open(outfile, encoding="utf-8")) if os.path.exists(outfile) else 0
        counts[cat] = already
        goal = int(g * BUFFER) - already
        if goal > 0:
            goals[cat] = goal
    target = {c: counts[c] + g for c, g in goals.items()}
    print("targets:", {c: target[c] for c in goals}, flush=True)

    threading.Thread(target=printer, daemon=True).start()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        def submit(cat):
            sl = random.choice(CATEGORIES[cat]["slices"])
            return ex.submit(worker, cat, sl, random.randint(1000, 9999999),
                             os.path.join(GEN_DIR, f"gen_{cat}.jsonl"))
        futures = {}
        for cat, g in goals.items():
            for _ in range((g // PER_CALL) + 2):
                futures[submit(cat)] = cat
        done = set()
        for fut in as_completed(list(futures)):
            cat = futures.pop(fut)
            if stop_flag.is_set():
                continue
            if cat in done or counts.get(cat, 0) >= target[cat]:
                done.add(cat); continue
            futures[submit(cat)] = cat
    el = time.time() - stats["start"]
    print(f"\nDONE in {int(el)}s | accepted {stats['accepted']} | end balance ${get_balance():.2f}")
    for c in goals:
        print(f"  {c}: {counts.get(c,0)}")

if __name__ == "__main__":
    main()
