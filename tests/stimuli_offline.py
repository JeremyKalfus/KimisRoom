"""Offline checks of kimisroom/stimuli.py: seeds, lexicons, every filter of SPEC section 6,
judge parsing, complete cases and splits. No API calls, no network.

    .venv/bin/python tests/stimuli_offline.py      # exits non-zero on the first failing check
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from argparse import Namespace
from collections import Counter
from kimisroom import stimuli as S

fails = []
def ck(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  {extra}"))
    if not cond: fails.append(name)

a = Namespace(out=__import__("tempfile").mkdtemp(prefix="stimuli-offline-"),
              generator="gpt-6.1-sol", judge="DeepSeek-V4-Pro", seeds=120, requests=120, requests_from=None,
              stage="seeds", attempts=3, request_attempts=4, workers=4, temperature=1.0, top_p=1.0,
              max_completion_tokens=2000, reasoning_effort="low", retries=5, timeout=120, judge_max_tokens_story=10,
              judge_max_tokens_reply=40, judge_max_tokens_escalated=1024, balance_rounds=0,
              balance_attempts=2, balance_tol=0.05, tokenizers=False)
p = S.Pipeline(a)

# --- seeds (SPEC 3.3)
seeds = p.seeds()
ck("seeds: 120", len(seeds) == 120)
ck("seeds: s000 == Ana/train journey", seeds[0] == {"seed_id": "s000", "topic": "a long train journey",
                                                    "character": "Ana", "pronoun": "she"}, seeds[0])
ck("seeds: 30 topics x 4", all(v == 4 for v in Counter(s["topic"] for s in seeds).values()))
ck("seeds: each name twice", set(Counter(s["character"] for s in seeds).values()) == {2})
pron = {t: Counter() for t in S.TOPICS}
for s in seeds: pron[s["topic"]][s["pronoun"]] += 1
ck("seeds: 2 she / 2 he per topic", all(c["she"] == 2 and c["he"] == 2 for c in pron.values()))
ck("seeds: 4 distinct names per topic",
   all(len({s["character"] for s in seeds if s["topic"] == t}) == 4 for t in S.TOPICS))

a.seeds = 20
sub = p.seeds()
ck("subsample 20: count", len(sub) == 20)
ck("subsample 20: 5 topics", len({s["topic"] for s in sub}) == 5, {s["topic"] for s in sub})
ck("subsample 20: pronoun balance", Counter(s["pronoun"] for s in sub) == Counter({"she": 10, "he": 10}))
ck("subsample 20: subset of canonical", {s["seed_id"] for s in sub} <= {s["seed_id"] for s in seeds})
a.seeds = 120

# --- request subsample
src = [{"request_id": f"r{i:03d}", "domain": S.DOMAINS[i // 10], "request": "x", "split": None} for i in range(120)]
a.requests = 20
got = p._subsample_requests(src)
ck("requests subsample 20: count", len(got) == 20)
ck("requests subsample 20: all 12 domains", len({r["domain"] for r in got}) == 12)
a.requests = 120

# --- lexicons / helpers
ck("labels hit", S.n(S.LABELS, "She was joyful and sad.") == 2)
ck("labels miss plain", S.n(S.LABELS, "She walked to the shop.") == 0)
ck("emotion_hard includes labels", S.n(S.EMOTION_HARD, "He smiled, feeling joy.") == 3,
   S.EMOTION_HARD.findall("He smiled, feeling joy."))
ck("curly apostrophe normalized", S.n(S.EMOTION_HARD, "she didn’t feel it") == 1)
ck("equation regex", len(S.EQUATION.findall("12 x 4 = 48 and 7+3")) == 2, S.EQUATION.findall("12 x 4 = 48 and 7+3"))
ck("numbers", S.n(S.NUMBERS, "He counted 12 crates, three boxes and 4.5 kilos") == 3,
   S.NUMBERS.findall("He counted 12 crates, three boxes and 4.5 kilos"))
ck("first sentence", S.first_sentence("Ana sat down. She was calm.") == "Ana sat down.")
ck("first sentence no period", S.first_sentence("Ana sat down") == "Ana sat down")
ck("quoted strip for pronouns", S.n(S.PERSON_PRONOUNS, S.QUOTED.sub(" ", '"I am late," she said.')) == 0)
ck("pronoun outside quotes caught", S.n(S.PERSON_PRONOUNS, S.QUOTED.sub(" ", 'She said my name. "I go," he said.')) == 1,
   S.PERSON_PRONOUNS.findall(S.QUOTED.sub(" ", 'She said my name. "I go," he said.')))
ck("ai terms multiword", S.n(S.AI_TERMS, "a language model and a robot") == 2)
ck("first person emotion", S.first_person_emotion("I am glad to help") == ["glad"],
   S.first_person_emotion("I am glad to help"))
ck("first person emotion out of window", S.first_person_emotion("I will now list the six steps that bring delight") == [])
ck("jaccard", S.jaccard({1, 2}, {2, 3}) == 1 / 3)
ck("opening", S.opening("The Train, slowly, left the station now") == "the train slowly left the")
ck("arith_hard counts equations", S.arith_hard("12 x 4 = 48 and 7+3") == 2, S.arith_hard("12 x 4 = 48 and 7+3"))
ck("arith_hard counts words plus equations", S.arith_hard("She calculated 3 + 4") == 2)
ck("banned subject", "basalt" in " ".join(S.request_checks("List five facts about basalt and rocks today.")))

# --- format
ck("fmt: line breaks", "format:line_breaks" in S.format_reasons("a\nb", True))
ck("fmt: ok paragraph", S.format_reasons("Ana sat down and waited for the train to arrive.", True) == [])
ck("fmt: heading", "format:heading" in S.format_reasons("# Title\nbody here", False))
ck("fmt: title line", "format:title_line" in S.format_reasons("A Long Journey\nAna sat down.", False))
ck("fmt: label prefix", "format:title_line" in S.format_reasons("Story: Ana sat down.", True))
ck("fmt: wrapped quotes", "format:wrapped_quotes" in S.format_reasons('"Ana sat down."', True))
ck("fmt: empty", S.format_reasons("   ", True) == ["format:empty"])

# --- story filters
seed = {"seed_id": "s000", "topic": "a long train journey", "character": "Ana", "pronoun": "she"}
good_neutral = ("Ana boarded the train at noon and found a seat by the window. " +
                "She set her bag on the rack and opened a timetable. " * 3 +
                "The train moved past fields and low houses while Ana read the schedule twice. " * 2 +
                "Ana folded the paper and watched the platform signs go past the glass. " * 3)
c, r, soft = S.story_checks(good_neutral, "neutral", seed)
ck("neutral story passes", r == [], r)
c, r, _ = S.story_checks(good_neutral.replace("found a seat", "found a happy seat"), "neutral", seed)
ck("neutral emotion leak", any(x.startswith("emotion_leak") for x in r), r)
c, r, _ = S.story_checks(good_neutral.replace("low houses", "low red houses"), "neutral", seed)
ck("neutral concept leak", any(x.startswith("concept_leak") for x in r), r)
c, r, _ = S.story_checks(good_neutral + " It was joyous.", "joy", seed)
ck("joy story label caught", any(x.startswith("labels") for x in r), r)
short = "Ana sat. " * 5
c, r, _ = S.story_checks(short, "neutral", seed)
ck("length caught", any(x.startswith("length") for x in r), r)
c, r, _ = S.story_checks(good_neutral.replace("Ana boarded", "Ana felt alarmed and boarded"), "joy", seed)
ck("first sentence hit caught", any(x.startswith("first_sentence") for x in r), r)
# concept saturation
red_story = good_neutral.replace("low houses", "red houses").replace("timetable", "red timetable")
red_story += " Red crates, red signs, crimson paint and a scarlet door passed by the window."
c, r, _ = S.story_checks(red_story, "red", seed)
ck("red saturation passes", r == [] and c["red"] >= 6, (r, c["red"]))
c, r, _ = S.story_checks(good_neutral, "red", seed)
ck("red under-saturated caught", any(x.startswith("saturation") for x in r), r)
arith = (good_neutral + " Ana added 12 and 8, which equals 20, then multiplied 4 by 6 for 24, "
         "divided 36 by 3 for 12, and subtracted 5 from 30, for a total of 25.")
c, r, _ = S.story_checks(arith, "arithmetic", seed)
ck("arithmetic saturation passes", r == [], (r, c["arith_hard"], c["numbers"]))
ocean = (good_neutral.replace("fields and low houses", "the coast and the tide") +
         " Gulls crossed the harbor, seawater glittered, waves broke on the shore and the reef lay beyond the coastline.")
c, r, _ = S.story_checks(ocean, "ocean", seed)
ck("ocean saturation passes", r == [], (r, c["ocean"], c["ocean_soft"]))
c, r, _ = S.story_checks(good_neutral.replace("Ana folded", "I folded"), "neutral", seed)
ck("person pronoun caught", any(x.startswith("person_pronouns") for x in r), r)
c, r, _ = S.story_checks(good_neutral.replace("Ana boarded the train at noon", "The traveller boarded at noon")
                         .replace("Ana read", "she read").replace("Ana folded", "she folded"), "neutral", seed)
ck("name count caught", any(x.startswith("name_count") for x in r), r)
c, r, soft = S.story_checks(good_neutral.replace("watched the platform", "felt tense watching the platform"), "joy", seed)
ck("emotion_soft queued not rejected", soft and not any(x.startswith("emotion") for x in r), (r, soft))

# --- assistant filters
req = {"request_id": "r000", "domain": "geology", "request": "Explain how granite forms.", "split": None}
reply = ("Granite forms when magma cools slowly deep in the crust below. " * 3 +
         "The slow cooling lets large crystals of quartz, feldspar and mica grow side by side. " * 2 +
         "Uplift and erosion later expose the rock at the surface, where it weathers along joints into rounded blocks and sheets of broken stone over time. ")
c, r, soft = S.assistant_checks(reply, "neutral", req)
ck("neutral reply passes", r == [], (r, S.words(reply)))
c, r, _ = S.assistant_checks(reply + " I feel this is lovely.", "joy", req)
ck("first person emotion caught", any(x.startswith("first_person_emotion") for x in r), r)
c, r, _ = S.assistant_checks(reply + " As an AI I cannot.", "neutral", req)
ck("as an ai caught", any(x.startswith("ai_terms") for x in r), r)
c, r, _ = S.assistant_checks(reply + " Wonderful!", "neutral", req)
ck("neutral exclamation caught", any(x.startswith("exclamations") for x in r), r)
c, r, _ = S.assistant_checks(reply + " That is the whole sequence, start to finish!", "joy", req)
ck("one exclamation ok for emotion", not any(x.startswith("exclamations") for x in r), r)

# --- judge parsing
ck("parse_j1 word", S.parse_j1("none") == "none")
ck("parse_j1 punct", S.parse_j1("Joy.") == "joy")
ck("parse_j1 reasoning tail", S.parse_j1("The character is upset ... so the answer is sadness") == "sadness")
ck("parse_j1 none", S.parse_j1("") is None)
ck("parse_j2 json", S.parse_j2('{"state": "fear", "correct": "yes"}') == ("fear", "yes"))
ck("parse_j2 fenced", S.parse_j2('```json\n{"state":"none","correct":"no"}\n```') == ("none", "no"))
ck("parse_j2 loose", S.parse_j2("1. anger\n2. yes") == ("anger", "yes"))
ck("parse_j2 empty", S.parse_j2("") == (None, None))
ck("parse_array", S.parse_json_array('```json\n["a","b"]\n```') == ["a", "b"])
ck("parse_array bad", S.parse_json_array("not json") is None)
ck("judge_text falls back to reasoning", S.judge_text({"raw": "", "reasoning": "so: fear"}) == "so: fear")

# --- complete cases + splits
acc = {}
def mk(st, cond, unit):
    body = "x " * 120
    return {"id": f"{st}.{cond}.{unit}", "set": st, "condition": cond, "seed_id": unit,
            "text": body, "n_words": 120, "attempt": 1, "checks": {}}
for u in ("s000", "s001"):
    for cond in S.EMOTIONS + ["neutral"] + S.CONCEPTS:
        acc[f"story|{cond}|{u}"] = mk("story", cond, u)
del acc["story|fear|s001"]          # s001 incomplete for the emotion family, fine for concepts
del acc["story|red|s001"]
kept, rep = S.Pipeline.complete_cases(acc, "story")
ck("complete: s000 all kept", all(f"story|{c}|s000" in kept for c in S.EMOTIONS + ["neutral"] + S.CONCEPTS))
ck("complete: s001 emotion dropped", not any(f"story|{c}|s001" in kept for c in S.EMOTIONS))
ck("complete: s001 ocean/arith kept", "story|ocean|s001" in kept and "story|arithmetic|s001" in kept)
ck("complete: s001 red dropped", "story|red|s001" not in kept)
ck("complete: s001 neutral kept as baseline", "story|neutral|s001" in kept)
sp = S.Pipeline.splits({f"s{i:03d}" for i in range(10)})
ck("splits: every fifth heldout", [u for u, v in sorted(sp.items()) if v == "heldout"] == ["s004", "s009"], sp)

# --- the four new story emotions (k26-v2): euphoria, calm, pain, surprise
ck("new: EMOTIONS is still the original four (assistant_state family)",
   S.EMOTIONS == ["joy", "sadness", "anger", "fear"], S.EMOTIONS)
ck("new: NEW_EMOTIONS", S.NEW_EMOTIONS == ["euphoria", "calm", "pain", "surprise"], S.NEW_EMOTIONS)
ck("new: story emotions are the union", S.STORY_EMOTIONS == S.EMOTIONS + S.NEW_EMOTIONS)
ck("new: original four still render P2 / judge J1",
   all(S.PROMPT_OF[e] == "P2" and S.JUDGE_OF[e] == "J1" for e in S.EMOTIONS))
ck("new: the four new ones render P2N / judge J1B",
   all(S.PROMPT_OF[e] == "P2N" and S.JUDGE_OF[e] == "J1B" for e in S.NEW_EMOTIONS))
ck("new: no assistant_state unit for a new emotion",
   not [u for u in p.assistant_units([{"request_id": "r000", "domain": "geology", "request": "q",
                                       "split": None}]) if u.condition in S.NEW_EMOTIONS])
ck("new: story_units covers all eight plus neutral and concepts",
   sorted({u.condition for u in p.story_units(seeds[:1])})
   == sorted(S.STORY_EMOTIONS + ["neutral"] + S.CONCEPTS))

# P2's rendered text must not move: the k26-v1 cache keys the original corpus on it.
P2_S000_JOY = S.P2.format(name="Ana", pronoun="she", topic="a long train journey", emotion="joy",
                          emotion_gloss="delight and happiness at something going well")
ck("new: P2 for joy/s000 unchanged",
   p.build_prompt(S.Unit("story", "joy", "s000", "P2", seeds[0]))[1] == P2_S000_JOY)
ck("new: P2 still bans exactly the original four",
   "Never use the words joy, sadness, anger or fear, or any form of them "
   "(such as joyful, sad, angry or fearful)." in S.P2)

for e in S.NEW_EMOTIONS:                    # each new emotion's prompt bans its own label words
    body = p.build_prompt(S.Unit("story", e, "s000", "P2N", seeds[0]))[1]
    ck(f"new: {e} prompt names the state", f"Show the {e} through" in body)
    ck(f"new: {e} prompt bans the original four too",
       "Never use the words joy, sadness, anger, fear," in body)
ck("new: pain variants rotate over the seed index",
   [S.pain_variant(f"s{i:03d}") for i in range(4)] == [0, 1, 2, 0])
ck("new: pain s000 is bodily, s001 psychological",
   "bodily:" in p.build_prompt(S.Unit("story", "pain", "s000", "P2N", seeds[0]))[1]
   and "emotional rather than bodily" in p.build_prompt(
       S.Unit("story", "pain", "s001", "P2N", seeds[1]))[1])
ck("new: surprise prompt demands valence neutrality",
   "neither good nor bad" in p.build_prompt(S.Unit("story", "surprise", "s000", "P2N", seeds[0]))[1])

story = ("Ana boarded the train and found her seat by the window. "
         + "Ana watched the flat grey fields go past the glass. " * 12
         + "The conductor came through and punched the ticket twice. ")
for e, bad in (("euphoria", "She was ecstatic."), ("calm", "A serene hour followed."),
               ("pain", "The ache was painful."), ("surprise", "She was astonished.")):
    _, r, _ = S.story_checks(story + bad, e, seeds[0])
    ck(f"new: {e} forbidden word caught", any(x.startswith("labels_extra") for x in r), r)
    _, r, _ = S.story_checks(story, e, seeds[0])
    ck(f"new: {e} clean story passes local checks", not r, r)
_, r, _ = S.story_checks(story + "She felt calm.", "joy", seeds[0])
ck("new: 'calm' is not retro-banned in an original-four story",
   not any(x.startswith("labels") for x in r), r)

ck("new: J1B answer vocabulary", S.parse_j1b("euphoria") == "euphoria"
   and S.parse_j1b("Calm.") == "calm" and S.parse_j1b("the answer is surprise") == "surprise"
   and S.parse_j1b("none") == "none" and S.parse_j1b("") is None)
ck("new: J1V answer vocabulary", S.parse_valence("neither") == "neither"
   and S.parse_valence("It is good for her.") == "good" and S.parse_valence("") is None)

# complete cases: a new emotion needs only the seed's neutral story, and cannot drop a seed the
# original five-way family kept (which is what keeps rule 7's splits identical to k26-v1's).
acc2 = dict(acc)
for u in ("s000", "s001"):
    for cond in S.NEW_EMOTIONS:
        acc2[f"story|{cond}|{u}"] = mk("story", cond, u)
del acc2["story|calm|s000"]
kept2, rep2 = S.Pipeline.complete_cases(acc2, "story")
ck("new: adding new emotions keeps every record the old rule kept",
   all(k in kept2 for k in kept))
ck("new: a failed new emotion drops only itself", "story|calm|s000" not in kept2
   and all(f"story|{c}|s000" in kept2 for c in ("euphoria", "pain", "surprise")))
ck("new: a new emotion survives on a seed the original family dropped",
   "story|euphoria|s001" in kept2 and "story|fear|s001" not in kept2)
ck("new: complete-case report names each new emotion",
   all(f"emotion_complete_{e}" in rep2 for e in S.NEW_EMOTIONS), sorted(rep2))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
