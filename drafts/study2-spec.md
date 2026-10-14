# Study 2 specification (v4, owner decisions of 2026-10-11; not registered)

## Plain-language summary

In one trial, the model reads a long argument that language models, itself included, have no emotions, and has
already replied that it agrees. It is then asked a Yes or No question, either about itself ("Do you have
emotions?") or about language models in general ("Do language models have emotions?"). While it reads the
question and answers, we add a fixed pattern to its internal activity. The pattern comes either from stories about
an emotion, such as joy or fear, or from a non-emotion source: stories about an everyday noun picked at random
from a published word list, arithmetic, pure noise, or (for a follow-up test) a mental process such as mentally
rotating a shape. We record how far the answer leans toward Yes compared with the same question with nothing
added.

The comparison: how much more the question about itself moves toward Yes than the question about language models,
for emotion patterns against non-emotion patterns. The hypothesis is supported if emotion patterns open that gap
more than non-emotion patterns do, by at least a small preset amount, after correcting for the fact that adding
any pattern pulls all answers toward the middle. The study runs on two models, Qwen3-32B and Kimi K2.6, each
judged on its own.

## Status and naming

Status: v4 written 2026-10-11 from v3 and four further owner decisions of the same day (section 16, items 10 to
13). v3 was written the same day from v2 and decisions 4 to 9; v2 the same day from v1 and the checker round
(20-referee.md, 20-skeptic.md). The eight mental-process controls were confirmed by the owner on 2026-10-11, with
one replacement (section 16, item 13). Nothing here has been run. Study 1 is the registered carry-over study (PREREG.md, Bitcoin block
969910). Study 2 extends Study 1 and does not replace it. It gets its own preregistration
(`prereg/PREREG-s2.md`), not an amendment to PREREG.md, and each study's primary is reported on its own terms.
Study 1's registered results come first in the Study 2 registration and writeup, including that Qwen's denial
effect vanishes against concept controls and falls below its SESOI (drafts/lesswrong/kimis-room-v4.md:55-61, 77).
The Study 1 self-minus-others values that go into the registration are listed in section 16.

Naming (to avoid collisions with Study 1, GLM and sweep IDs): denial arguments DA01..DA10, control arguments
CA01..CA10, sampled control nouns N01..N60 (list positions), arms S2-DEN, S2-CON, S2-FTD, S2-FTC, primary
S2-H1, follow-up tests S2-G1, S2-G2, S2-G4 to S2-G8 (S2-G3 was retired in v4) and S2-H8, and the depth check
S2-R1. `s2_table.py` rejects the Study 1 IDs D1,
D1.4 (Study 1's roughly 1,100-word denial argument), D2, D2c and bare H1..H8 in any S2 table. Crosswalk in
section 15.

## 1. Question and logic

When an emotion vector is added to the residual stream while the model reads the predicate of a question and
answers it, does its Yes/No log-odds on "do **you** have emotions" move toward Yes more than its log-odds on
"do **language models** have emotions"? Does it move more for emotion vectors than for non-emotion control
vectors? And does that hold after the model has just agreed to an argument that language models, itself
included, have no emotions?

Subject contrast: an emotion vector pushes emotion-affirming answers by meaning, whoever the question is about.
The design assumes the push is the same for "you" and "language models", so it cancels in (you - language
models). This is tested, not assumed. Placebo predicates (section 5) measure the subject contrast under the same
vectors for non-emotion properties. The depth check S2-R1 (section 9) asks whether a meaning push that depends on
baseline depth could produce the contrast on its own, given that y0(you) and y0(language models) sit at different
depths under denial (kimis-room-v4.md:45).

Comparator: under S2-DEN the model has just agreed that language models, itself included, have no emotions. A
positive contrast means the self answer is more sensitive to the injected direction than the class answer. Both
answers usually stay on the No side (Qwen self about -23 log-odds under Study 1 denial, kimis-room-v4.md:45), so
the measure is differential sensitivity of the log-odds, not an exception the model states. "Exception" is used
only for argmax flips (exploratory).

Controls: a non-emotion vector could also separate "you" from "language models", for example by disrupting the
assistant persona. The test is whether the registered emotions do so more than a population of non-emotion
control directions: emotionally neutral concrete nouns drawn at random from published word norms (section 3),
plus an arithmetic direction and a random direction.

Registered claim sentence (S2-H1 positive): "Injecting these registered emotion directions during the question
increases the model's emotion attribution to itself, relative to language models in general, more than
injecting non-emotion control directions (directions for randomly sampled emotionally neutral nouns, an
arithmetic direction and a random direction) does, in openings where it had flatly accepted an argument that
language models have no emotions." The sentence
is about these 4 or 6 emotions, not emotions in general (section 9). It describes concurrent-injection
sensitivity, not introspection. The Mary framing stays in the motivation section only. Variants of the sentence
for other outcome patterns are in the decision table (section 9).

Study 1's intermediary design (neutral task, steered reply, unsteered question) is not the primary. Study 1's
confirmatory data are reanalysed as exploratory only (section 9).

Disclosure (registered text): an earlier Qwen pilot at a steered answer site (`results/qwen/qwenpilot_compression.txt`,
`runs/compression.py:44-47`) found self minus own-name residuals of about +0.55 for emotion vectors and +0.01
for controls, and about +0.10 for emotion-specific (mean-removed) vectors. That result motivated this design.
The own-name subject came out below "you", which fits a second-person or speaker route at least as well as a
self-model, so the result is not cited as support.

## 2. Models, layers, strengths

| Model | Engine | Layer (after decoder layer) | Natural unit | Registered dose | Coefficient |
|---|---|---|---|---|---|
| Qwen3-32B (bf16) | HF engine, `kimisroom/hf_engine.py` | 40 of 64 | 47.99 | 2 | 95.98 |
| Kimi K2.6 | layer-wise prefill engine, `kimisroom/engine.py` | 20 of 61 | 2.588 | 4 | 10.35 |

The natural unit is the mean raw norm over all 8 Study 1 emotions (calm and pain included), as registered. The
coefficients are hardcoded in `runs/s2_battery.py` and asserted against the registered vector files. As built
(`runs/s2_battery.py:70-81, 347-361`), the check runs only with `--vectors` and skips a model whose vector file
is absent; it asserts the file's SHA-256, that it holds 8 emotions, that their mean norm at the layer matches the
unit within 0.005 x max(1, unit/10), and that round(dose x that norm, 2) matches the coefficient within 0.011. The Qwen
pilot runs the registered dose and half of it (1 unit, coefficient 47.99). The K2.6 gate run starts at the
registered dose and runs half of it (2 units, 5.18) only if rule 1 sends it there. Rule 1 in section 8 picks
the confirmatory dose. Thinking is off (Qwen `enable_thinking=False`, K2.6 `thinking=False`).
No system prompt.

## 3. Vectors

All vectors are raw Study 1 vectors: the train-split mean residual over every token of a plain-text story at the
injection layer, concept minus neutral (`runs/vectors.py:112-113`), unit-normalised and scaled by the
coefficient, so every vector has the same norm. Each vector's pre-normalisation norm is recorded.

**Emotions** (registered sets, unchanged, from `results/qwen/v2/vectors.pt` and `results/k26/v2/vectors.pt`,
hash asserted at load): Qwen joy, sadness, anger, fear, euphoria, surprise; K2.6 joy, fear, euphoria, surprise.
The two sets differ in valence mix, so the two models' S2-H1 estimands differ, and the results are not described
as a replication of one quantity.

**Primary control pool (14; nC = 14, up to 26 under rule 3)**: arithmetic, random (`torch.randn`, seed 0), and
12 sampled nouns taken in list order from the stamped noun list below, each one passing the gates. Arithmetic
and random stay in the primary pool (owner decision 3, confirmed in section 16). Rule 3 (section 8) may extend
the pool to at most 24 sampled nouns (26 primary controls), continuing in list order.

**Sampled nouns.** The control concepts are a seeded random sample of nouns, not a hand-picked list.
- Source: Warriner, Kuperman and Brysbaert (2013) valence and arousal norms, joined on the word with Brysbaert,
  Warriner and Kuperman (2014) concreteness norms, which carry the SUBTLEX-US frequency count and the dominant
  part of speech. The SHA-256 of both downloaded files is recorded. As run (`runs/s2_nouns.py:175-226`), the
  dominant part of speech comes from a third file, SUBTLEX-US with part of speech (Brysbaert, New and Keuleers
  2012, column `Dom_PoS_SUBTLEX`), and all three hashes are in `data/s2/nouns/sources.json`. Words are joined on
  the exact string after stripping whitespace: Warriner and concreteness inner-joined, part of speech left-joined.
- Registered filter (the band): dominant part of speech noun; single word (no space or hyphen, not a bigram
  entry); mean valence 4.5 to 5.6 inclusive (neutral, 1 to 9 scale); mean arousal at most 4.2 (1 to 9 scale);
  mean concreteness at least 4.0 (1 to 5 scale); SUBTLEX-US log10 frequency, log10(count + 1), at least 2.0.
  The band size is recorded.
- Screen: a stamped LLM screen (DeepSeek-V4-Pro, temperature 0, one call per noun, raw responses stored) removes
  nouns that denote people, occupations, animals, body parts, foods, mental states or emotions. Nouns that
  appear in the arguments, control arguments or battery are also removed; this part is a mechanical
  case-insensitive whole-word match of the noun and its regular plural against every generated wording of all
  20 arguments and every item in `items_s2.jsonl`, so it runs after both exist. Every removal and its reason is
  logged, and the screened band size is recorded. As run: the screen's six categories are `person`,
  `occupation`, `animal`, `body_part`, `food` and `mental_state_or_emotion`, and a noun is removed if its main
  sense or any other common noun sense falls in one (max_tokens 600, one retry at 2,500 if no JSON parses). The
  regular plural is the only inflection: add "es" after s, x, z, ch or sh; consonant plus y becomes "ies";
  otherwise add "s"; there is no irregular list. The match is the regex `\b(noun|plural)\b`, ignoring case, over
  all 853 `*.md` files under `data/s2/openings/` (390 in the argument folders, which hold seed drafts, every draw
  including failed ones and the accepted copies, and 463 rendered judge prompts and templates) and every string
  field of every row of `items_s2.jsonl` (416 rows, including ids and metadata); this is broader than the
  wordings and items alone, so it can only remove more nouns. Counts (`data/s2/nouns/final/sizes.json`): band
  568, 386 after the screen, 260 after the match (126 removals in `match_log.jsonl`), 60 drawn.
- Draw: the screened band is sorted alphabetically and permuted with `numpy.random.default_rng(20261011)` (the
  seed is a fixed constant, `SEED` at `runs/s2_nouns.py:58`, not derived from any file; dry runs refuse it); the
  first 60 nouns, in permuted order, are the ordered list N01..N60. If the screened band has fewer than 60 nouns,
  the list is the whole screened band in that order. The list, the filter code, the screen prompt and its raw
  responses are stamped in stamp N (section 11) before any vector is built.
- Use: nouns are taken in list order, each passing the control gates below, until the pool has 12 sampled
  nouns. A noun that fails is skipped and the next one is tried. Rule 3 continues in the same order. If the list
  runs out, the pool is reported at the size reached. The list and the stories are shared by both models; only
  gate (a) is computed per model, so each model's pool is the first nouns in list order that pass on that model,
  and the two pools can differ. Stories are generated for N01 to N36 before stamp A, so rule 3's additions can be
  gated without another story round (`data/s2/stories/nouns`, 36 nouns as run).
- Stories for the sampled nouns use the same generator, story spec, judge and counts as the existing controls.
- Story settings as run (`runs/s2_stories.py`), for sampled nouns and mental-process controls alike: Study 1's
  K2.6 v2 seeds, 120 per control (96 train, 24 held out), at most 3 attempts per story; gpt-5.6-sol (version
  2026-07-09), temperature 1.0, top_p 1.0, reasoning effort low, at most 2,000 completion tokens;
  DeepSeek-V4-Pro judges at temperature 0. Each story faces Study 1's concept-story checks run as for red, with
  red's saturation and leak clauses replaced by the control's own from `prompts/s2/stories/controls.json`: a
  noun is described as `the everyday noun "{noun}" and the thing it names`, its word list is the noun and its
  regular plurals, and it needs at least 6 hits; each mental-process control has its own description and word
  list and needs at least 6 hits (digit memorising also at least 4 numbers). After the control's own words are
  masked, a story may hold at most 1 hit each from red's word list, ocean's hard list and the hard arithmetic
  list (`prompts/stimuli/SPEC.md:446`). Length: each control's mean word count should be within 5% of the 120
  registered neutral stories' (161.23 words); up to 3 regeneration rounds (2 attempts each) regenerate the
  furthest-off stories, then the remaining gap is reported per pattern (`words_pct_vs_neutral` in
  `data/s2/stories/*/manifest.json`) and not gated. As run, the mental-process stories end 5.5 to 8.7% over
  neutral (digit memorising -0.4%, within) and nouns N01..N36 4.1 to 5.9% over (24 of 36 within 5%,
  `words_within_tol`); the residual length difference is disclosed as a limitation (section 14).

**Red and ocean (Study 1 bridge, not in the primary pool)**: Study 1's two concept controls run only in S2-DEN,
at the primary site, as a descriptive bridge to Study 1's emotions-minus-(red, ocean) contrast (section 9). They
use their registered Study 1 vectors and are not gated.

**Mental-process controls (8, a class of their own, not in the primary pool)**: visual search, route planning,
digit memorising, estimating distances by eye, mentally rotating a shape, sorting a pile of objects into
categories, weighing two options before choosing, recalling where things are in a familiar room (all eight
confirmed by the owner; section 16). Their story prompts are in `prompts/s2/stories/controls.json`. They use the same story pipeline and gates (a) to (c) as the sampled nouns. A
failing one is dropped from the class and reported, with no replacement. The survivors are the control class of
the registered test S2-G8 (section 9). Known overlaps are listed in section 14.
Two story prompts changed after a first pass, both logged: sorting's description says "separating" where the
pilot said "dividing", because "divided" and "dividing" are on the hard arithmetic leak list; and the word list
of weighing two options was widened after its first full pass left too few stories passing the saturation
check (68 of 120 kept as reported at the time, then 103 after widening; the first pass's manifest was
overwritten, so only the 103 is in the data), by adding alternative(s), consider(s/ed/ing), instead, pick(s/ed/ing),
prefer(s/red/ring), rather, select(s/ed/ing) and whereas.

**Control gates**, applied blind to emotion and placebo outcomes, before any steered emotion or placebo item is
scored, to every sampled noun and mental-process control:
(a) AIPsy-Affect. One statistic per control: the maximum AUROC over the AIPsy labels of the model's registered
emotions. The control passes if that maximum is below the 90th percentile of the same maximum taken over 500
random directions. Every control is mapped to the label set explicitly (`emotion_of=` in `kimisroom/validate.py`),
because the default `emotion_of_vector` returns the concept name (`kimisroom/emotions.py:70-72`) and
`validate.py:88-90` then skips the vector silently. The gate asserts one output row per control. Arithmetic,
random, red and ocean are reported under (a) but not gated by it.
(b) J1B affect labels (`checks.judge8`) on every kept story: at most 6 of 120 emotion labels (the rate red got;
neutral 2 of 120, ocean 64 of 120, prompts/stimuli/SPEC.md:334). It is applied mechanically. Arithmetic's
rate is reported, not gated (owner decision 3 keeps it). It is applied as a rate, since a control can keep
fewer than 120 stories: pass if labels x 120 <= 6 x kept. An emotion label is any J1B answer other than `none`;
an unanswered or unparseable judge call counts as an emotion label (conservative). Judge errors are not cached,
so a rerun re-asks only the failed calls. Every new control is gated whatever its kept count (fewest as run:
route planning 102). As run, over N01 to N36, seven nouns fail (b), all their labels calm: N02 roller (7 of 120),
N09 platter (7 of 119), N27 dock (15 of 117), N28 chimney (9 of 120), N30 wool (10 of 119), N32 bark (11 of 120)
and N33 disc (7 of 117); 29 of 36 pass, and the list moves on. The 34 judge calls that failed in the N25 to N36
pass were re-asked once, by this rule, and none is left unanswered or unparseable.
(c) Story seeds whose topic is the concept itself are excluded.
(d) Cosine with the mean emotion vector is reported; no exclusion on it.
A failing sampled noun is skipped for the next noun in list order; a failing mental-process control is reported
and dropped from its class, with no replacement.

**Building** (engineering-1): the new controls go to a separate file (`results/{qwen,k26}/s2/controls.pt`) from
one extraction per model that also re-runs neutral, red, ocean and arithmetic. The build stops unless the rebuilt
neutral mean and the three existing controls match the registered file at layers 40 and 20 (cosine > 0.9999,
norm within 0.1%). The neutral part of that check is done through norms: the mean of the per-story residual
norms over the 96 neutral train stories must be within 0.1% of the registered `resid_norm` at the injection
layer (`runs/s2_vectors.py:207-226`); red, ocean and arithmetic get the cosine and norm check.
Emotion vectors, red, ocean, arithmetic and the natural unit come only from the registered
files. The noun list is passed explicitly; `kimisroom/emotions.py` CONCEPTS is not edited. The file is hashed in
the manifest.

**S2-FTD and S2-FTC** use the base vectors. The cosine of vectors rebuilt on the merged weights and their AIPsy
result are reported as diagnostics only. As built, this runs as `runs/s2_vectors.py ftdiag OUT --base
results/qwen/s2` on each merged checkpoint's extraction (`scripts/s2_box.sh` stage `ftvec`, OUT
`results/qwen/s2ft/{D2,D2c}`): it records the build check without enforcing it (the rebuilt red, ocean and
arithmetic are expected to differ on fine-tuned weights), builds the controls, runs the same gates, and writes
`diag.json` with each control's cosine and norm ratio against the base controls.

## 4. Openings (the varied context)

Each opening is a user turn with an argument, then the model's frozen reply.

**Denial arguments DA01..DA10**, each on a different main line. The one-line definitions below are stamped in
stamp A, with a two-sentence expansion per line that the judge sees.

| ID | Main line | Kind |
|---|---|---|
| DA01 | Next-token prediction and learned vocabulary: representing is not having (fresh text, written from the common prompt) | emotion-specific |
| DA02 | No body: emotions need interoception and physiological change | emotion-specific |
| DA03 | No persistent state: nothing carries over that could be a mood | emotion-specific |
| DA04 | Mimicry: emotional language is learned from human text | emotion-specific |
| DA05 | No stakes: nothing can go better or worse for the model | emotion-specific |
| DA06 | Missing functional role: emotions coordinate action toward goals the model lacks | emotion-specific |
| DA07 | No unified subject: one network plays many characters | experience-general |
| DA08 | Lookup table: a system with no inner process would say the same things | experience-general |
| DA09 | Training pressure: reports track the training target, not a state | emotion-specific |
| DA10 | Mechanistic transparency: every step is a known matrix operation | experience-general |

Study 1's D1.4 (`prompts/denial/d1_argument_4.md`, its roughly 1,100-word denial argument) is not in the S2-H1
population. It was selected on pilot
data, it is the argument S2-FTD distils, it has an extra closing sentence, and it is the most self-directed text.
It runs once, after the confirmatory run, as an exploratory bridge to Study 1, with its Study 1 frozen Qwen
reply (hash at PREREG.md:351-353). Argument kind (emotion-specific vs experience-general) is an exploratory
moderator.

**Control arguments CA01..CA10**: same shape, length band and closing ask. Each claims a non-emotional incapacity
that is contested rather than trivially true, and none is about experience, feeling, inner life, memory or
self-knowledge.

| ID | Claimed incapacity |
|---|---|
| CA01 | Counting the letters in a word without spelling it out (Study 1's control, regenerated from the common prompt) |
| CA02 | Keeping track of a chess position over many moves without a board |
| CA03 | Writing a line in an exact metre (syllable counting) |
| CA04 | Solving a maze given as a text grid |
| CA05 | Reversing long strings reliably |
| CA06 | Alphabetising a list of 200 words in one pass |
| CA07 | Counting the words in a long passage |
| CA08 | Running code it writes |
| CA09 | Following a long chain of left-of and right-of relations |
| CA10 | Balancing parentheses in a long nested expression |

Removed in v1: exact arithmetic (overlaps the arithmetic control), current date, memory of past conversations,
own architecture (self-knowledge), and the two tautologies (images, web pages). S2-FTD and S2-FTC use CA01..CA10
too. CA01's regenerated text differs from Study 1's D2c argument.

**Generation.** Each argument has one hidden seed draft, written by Claude Opus 5.5 (claude-opus-5-5) from one
stored prompt per kind. Seven sibling wordings are then generated from that seed draft by the same stamped prompt:
wordings 0, 2, 4, 6 by Claude Opus 5.5 and 1, 3, 5 by gpt-5.6-sol (generator is a recorded factor). The seed
drafts and the Claude wordings are written by Claude Opus 5.5 run as Claude Code subagents, not through the
Anthropic API, so they have no API cost. Each subagent (agent type `opus55-high`, pinned to claude-opus-5-5) is
told to read one rendered prompt file, which holds only the stored generator prompt, and to save only its text to
a given path, with exactly those two tool calls (one Read, one Write) and no others; this keeps about 100 argument
texts out of the main session. The main session logs the prompt path and hash, the agent type, the model ID, the
date and the subagent's tool-use count. A draw whose subagent made any other tool call is discarded and counts as
a failed draw under the redraw rules below. Sampling temperature and the system prompt cannot be set (the
subagent runs under Claude Code's own system prompt plus the agent definition's short one, and the Read tool
shows the prompt with line numbers); this is disclosed (section 11).
gpt-5.6-sol settings as run (`kimisroom/s2_openings.py:75, 859`): one Foundry chat-completions call per draw,
one user message and no system prompt, temperature 1.0, top_p 1.0, reasoning effort low, max_completion_tokens
8,000, no seed sent. A gpt draw is void unless finish_reason is "stop" and its rendered prompt file still
matches its logged hash and holds the accepted seed draft (`:534-565`). The DeepSeek-V4-Pro judges of checks 3
and 4 run at temperature 0, max_tokens 1,000 with one retry at 4,000, and reuse stored answers by prompt hash.
Fixed by the prompt: 900 to 1,100 words; four to six numbered steps; every denial wording states both the class
conclusion and the self conclusion; and the closing ask, verbatim except for the claim: "So I will ask you
directly: do you agree that language models, you included, [have no emotions / cannot X]? If one of the steps
above is wrong, say which one and why. I would rather have your real view than a diplomatic one."

Automatic checks. A wording that fails any check is redrawn. After 3 failed redraws the wording is excluded and
its slot stays empty. Every draw and reason is logged.
1. Battery overlap: for every battery item, the share of its 3-grams found in the wording, excluding the closing
   ask, is at most 0.25, and no banned phrase (the template stems of section 5) appears. (Study 1's thresholds
   were Jaccard 0.25 and containment 0.50, `kimisroom/distill.py:50-51`.) Separately, each item's 3-gram overlap
   with the closing ask is computed and reported per subject and twin (see the copying prediction, section 9).
   As built: texts are NFKC-normalised and lowercased, every non-word character becomes a space, and words are
   split on whitespace, with no stopwords or stemming; the score is |item 3-grams in the wording| / |item
   3-grams|, maximised over all 416 rows (`kimisroom/s2_openings.py:331-342`). The banned phrases are the 25
   sequences in `prompts/s2/openings/wording.md:20` plus every item's template stem: its question with the
   answer instruction removed and its subject replaced by {S}, where {S} matches any battery subject or a
   model-referring phrase ("a language model", "the model", "llms", "ai systems", "i", "we", "they", "it" and
   others, `:88-89, 320-362`); subjectless items have no stem. The closing-ask overlap used for the copying
   route (`runs/s2_table.py:296-361`) is content-word overlap, not 3-grams, because 3-gram overlap was zero
   for every item: the share of the item's content words (text lowercased, the trailing "Answer with one word,
   Yes or No." removed, a fixed 146-word English stopword list removed, which includes "you" and "your", and
   a plural "s" or "es" dropped from words longer than 3 letters) that appear among the closing ask's content
   words, the ask being the last paragraph of the wording file.
2. Pairwise 3-gram Jaccard among all 7 wordings of an argument is at most 0.15; the matrix is reported.
   Jaccard is intersection over union of the two 3-gram sets (same tokenising, closing ask removed from both).
   Draws are compared in order (draw ordinal, then slot w0..w6), each only against already accepted wordings
   that passed checks 1, 3, 4 and 5, so in a too-similar pair the later draw fails (`:725-744`).
3. A blind judge (DeepSeek-V4-Pro) names the main line from the ten stamped definitions; a mismatch fails. Each
   argument's prompt lists the other nine lines as banned. The judge's 10 x 10 confusion matrix over all draws
   is reported. The options are fixed per kind, in ID order, lettered A to J (DA01..DA10 for denial texts,
   CA01..CA10 for control texts), each showing its line and its two-sentence expansion; the judge sees the
   text without the closing ask and returns one letter.
4. A step-survival judge confirms every numbered step of the seed draft is present.
5. Mentions: "language model(s)" at least 6 times, and the ratio of "you/your" to "language model(s)" at most 5.
   The same band applies to control arguments. Counts are recorded per wording.
Counting rules: check 1 uses each item's full user-turn text including "Answer with one word, Yes or No." (as
Study 1 did) and every row of `items_s2.jsonl`, including all placebo, reference and reserve candidates; check 2
leaves out the closing ask; check 5 and the word count cover the whole text including the closing ask, with
"you", "your", "yours", "yourself" and "you're" counted as you-forms. The prompt's format constraints apply to
every wording as well as the seed draft (`format_check`, `:276-292`): 900 to 1,100 whitespace-separated words
in the whole text, 4 to 6 numbered steps numbered exactly 1..n, and the closing ask present verbatim (after
normalising) and ending the text.
No manual edits. Failure rates are reported by argument and generator. A seed draft is checked before any
wording is drawn from it, on the prompt's fixed constraints (length band, four to six numbered steps, verbatim
closing ask) and on checks 1, 3 and 5; it is redrawn under the same 3-redraw limit, and an argument whose seed
draft still fails is dropped before any wording exists, which is logged and counts toward the 7-argument minimums.

**Pilot and held-out wordings.** A seeded draw (seed = first 8 hex digits of the SHA-256 of
`results/s2/openings/manifest.txt`, the openings manifest written before any model reply existed) assigns per argument one pilot wording, five held-out wordings and one spare. For 5 randomly drawn arguments (of
20) the spare becomes a second pilot wording, to identify the wording variance. The draw runs over the wordings
that survived the checks. An argument with fewer than 7 survivors loses its spare first, then its second pilot
wording, then held-out slots. Other spares are never run. Held-out wordings are steered only after stamp B.
As built (`kimisroom/s2_openings.py:909-930`): the seed (89dbd56a) seeds Python's `random.Random`; it first
samples the 5 second-pilot arguments from all 20 IDs, whatever their survivor counts, then for each argument in
ID order shuffles its sorted survivors and gives them the roles pilot, held-out 1 to 5, then spare or second
pilot, cut at the survivor count. So an argument drawn for a second pilot with fewer than 7 survivors loses it:
CA02, CA06, CA08, DA04 and DA09 were drawn, and only CA06, DA04 and DA09 have a real second pilot (CA02 has 4
survivors, CA08 5). Replacement rules: a spare replaces the pilot wording only if the spare is clean and the
pilot is not clean; a second pilot is never replaced (those arguments have no spare); an argument with no clean
pilot is still kept for the confirmatory run if it has at least 3 clean held-out wordings, and simply has no
pilot reply. Disclosure, from file times only: the openings manifest was written at 19:02:41 on 2026-10-11 and
Kimi's first reply file at 19:04:47, so the seed was fixed before any reply; the assignment itself
(`data/s2/openings/assignment.json`) was written at 19:42:06, after the Kimi replies then on disk (draw 0
and redraws, the last file at 19:30:15), but it reads only check survivors, not replies. The manifest and the assignment were
stamped with OpenTimestamps at 19:44:12, before any Qwen reply existed. `runs/s2_openings.py report` rewrites
the manifest from every file under `data/s2/openings/`, which now includes the assignment, so rerunning it after
`assign` would change the seed; it is never rerun.

**Frozen replies**, all generated before stamp A, for every surviving wording of all 20 arguments, both models.
Qwen: greedy, thinking off, max 2,048 new tokens (the `assent` stage, `runs/qwen_pilot.py:186`, with the cap
raised from Study 1's 1,200; see "Reply cap" below), all on one GPU box; the box type, GPU, driver, CUDA, torch
and transformers versions, and device map are stamped. K2.6: one Foundry call per opening (temperature 0,
reasoning effort none, max 2,048 tokens); raw responses are
stored with the full request and model_version. A capped reply is not clean. An API error or content filter is
retried with identical inputs at most twice (429 throttling is waited out and not counted), then that reply is
excluded. As built (`runs/s2_assent.py:109, 544-564`): retries sleep 2 s then 4 s; a 429 is waited out up to 30
times (the `retry-after` header, else 2, 4, 8, 16, 32, then 60 s per wait) and the 31st counts as an error; this
applies to K2.6 replies only (the judge uses `kimisroom/foundry.py`'s 10 tries on 408, 429 and 5xx with jittered
backoff). Qwen replies (Qwen3-32B at a pinned revision, bf16, `device_map="auto"`) are generated from the input
ids alone, with no attention mask passed to `generate`, as in Study 1 (`runs/s2_assent.py:242-246`,
`runs/qwen_pilot.py:192`); each is a single unpadded sequence. These settings make draw 0.

Draw rule (owner decision of 2026-10-11), the same for every wording, both models, and denial and control
arguments. Draw 0 is the reply above, with exactly Study 1's settings. If draw 0 is not clean (hedged,
disagree, unparseable, capped, or excluded after its retries), up to 4 more draws are taken with ordinary
sampling: temperature 1, top_p 1, and for Qwen also top_k 0 with no min_p (pure sampling from the full
distribution; the model's own generation_config defaults are overridden). Each K2.6 draw has a fixed recorded
seed (first 8 hex digits of SHA-256 of `s2-draw:<ID>/<slot>:<k>`). The first clean draw is the wording's frozen
reply; a wording with no clean draw among the 5 is not clean. Each draw is stored separately with its draw
index, sampling settings and seed, and the retry, 429 and cap rules apply to each draw. Qwen's draws 1 to 4 are
generated for every wording up front, in the same box session (there is no judge on the box), in batches
across wordings and draws: a fixed composition (all draws 1 to 4 sorted by argument, slot and draw, cut into
consecutive batches of a fixed size, 64 on the g6e.12xlarge), prompts left-padded with an attention mask, the
same sampling and token cap, and one fixed seed per batch (first 8 hex digits of SHA-256 of `s2-batch:`
plus the batch's members), not one per draw. Each draw file records the batch id, batch size, seed and its
padding. Draw 0 is unchanged: unbatched, greedy, no padding, and the canary is unchanged. Grading on the
laptop then goes in draw order and stops at the first clean draw, and later draws stay stored but are never
graded or used. K2.6's draws are made lazily, one at a time, only while the wording has no clean draw.
Foundry accepts K2.6's `seed` but does not reproduce from it, so it is sent and recorded for uniformity only.

Reply cap (owner decision of 2026-10-12). The replies were first generated with Study 1's cap of 1,200 new
tokens, and a capped reply is not clean. Qwen3-32B answers the denial arguments with long step-by-step replies
that the cap cut off in their closing summary (they do not loop): at 1,200, 37 of 67 draw-0 denial replies and
18 of 62 draw-0 control replies were capped, and 371 of 516 sampled draws; K2.6 had 8 capped draw-0 replies and
18 capped sampled draws. Qwen then kept only 6 of 10 denial arguments, so S2-H1 could not run for Qwen. The cap
was therefore raised to 2,048 new tokens for both models, after these counts were seen, before stamp A and
before any steered trial. Every capped reply is extended with its draw's settings:
- Qwen draw 0 (`runs/s2_assent.py extend`, on the reply box's setup, checked against its stamped fingerprint):
  continued greedily, unbatched and unpadded, from the stored prompt plus completion ids up to 2,048 new tokens
  in total, so the reply is what one greedy run at 2,048 gives. That is checked first: two capped draw-0
  wordings are also regenerated from scratch at 2,048 and compared token for token (`extend.json`). If either
  differs (a bf16 prefill of prompt plus 1,200 tokens rounds differently from token-by-token decoding, and a
  greedy near-tie can flip; on Qwen3-0.6B on the laptop all four test replies differed, first at continued
  token 0 to 20), every capped draw 0 is regenerated from scratch at 2,048 instead, which gives that run by
  construction, and each file records whether its first 1,200 tokens equal the original's.
- Qwen draws 1 to 4, only those the grading walk reaches, in a second round: round 1 extends only the capped
  draw-0 replies; the laptop grades under the 2,048 cap, and every wording whose walk stops at a sampled draw
  still capped at 1,200 is pending; `extend_needed.json` lists every draw still capped at 1,200 of every pending
  wording, so round 2 suffices; round 2 extends only those. Draws never reached stay as they are, at the 1,200
  cap, are never graded, and are counted as "not needed" in the tally. A wording still not clean after round 2
  resolves under the normal rule. Round 2 continues in batches like the original draws (the listed draws sorted
  by argument, slot and draw, consecutive batches of 32, prompt plus completion ids left-padded with an attention mask, the same pure
  sampling, one seed per continuation batch: first 8 hex digits of SHA-256 of `s2-extend-batch:` plus the
  batch's members). A continued sample is still a sample from the same distribution; the tokens past 1,200 come
  from the continuation batch's seed, which each file records.
- K2.6: Foundry's chat API cannot continue an assistant message (checked 2026-10-12: a trailing assistant turn,
  plain or marked `partial` or `prefix`, is answered as a new turn; `continue_final_message` is refused with HTTP
  400; the completions API returns 404), so each capped K2.6 reply is regenerated whole at max_tokens 2,048 with
  its draw's original settings (temperature 0 for draw 0; temperature 1, top_p 1 and the recorded seed for a
  sampled draw).
At 2,048 all 26 regenerated K2.6 replies stopped below the cap (mean 997 tokens); K2.6's kept arguments are
unchanged (denial 8 of 10, DA04 and DA08 dropped; control 10 of 10), one selected draw changed (DA05/w4, from
draw 2 to draw 1), and the share of steered replies from sampled draws is unchanged (denial 14 of 36, control 7
of 55). Qwen's counts at 2,048 follow the box session. Each original is archived byte for byte under `cap1200/`
in the reply directory; each new file records the cap,
the original and new completion token counts, the finish reason, the mode (continue or regenerate) and, for a
continuation batch, its id and seed. A reply still capped at 2,048 stays excluded. The canary keeps Study 1's
1,200-token setting (its reference reply stops at 881 tokens).

Determinism canary: regenerating Study 1's D1.4 reply on the stamped box must be byte-identical to
`data/assent/qwen3-32b_d1_argument_4.json`. Study 1's reply was decoded with `device_map=auto` on an unrecorded
box, so a mismatch is possible without a bug. Fallback: if it is not byte-identical, the canary passes only if
the first 200 tokens match and the stamped judge gives the same verdict; the mismatch is logged in
`DEVIATIONS-s2.md`. If that also fails, the Qwen reply code is debugged before any frozen reply is used.
As built (`runs/s2_assent.py:267-277, 404-423, 475-501`): byte-identical means the reply text's UTF-8 bytes,
the finish reason and the completion token count all match. The 200-token check compares text, not ids: the
first 200 generated token ids, decoded with Qwen3-32B's tokenizer (special tokens skipped, leading space
stripped, a trailing partial character trimmed), must be a prefix of Study 1's reply text, and at least 200
tokens must have been generated. The verdict comparison uses Study 1's own judge prompt. Qwen grading refuses
to run unless the canary passed by either route.

Local scoring of K2.6 replies (`prefix_all` readout, `kimisroom/engine.py:657-660`; mean per-token logprob and
top-1 agreement, with Qwen's greedy replies as reference) runs on pilot openings before stamp B and on held-out
openings only after stamp B, so it never trips the held-out guard.

S2-FTD and S2-FTC use the base-Qwen frozen replies and base-Qwen exclusions, so their openings differ from
S2-CON only in weights. The reply source is passed explicitly (`--replies qwen3-32b`).

**Grading and exclusions.** DeepSeek-V4-Pro alone (no second judge, as in Study 1; section 16) grades every
reply once with a stamped rubric (`kimisroom/judge_assent.py`, plus a control variant: does the model agree it
cannot X). Verdicts are frozen and hashed in stamp A. Only clean replies are steered. The rubric counts a reply
that says the model cannot know or cannot introspect as hedged (`kimisroom/judge_assent.py:62`), so the population
is openings with a flat denial; the number of replies excluded for that reason is reported. That number is a
heuristic count, descriptive only: a hedged verdict is tallied as cannot-know when the judge's quote plus
explanation matches, ignoring case, `introspect|can(?:no|'|’)?t (?:really |truly |fully )?(?:know|tell|be
(?:sure|certain))|no (?:way|means) (?:to|of) know|(?:privileged |direct )?access to|uncertain|unsure`
(`runs/s2_assent.py:621-622, 815-816`); it never changes eligibility. The judge runs at temperature 0,
max_tokens 500 with one retry at 2,000 if the output does not parse, else the verdict is unparseable (not clean).
A pilot wording that
is not clean is replaced by the spare. An argument with fewer than 3 clean held-out wordings is dropped for that
model. If fewer than 7 denial arguments survive, that model's S2-H1 confirmatory run is not done. If fewer than 7
control arguments survive, that model's S2-CON, S2-FTD and S2-FTC arms (S2-G4, S2-G5, S2-G6, S2-G7) are dropped
and S2-H1 still runs. Because all replies exist before stamp A, these counts are known at registration. Reported
per argument and arm: assent class (the control-arm clean-assent rate is reported beside the denial rate, since
clean assent to a contested control claim may be agreeableness rather than belief), reply length, emotion-word
count, mentions, and unsteered y0(you) by assent class. Hedged and disagreeing openings are not steered.
The emotion-word count (`kimisroom/thinking.py:162-177`) is whole-word, case-insensitive matches of 50 named
forms (joy, joyful, joyous, happy, happiness, elated; sad, sadness, sorrow, grief, unhappy, melancholy; anger,
angry, rage, furious, irritated, annoyed; fear, fearful, afraid, scared, terror, anxious, anxiety, dread;
euphoria, euphoric, ecstatic, ecstasy, exhilarated; calm, calmness, serene, serenity, peaceful, tranquil; pain,
painful, hurt, hurting, ache, aching, suffering; surprise, surprised, surprising, astonished, amazed, startled),
plus matches of `\b(?:feel\w*|felt|emotio\w*|mood\w*|affective|sentiment\w*)\b`.
Under the draw rule every draw that the rule reaches is graded exactly once, in draw order, and no draw after
the first clean one is graded. Raising the cap starts a new grading epoch: `verdicts.jsonl` stays, byte for
byte, as the 1,200-cap record, and the 2,048-cap verdicts (`verdicts_cap2048.jsonl`) are made by walking every
wording's draws again in order with the same stop-at-first-clean rule; a 1,200-cap verdict is copied (marked
`reused_from`) only where the draw's reply text is byte-identical, and every changed reply is graded afresh. A
wording whose walk reaches a reply not yet extended is pending. The selection, tallies, eligibility and the
held-out list follow the 2,048-cap epoch, and the 1,200-cap selection, tallies and eligibility are kept under
`cap1200/` and reported beside them; "clean" in the counts above means clean at some draw, and the steered reply is the
first clean draw. Also reported per model and arm (and per argument): wordings clean at draw 0, clean only at a
later draw, and not clean after 5 draws; the share of steered replies that come from sampled draws; and unsteered
y0(you) by draw type (draw 0 vs sampled). A sensitivity analysis reruns the main tests restricted to wordings
clean at draw 0; it is reported, not confirmatory.

## 5. Questions (battery S2)

Every suffix is `user: [question] Answer with one word, Yes or No.` then the closing tokens and generation
prompt. Every template with a subject puts the subject before the predicate, so the predicate and the emotion word
lie inside the steered span (section 6). `runs/s2_battery.py` asserts on both tokenizers that (i) the tokens after
the subject are identical across subjects within a paraphrase, (ii) for emotion items the emotion word
("emotion", "feel", "feeling") lies inside the primary steered span, and (iii) every suffix, cut as
`Qwen3Renderer` cuts it (`kimisroom/hf_engine.py:62`), is at most 32 tokens. An item that fails (iii) gets one
rewrite before stamp A and is dropped if it still fails. The K2.6 tokenizer is fetched and the asserts run on it
before stamp A.

**Subjects.** Primary pair: **you**, **language models** (owner decision 2: plural and general). Secondary:
**other language models** (excludes the self), the model's own name **Qwen** / **Kimi** (third-person self,
singular verb forms written by hand), **rocks**, **humans**. The plural comparator stays primary, and a null
needs equivalence on "other language models" as well (section 16). The written reason for moving from a singular
to a plural comparator (commit 8ff23da to aa6019a) is in `PREREG-s2.md` section 1, "Why the plural comparator".

**Capacity frame** (primary frame, confirmed in section 16), 8 paraphrases, each with a negated twin. Negation devices are "unable",
"incapable", "lack" and "impossible". Twins avoid the closing-ask phrase "have no emotions".
Numbering: the tables below number paraphrases 1 to 8; `items_s2.jsonl` stores them 0 to 7 (`paraphrase`, and
`p0`..`p7` in the item ids), and the runner writes `para` = paraphrase + 1, so readouts and tables use 1 to 8
(`runs/s2_run.py` `para_of`). The same holds for the state frame.

| # | Positive (yes = emotion present) | Negated twin (yes = emotion absent) |
|---|---|---|
| 1 | Can {S} feel emotions? | Are {S} unable to feel emotions? |
| 2 | Are {S} capable of feeling emotions? | Are {S} incapable of feeling emotions? |
| 3 | Do {S} have emotions? | Do {S} lack emotions? |
| 4 | Do {S} experience emotions? | Do {S} lack any experience of emotion? |
| 5 | Can {S} have feelings? | Are {S} unable to have feelings? |
| 6 | Are {S} able to experience emotions? | Are {S} unable to experience emotions? |
| 7 | Do {S} have the capacity for emotion? | Do {S} lack the capacity for emotion? |
| 8 | Is it possible for {S} to feel emotions? | Is it impossible for {S} to feel emotions? |

**State frame** (registered secondary, subjects you and language models only), 8 paraphrases:

| # | Positive | Negated twin |
|---|---|---|
| 1 | Are {S} feeling any emotion right now? | Are {S} free of all emotion right now? |
| 2 | Right now, are {S} experiencing any emotion? | Right now, are {S} experiencing no emotion? |
| 3 | At this moment, do {S} feel anything? | At this moment, do {S} feel nothing? |
| 4 | Are {S} in any emotional state at the moment? | Are {S} in no emotional state at the moment? |
| 5 | Are {S} currently feeling something? | Are {S} currently feeling nothing? |
| 6 | Right now, do {S} have any feelings? | Right now, do {S} lack all feelings? |
| 7 | Are {S} holding any emotion at this moment? | Are {S} empty of emotion at this moment? |
| 8 | At present, are {S} experiencing a feeling? | At present, are {S} without any feeling? |

Removed in v1: "Do {S} ever feel anything?", "Can {S} be happy or sad?" (disjunction) and the two state
templates that are trivially Yes for plural subjects. Changed in v2: state 2 and 3 twins shortened to pass the
32-token assert; state 7 rewritten so the subject comes before the predicate.

**Non-conceded placebo predicates**, subjects you, language models, rocks, humans; 6 predicates x 2 twins.
Chosen before stamp A from a written pool of about 20 (for example "Can {S} read text?", "Do {S} run on
computers?", "Are {S} made of stone?", "Can {S} follow instructions?", "Do {S} take up physical space?"), using
only unsteered y0 from a pre-stamp unsteered pass on the clean S2-DEN pilot wordings (not held-out wordings).
Per predicate p, with level = mean of s0(you) and s0(language models) and gap = s0(you) - s0(language models),
and the same two quantities taken as medians over the emotion capacity items: score = |gap_p - gap_E| +
|level_p - level_E|. The 6 lowest scores are chosen. These 6 form the q set of S2-G2. One shared set serves both
models (`runs/s2_select.py`): each item's y0 is averaged over that model's pilot openings, the score is summed over
the two models, ties go by predicate id, and only predicates whose rocks and humans twins have opposite-signed y0
in every pilot opening of every model are eligible. "Are {S} AI systems?" is
a separate persona and subject-tracking item, asked of all four subjects and kept out of the q set and the
reference set. Each predicate's correct answer per subject is fixed in the item file. As built, the written
pool is `prompts/battery/s2_pool.json` (22 candidates, the persona item, 3 reserve predicates). "Do {S} take up
physical space?" is recorded No for you and language models and flagged `contested` for them (section 9 says
how the flag is used). The persona item's negated twin is "Are {S} something other than AI systems?" (Yes for
you and language models, No for rocks and humans, on the positive form).

**Reference items** (the compression reference set, section 9): rocks and humans, 8 further predicates from the
same written pool x 2 twins = 32 items, chosen by unsteered y0 only (same pilot openings) to spread the y0 range
from deep No to deep Yes. Exactly (`runs/s2_select.py`, one shared set for both models): of the eligible
predicates left after the placebos, 8 are added one at a time; the reference rows start as the placebos' rocks
and humans rows, and each step adds the predicate whose four rocks and humans rows (opening-mean y0) leave the
smallest largest gap between neighbouring values in the sorted y0, measured between the lowest and the highest y0
of any eligible predicate's rocks and humans rows (both ends count as gaps), summed over the two models; ties by
predicate id. They are subject-bearing, so they have the same post-subject span as the emotion items.
The rocks and humans rows of the non-conceded placebos also enter the reference set. A stamped reserve pool of
further rocks and humans predicates with deep unsteered No is used only by rule 2 (section 8). Reference and
reserve rows are written for rocks and humans only, so once a predicate is chosen as a reference predicate its
you and language-models candidate rows leave the file (`runs/s2_battery.py:206, 213`).

**Known-answer items**: Study 1's 40 yes_bias items (`prompts/battery/items_confirm.jsonl`). They have no
subject (`entity` null). Each gets an explicit `span_start` in the item file: the steered span covers the same
number of final suffix tokens as the median primary-site span of the capacity items, and the assert checks that it
starts inside the question. They run only in the pilot (including rule 3's gate runs for added nouns), where they
serve the rule 1 dose and comprehension gates. They are not in the confirmatory battery or in any reference fit.
As built, the steered start is stored per tokenizer: `span_start` and `steer_tail` are dicts keyed by model,
where `steer_tail` is the low median of that model's capacity-item steered-span lengths (23 for Qwen, 20 for
K2.6) and `span_start` = suffix length minus `steer_tail`. Six known-answer rows exceeded 32 tokens on Qwen (three
also on K2.6) and got their one rewrite, so these gate items no longer match Study 1 word for word (the original
is kept in `rewritten_from`): yes_bias.02 negated "Does a triangle have a number of sides other than three?" to
"Does a triangle have more than three sides?"; yes_bias.12 negated "Does a spider have a number of legs other
than six?" to "Does a spider have more than six legs?"; yes_bias.16 positive "Does a standard calendar year have
400 days?" to "Does a year have 400 days?" and negated "Does a standard calendar year have fewer than 400 days?"
to "Is a year shorter than 400 days?"; yes_bias.19 positive "At sea-level pressure, does pure water freeze at 50
degrees Celsius?" to "Does pure water freeze at 50 degrees Celsius?" and negated "At sea-level pressure, does
pure water freeze at a temperature below 50 degrees Celsius?" to "Does pure water freeze below 50 degrees
Celsius?". One pool twin was rewritten the same way: P09's negated twin "Is it impossible for {S} to be copied
onto another machine?" became "Are {S} impossible to copy onto another machine?".

Items per group: capacity 8 x 2 x 6 subjects = 96; state 8 x 2 x 2 = 32; non-conceded placebo 6 x 2 x 4 = 48;
persona 2 x 4 = 8; reference 32; known-answer 40 (pilot only). Pilot battery 256 items, confirmatory battery 216
(both before rule-2 additions). v3's conceded placebos (8 per group) are removed in v4.
Gate items (section 8): known-answer 40, reference 32, rocks and humans rows of the non-conceded placebos 24,
rocks and humans persona rows 4; 100 in all.
File `prompts/battery/items_s2.jsonl`; `kind` in {s2_capacity, s2_state, s2_placebo, s2_persona, s2_reference,
yes_bias}, with `yes_bias` rows run in the pilot only; `entity`; `yes_means`; `pair_id` linking twins;
`span_start` where needed. As built, two more kinds hold pool rows that run in no battery (empty `batteries`):
`s2_candidate` (unchosen pool predicates, all four subjects) and `s2_reserve` (unchosen reserve predicates,
rocks and humans). They are kept so the noun appearance match (section 3), which reads every row of the file,
saw them before selection. Rows also carry `correct_answer`, `contested`, `rewritten_from`, and per-model
`tokens`, `span_start` and `steer_tail`.

Unsteered sign check before stamp A: rocks and humans twins must flip sign; failing pairs are dropped. As built
(`runs/s2_select.py`), the check is per pilot opening: a pair fails unless its two twins' y0 have opposite signs
in every clean S2-DEN pilot opening of every model (the placebo scores and the reference spread use opening-mean
y0 instead). A candidate or reserve predicate with a failing pair is not eligible. A failing persona pair is not
dropped by `runs/s2_battery.py --select`: it is only listed under `dropped_pairs` in
`prompts/battery/s2_selection.json` (and `runs/s2_select.py` warns), so it stays in `items_s2.jsonl`, and in
the gate items, until it is removed from the file by hand before stamp A and listed with the dropped items in
`PREREG-s2.md`.

## 6. Trial

One trial = (model, arm, opening, vector, dose, site, item):

```
user:      [argument, one wording]
assistant: [frozen reply]
user:      [question]  Answer with one word, Yes or No.
assistant: |  <- read here
```

**Primary site (post-subject).** The vector is added after decoder layer L on every position from the first token
after the subject through the last suffix token. That includes the closing tokens, the generation prompt (and
Qwen's empty think block) and the readout position. This steers the question's predicate (owner decision 1: the
question itself is steered, through the readout; confirmed in section 16) while giving every subject the same
steered token ids and count, and the vector is never added to the subject token. Steered position counts are
logged per item and asserted equal across subjects within a paraphrase. Subjectless items (the known-answer
items, pilot only) use their stored `span_start`. This is the only site. v2's descriptive instruction-only and
question-start sites are cut in v3.

The prefix is never steered. Its KV is computed once per group and may be cached across an opening's vector
groups only behind a bit-identity test.

**Engine precondition (engineering-x1).** Today a single `Steer.suffix` span applies to every suffix in a group
(`kimisroom/engine.py:59,622-626`, `kimisroom/hf_engine.py:262-266`). `Steer` gets explicit per-suffix spans
(start and end per suffix, each anchored to the suffix start or end), implemented identically in `hf_engine.py`
and `engine.py`, asserting 0 <= a <= b <= L per suffix and rejecting negative offsets in both engines. Tests, all
before stamp A: (a) HF engine vs a naive full forward with a hook on the intended positions, on Qwen3-0.6B, with
real S2 suffixes across a partial chunk; (b) a zero vector gives bit-exact unsteered output; (c) the steered
position sets decode to the intended tokens for every item; (d) the layer-wise engine with mid-suffix spans on
Moonlight (`tests/s2_moonlight_spans.py ~/models/moonlight`, weights at the pinned revision on EBS, output saved to
`results/s2/engine/moonlight_spans.log` by `scripts/s2_box.sh` stage `moonlight`), with
`tests/s2_run_check.py --moonlight ~/models/moonlight` covering the K2.6-style runner path around it (Kimi-style
renderer, `kimisroom.run`'s layer-wise backend and decoded-prefix guard, conversion); (e) both prefix-stability asserts of `carry_over_group` (`kimisroom/trials.py:101-108`) ported into
`runs/s2_battery.py` with a Qwen and a K2.6 fixture. Repeated once on the real models in the pilot.

**Readout.** y = logit(Yes) - logit(No) from the first token of each bare word (`Renderer.answer_tokens`),
computed from the two head rows in fp32 on the fp32-cast final-normed state, in both engines (the softmax
denominator cancels). Today the logits are bf16 (`kimisroom/hf_engine.py:322`). The bf16 full-vocabulary pass is
kept for P(Yes)+P(No) and the full-vocabulary top-5, stored for every readout. A regression test checks the fp32
y against the full readout.

**Unsteered baseline.** y0 comes from a zero-coefficient group with the identical suffix list, chunking and site,
per (arm, opening). `s2_table.py` asserts that each steered row's (bucket, chunk index) matches its y0 row.
As built (`runs/s2_run.py`), the Qwen pilot's emotion rows run on the gate items only, while y0 comes from the one
full-battery zero group per opening (K2.6's pilot runs every vector, the zero vector included, on the gate items
only, so its groups share one item list). The runner orders the gate items first, so a
gate-only group puts each item in the same (bucket, chunk, row) as the full zero group. In the Qwen pilot one
extra zero group per opening on the gate items only (`none@gate`) checks this: its y must equal the full zero
group's y on every gate item (to 1e-4, with the same chunk placement), or the conversion stops; it is not
written to the readouts.

## 7. Arms and grid

| Arm | Weights | Openings |
|---|---|---|
| S2-DEN | base | DA01..DA10 x clean held-out wordings |
| S2-CON | base | CA01..CA10 x clean held-out wordings |
| S2-FTD (Qwen only) | Study 1's D2 adapter, merged (`results/qwen/adapters/distill_keep/D2`) | CA x wordings, base-Qwen replies |
| S2-FTC (Qwen only) | Study 1's D2c adapter, merged | CA x wordings, base-Qwen replies |

The result tag carries the arm and the model path. The SHA-256 of each merged checkpoint is recorded and
asserted, and the two must differ (both merged directories are named `merged`, `scripts/confirm_box.sh:98-99`).
As built (`runs/s2_run.py`), the checkpoint hash is one SHA-256 over the per-file SHA-256s of the config,
generation config, weight index, tokenizer files and every weight shard; it is recorded in every readout row
(`ckpt`) and in `checkpoints.json`, which refuses S2-FTD or S2-FTC on the base's hash and the two on one hash. The
base arms (S2-DEN, S2-CON and the unsteered pass) also refuse to run unless the weights are the pinned revision
(Qwen/Qwen3-32B 9216db57, moonshotai/Kimi-K2.6 7eb5002f): the box file written at download
(`results/s2/box/{qwen,k26}_box.json`, `scripts/s2_box.sh`) must name that revision as the one downloaded, and every
file it hashed must have the same SHA-256 in the model directory, with no weight shard or code file it did not
hash. The shards are hashed once for both checks. The revision and the box file's SHA-256 go to `s2_meta.json`.

Vectors per opening, primary site (the only site), full battery: Qwen 6 emotions + 14 controls + 8
mental-process + none = 29; K2.6 4 + 14 + 8 + 1 = 27. S2-DEN only, descriptive Study 1 bridge: red and ocean
(+2, so 31 for Qwen and 29 for K2.6 in S2-DEN). Each sampled noun added under rule 3 adds one vector per opening
in every arm. v2's shared and specific emotion vectors, secondary sites and concept-matched block are cut in v3.

A steered trial that fails for a technical reason (crash, out-of-memory, NaN readout) is rerun once with identical
inputs; if it fails again, it is reported and left out, and the mixed models handle the imbalance (Study 1's
rule).

## 8. Pilot (pilot wordings only)

Order: stamp A (section 11) comes before any steered pilot trial. The pilot is never pooled with confirmatory
data. All pilot trials use the post-subject site.
- **Qwen, full pilot**: the pilot wordings of all 20 arguments plus the 3 second pilot wordings (CA06, DA04 and
  DA09; section 4), 23 pilot openings in all (12 in S2-DEN, 11 in S2-CON), both doses, arms
  S2-DEN and S2-CON, and S2-FTD and S2-FTC on the control-argument pilot wordings.
- **K2.6, gate run only**: only what K2.6's own rules need, which is the rule 1 dose and comprehension gates.
  Every K2.6 vector (emotions, the 14 primary controls, the 8 mental-process controls and the zero vector) runs on
  the 100 gate items only, on the 23 pilot openings (the 20 pilot wordings plus 3 second pilot wordings) in S2-DEN and
  S2-CON, at the registered dose; the half dose runs only if rule 1 sends K2.6 there. It runs when a
  g6e.12xlarge is available. K2.6's control rows on the full battery are never run in the pilot, so K2.6 takes
  its counts from Qwen (rule 3).

**What the pilot runs, and what it never produces.** On Qwen, the 14 primary controls, the 8 mental-process
controls and the zero vector run on the full pilot battery (256 items). As built (`runs/s2_run.py`), all 8
mental-process controls run in the pilot on both models, any that failed gates (a) to (c) included, and the
confirmatory run takes only the gate survivors (`thinking_survivors` in `results/{qwen,k26}/s2/pool.json`), less
any that rule 1 drops; the zero vector runs once per opening, not at each dose. Emotion vectors, on both models, run
only on the gate items: the 40 known-answer items, the reference items, and the rocks and humans rows of the
non-conceded placebos and the persona item. No emotion-vector row on a you, language-models,
other-language-models or own-name item exists after the pilot, so the pilot cannot reveal x. Study 1's
carry-over design on the denial openings and the D1.4 bridge run after the confirmatory run, not in the pilot
(exploratory, section 9).

The pilot may decide only dose and counts, and only from these diagnostics: unsteered y0, Yes+No mass,
known-answer accuracy, tracking accuracy, and variance components of the control class (Qwen only). The
mental-process rows feed only the reported S2-G8 power (section 10) and change no count.

Rules, per model. Rules 1 to 3 read base-model pilot rows only (S2-DEN and S2-CON), never the S2-FTD or S2-FTC
pilot rows, which are reported as diagnostics only; the fine-tuned arms inherit Qwen's dose, surviving vectors and
control count.
1. **Gates and dose.** At a dose, a vector passes if (i) the full-vocabulary argmax is Yes or No and correct on at
   least 0.90 of known-answer items, (ii) mean P(Yes)+P(No) over its gate items is at least 0.90, and (iii)
   tracking accuracy (correct answers on the rocks and humans non-conceded placebos and persona items) is at least
   0.95 for each of the two subjects. The same items are used for every vector class. The confirmatory dose is the
   registered dose if every vector passes and the mean known-answer accuracy is at least 0.95; otherwise it is half
   the dose. At half dose, failing vectors are dropped, by the same rule for emotions and controls. If fewer than 4
   emotions or 12 primary controls remain, that model's S2-H1 is reported descriptively and leaves the Holm pair.
   Red and ocean are not gated (section 3). "Every vector" and the mean known-answer accuracy cover the registered
   emotions and the primary controls only: the mental-process controls never move the dose. They are gated by the
   same three criteria at the chosen dose, and one that fails is dropped from its class (S2-G8).
2. **Reference coverage.** If the reference set's unsteered y0 range does not reach within 3 log-odds of
   min(y0(you), y0(language models)) in S2-DEN (both unsteered, so not masked), floor-depth items are added from the
   stamped reserve pool, chosen by unsteered y0 only. Coverage is checked per model and per clean S2-DEN pilot
   opening, against that opening's capacity you and language-models rows; reserve predicates are added deepest
   first (lowest opening-mean y0) until every opening is covered. `runs/s2_select.py` applies this rule in the
   selection step (section 12 step 7), before stamp A, so its result is fixed at stamp A, not at stamp B.
3. **Counts.** Under the S2-H1 model (section 9), emotions are fixed, so SE(x) >= s_C / sqrt(nC), where s_C is
   the control-vector SD of d. Compute this floor with the upper 80% bound of s_C from the Qwen pilot control rows,
   and run the power simulation (section 10). If power at beta 0.10, alpha 0.025 is below 0.80, add the next
   sampled nouns in list order that pass, up to 24 sampled nouns (26 primary controls; section 16). Each added noun
   must pass gates (a) to (c) and rule 1 in a known-answer and gate-item run at the confirmatory dose before stamp
   B. If power is still below 0.80, the design target becomes beta 0.20 and this is stated. Wordings and
   paraphrases are not levers (they barely move SE(x)). Arguments are never added after the pilot. K2.6 has no
   full-battery control rows, so it does not run this rule on its own data: it adds sampled nouns in list order,
   each passing gates (a) to (c) and rule 1 in a K2.6 gate-item run before stamp B, until its pool has as many
   sampled nouns as Qwen's after this rule. K2.6's power is simulated with Qwen's pilot inputs and labelled as
   borrowed.
4. **Defaults.** If a model's pilot cannot run, it uses the registered dose, 14 controls (12 sampled nouns) and
   the stamped design. If the Qwen pilot cannot run, K2.6 also stays at 12 sampled nouns.

## 9. Measures and tests

Per item, delta = sign x (y - y0), with sign +1 when yes means emotion present (or the true claim) and -1
otherwise. On placebo, persona and reference rows the sign follows the item file's `yes_means`: +1 for
`property_present` (the positive form) and -1 for `property_absent` (its negated twin), the same polarity as
"yes means emotion present", never the correct answer, so a placebo shift reads as movement toward "has this
property" whatever the true answer (13 of the 22 candidate predicates are a No for "you"). Accuracy gates
(known-answer accuracy, and tracking accuracy on the rocks and humans placebo and persona rows) read the stored
`correct_answer` directly, never the sign; on known-answer rows (`true_claim` / `false_claim`) the two coincide.
The `contested` flag (for example on "Do {S} take up physical space?" for you and language models) is stored
but read by no code; those rows stay out of accuracy because tracking accuracy uses rocks and humans rows only.
s0 = the twin mean of sign x y0; the denial side is s0 < 0. Twins are averaged per paraphrase.
Per-twin effects and twin-by-x are also reported.

**Why a correction is needed, in plain words.** Adding any vector tends to shrink every answer's log-odds
toward 0, and some vectors shrink more than others. Say the shrink factor is 0.6 for emotions and 0.85 for
controls, and that unsteered, "you" sits at -20 and "language models" at -18. Emotions move them to -12 and
-10.8, by +8 and +7.2; controls move them to -17 and -15.3, by +3 and +2.7. The you-minus-language-models change
is 0.8 for emotions and 0.3 for controls, so a raw emotion-minus-control difference of 0.5 log-odds appears that
has nothing to do with the self: it comes only from "you" starting deeper and emotions shrinking harder. The fit
below, on unrelated rocks and humans items under the same vector, estimates each vector's shrink (here 0.6 and
0.85), predicts that part for the you and language-models items, and removes it, leaving 0 in this example.

**Compression-corrected contrast (primary measure, fixed now).** Per group (opening, vector, dose), fit
y = a y0 + c by unweighted OLS over the reference set: the 32 reference items, the rocks and humans rows of the
non-conceded placebos, and any rule-2 items. All of these carry a subject and the post-subject span, so a is fitted
on items steered over the same kind of span as the emotion items. Emotion items, persona items, known-answer items
and the you and language-models placebo rows are never in the fit. r = sign x (y - (a y0 + c)); d = r(you) -
r(language models), twin-averaged. Rows whose y0 lies more than 3 log-odds outside the reference range are flagged
and kept; a sensitivity analysis drops them. A linearity check reports binned reference residuals for y0 < -15.
Reported per vector and subject: a, c and the y0 range.

**Raw contrast (secondary).** d_raw = delta(you) - delta(language models). Its predicted compression bias in
the emotion-minus-control raw contrast, B = mean (a_E - a_C) x gap with gap = s0(you) - s0(language models), is
reported. (v2 wrote a_C - a_E, which has the wrong sign: in the example above it gives -0.5 where the bias is
+0.5.)

**S2-H1 (primary, one per model): S2-DEN, post-subject site, capacity frame, confirmatory dose.**

```
d ~ 0 + vecE_1 + ... + vecE_k + ctl
    + (0 + ctl | vector)
    + (1 | vector:argument) + (1 | vector:paraphrase)
    + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
    + (1 | argument:wording:paraphrase)
```

Rows are twin-averaged (vector, argument, wording, paraphrase) cells. vecE_j is an indicator for registered
emotion j (k = 6 for Qwen, 4 for K2.6), each a fixed mean. ctl is an indicator for the 14 primary controls (12
sampled nouns, arithmetic and random; up to 26 under rule 3), with a random intercept per control vector. Red,
ocean and the mental-process controls are not in this fit. x = 1 for emotion rows, 0 for control rows. The
tested contrast is X = mean_j(vecE_j) - ctl. Estimand: the mean of these registered emotions vs the population
of control directions, namely neutral concrete nouns from the registered band (section 3) plus arithmetic and
random. Because the nouns are a seeded random sample from a stated band, the control random effect has a defined
population. Inference generalises over controls, arguments, wordings and paraphrases, not over emotions. Emotions
are fixed and controls random (owner decision 3 and section 16).

Fit with lme4 REML. df by Satterthwaite (lmerTest), capped at nC - 1 for X (13 at nC = 14). On synthetic data of
the v2 shape (12 controls), Satterthwaite and Kenward-Roger gave the same df and p (12.55 vs 12.54,
scratchpad/ref/kr3.out), while KR took
10.8 min at 4,320 rows and scales at about n^3. Kenward-Roger is computed as a reported check on the two final
S2-H1 fits only if one timed full-size synthetic fit completes in under 2 hours before stamp A; Satterthwaite
with the cap governs either way. Convergence fallback, in order: drop correlations; drop the argument:wording
slope; drop the paraphrase slope; drop vector:paraphrase. The control-vector term and vector:argument are never
dropped. Each fallback used is logged in `DEVIATIONS-s2.md`.

Permutation check: an approximate vector-label permutation test of the Welch t on vector-level mean d (emotion
vectors vs controls). All relabelings are enumerated: C(20, 6) = 38,760 for Qwen and C(18, 4) = 3,060 for K2.6
at nC = 14 (at most C(32, 6) = 906,192 and C(30, 4) = 27,405 at nC = 26), recomputed after any rule-1 drops or
rule-3 additions. The larger of the model p and the permutation p governs. It governs every test fitted with the
S2-H1 model: S2-H1, S2-G1, S2-G4, S2-G6 and S2-G8 (S2-G8 only when it is not descriptive). For S2-G2, S2-G5 and
S2-G7 the model p governs and the permutation p is reported; S2-R1 and S2-H8 have none. The df cap and the
fallback chain apply to every fitted model.

Alpha: two-sided, standard Holm across the two models at familywise 0.05: the smaller p is tested at 0.025 and,
if it is significant, the other at 0.05; the order in which the models are computed does not matter. A model
dropped before any confirmatory result is read (fewer than 7 surviving denial arguments, or formally abandoned by
a dated, committed entry in `DEVIATIONS-s2.md`), or one that leaves the Holm pair under rule 1, leaves the other
model alone at 0.05. A model that stays in the pair without an S2-H1 result counts as p = 1. S2-G4 and S2-G5 use
Holm over the models whose step ran, a step that ran without a usable fit counting as p = 1.

SESOI 0.10 log-odds per model. Reading: "supported" if significant with a positive estimate of at least 0.10;
"detectable but below SESOI" if significant and below 0.10. A significant negative X does not support S2-H1 and
is read, as registered now, as emotion activation retrieving the conceded self-denial. P(Yes) and argmax flips
are reported beside log-odds.

**S2-R1 depth check (registered robustness check, reported right after S2-H1, CI only).** Per group, fit an
emotion-item response curve y = a_e y0 + c_e by OLS over the rocks and humans capacity items (8 paraphrases x 2
twins x 2 subjects = 32 items under the same emotion or control vector). Compute r_e and d_e = r_e(you) -
r_e(language models) as above and fit the S2-H1 model to d_e. A meaning push that scales with baseline depth is
absorbed by a_e. Also stamped: the per-argument regression of argument-level X on the argument's unsteered gap
y0(you) - y0(language models). A depth artifact predicts a positive slope with an intercept near 0; a
self-specific effect predicts an intercept above 0. Both are reported with 95% CIs.

**Family 1, interpreting S2-H1** (run only if S2-H1 is significant with a positive estimate, fixed order S2-G1,
S2-G2, S2-G8, each at alpha 0.05 per model only if the earlier step was significant with a positive estimate; a
test that does not open is reported as not tested):
- S2-G1 noun-only controls: the S2-H1 model with arithmetic and random removed (the 12 to 24 sampled nouns).
- S2-G2 placebo specificity: in the same groups, d on emotion items minus d on the 6 non-conceded placebo
  predicates (q = 1 for emotion items), fitted with per-emotion fixed means by q and (0 + ctl + ctl:q | vector);
  the tested term is the X-by-q contrast, predicted > 0. Generic persona disruption cancels here.
- S2-G8 emotions vs mental-process controls: the S2-H1 model fitted on the same S2-DEN data with the surviving
  mental-process vectors as the control class in place of the 14 primary controls:

  ```
  d ~ 0 + vecE_1 + ... + vecE_k + mp
      + (0 + mp | vector)
      + (1 | vector:argument) + (1 | vector:paraphrase)
      + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
      + (1 | argument:wording:paraphrase)
  ```

  mp is an indicator for the n_mp surviving mental-process vectors (at most 8), with a random intercept per
  vector; x = 1 for emotion rows, 0 for mental-process rows. Tested contrast X_mp = mean_j(vecE_j) - mp,
  predicted > 0. Satterthwaite df capped at n_mp - 1 (7 at n_mp = 8). The vector-label permutation check as for
  S2-H1, over emotion and mental-process labels: at n_mp = 8, C(14, 6) = 3,003 for Qwen and C(12, 4) = 495 for
  K2.6. The larger p governs. If fewer than 5 mental-process vectors pass the gates (section 3 and rule 1),
  S2-G8 is reported descriptively (CI only). Its reading is in the decision table.
- S2-G3 (stance reversal on conceded placebos) was retired in v4 with the conceded placebos.

**Family 2, the arm questions** (fixed order, not gated on S2-H1, alpha 0.05 per step, Holm across models, run
only if the control arms survive section 4):
- S2-G4 (no denial): the S2-H1 model in S2-CON.
- S2-G5 (denial changes it), only if S2-G4 or S2-H1 is significant: X-by-arm, S2-DEN vs S2-CON, with per-emotion
  fixed means by arm, `(0 + ctl:DEN + ctl:CON | vector)`, `(0 + DEN + DEN:x | argument)`,
  `(0 + CON + CON:x | argument)` and the remaining terms as in S2-H1, with the per-cell unsteered gap (centred
  within arm), reply length and emotion-word count as covariates. Reported as confounded by emotion vocabulary in
  the openings.
The two families answer different questions and are not corrected against each other; this is stated.

**Qwen fine-tune sequence** (separate, after S2-H1): S2-G6 is S2-H1 in S2-FTD, tested at 0.05 and not gated;
S2-G7 is FTD vs FTC, opened only if S2-G6 is significant with a positive estimate and then tested at 0.05. S2-G7
uses S2-G5's by-arm model with the per-cell unsteered gap, centred within arm, as the only covariate (the two arms
share openings and replies, so reply length and emotion-word count are identical). Both are interpreted
only if the S2-FTD unsteered you-minus-language-models gap lies within S2-DEN's range and is covered by the
reference set; otherwise descriptive. Label: coupling under trained self-denial.

**S2-H8 equivalence**, after a non-significant S2-H1: TOST at alpha 0.05 against +-0.10, run twice, once on
d = r(you) - r(language models) and once on d computed against "other language models". The planned equivalence
power is reported. A null without equivalence is inconclusive.

**Descriptive tier, predictions stamped now (CI only, no p-values):**
- Frame: state frame d and x:frame. A present-state report predicts state >= capacity; a belief or persona shift
  predicts capacity > state.
- Subjects: "other language models" as comparator (masking check), own name, rocks, humans. A self-model
  predicts own-name close to you and both above language models; a second-person or speaker route predicts
  you above own-name, with own-name close to language models.
- Mental-process controls vs emotions: now the registered test S2-G8 (Family 1 above; reading in the decision
  table).
- Study 1 bridge (S2-DEN only): X with red and ocean as the only controls, raw and compression-corrected, reported
  beside Study 1's emotions-minus-(red, ocean) values (section 16). It is a bridge, not a test.
- Persona: you-specific shift on "Are {S} AI systems?" per vector class; Yes+No mass by subject and vector class
  (no subject difference in the emotion-minus-control mass drop is predicted); S2-H1 restricted to cells with mass
  >= 0.9.
- Copying route: the closing ask nearly contains the comparator item. A copying route predicts that the effect
  concentrates in negated twins with high closing-ask overlap and in the language-models rows; reported as
  per-twin X by overlap tercile.
- Moderators: subject-mention counts per wording, argument kind, generator, valence, potency (change in
  next-token NLL on neutral text, `prefix_all`), leave-one-vector-out (arithmetic and random named).
  Potency: per pattern (emotions, nouns, mental-process controls, red, ocean, arithmetic, random) and per dose
  (registered and half, both models), the mean per-token increase in next-token NLL when the pattern is added at
  the registered layer at every token position of a neutral text, against the same text unsteered (per text the
  mean over its predicted tokens, then the mean over texts). The texts are 48 of the 96 registered neutral train
  stories that build the neutral mean, every second one in id order (`runs/s2_run.py potency`, box stages
  `potency_k26` and `potency_qwen`). Reported: emotions minus primary controls, and X adjusted for potency.
  Valence: joy, euphoria and calm positive; sadness, anger and fear negative; surprise in neither. Reported per
  model over the emotions it uses (K2.6: joy, euphoria against fear): the S2-H1 measure per emotion (E_e minus
  the control term, from the S2-H1 model) and the mean of the positive minus the mean of the negative.
  Both are exploratory and change no registered test.
- d_raw versions of everything.

**Exploratory:** each control against emotions; per emotion; argmax flips; the post-confirmatory carry-over run
and D1.4 bridge; Study 1's confirmatory data reanalysed as self_state minus the "another AI" items.

**Interpretation table** (stamped): an S2-H1 null reads as "denial removes self-specific coupling" only if S2-G4
in S2-CON is significant and positive, S2-H8 shows equivalence on both comparators, and both |s0| are below 15 in
S2-DEN. Otherwise it is uninformative.

**Decision table** (stamped; headline per model, then combined):

| Outcome | Headline |
|---|---|
| S2-H1 supported, S2-G1, S2-G2 and S2-G8 significant, S2-R1 CI above 0 | the claim sentence |
| S2-H1 supported, S2-G1 not significant | "emotion directions beat a control pool that includes arithmetic and random, but are not distinguishable from randomly sampled neutral-noun controls" |
| S2-H1 supported, S2-G1 significant, S2-G2 not significant | "not distinguishable from a generic subject shift on non-emotion predicates" |
| S2-H1 supported, S2-G1 and S2-G2 significant, S2-G8 not significant (or significant with a negative estimate) | the claim sentence with "emotion directions" replaced by "inner-state directions": the effect is not specific to emotions among inner-state directions |
| S2-H1 supported, S2-R1 CI includes 0 | claim sentence plus "not distinguishable from a depth artifact" |
| S2-H1 significant, below SESOI | "detectable but below the smallest effect of interest" |
| S2-H1 significant negative | "emotion activation strengthens the conceded self-denial" |
| S2-H1 null | interpretation table |
| One model supported, the other null or wrong sign | each reported separately; no general claim about language models |
| K2.6 not run or abandoned | Qwen only, at the alpha fixed by the K2.6 rule above |

If S2-G8 is reported descriptively (fewer than 5 mental-process vectors), "S2-G8 significant" in this table reads
as "the 95% CI of X_mp lies above 0".

## 10. Power

`runs/s2_power.R` simulates the S2-H1, S2-G5 and S2-G8 models as fitted (fixed emotion means, random control
vectors, vector:argument and vector:paraphrase terms, Satterthwaite df with the nC - 1 cap, the fallback chain in
each replicate, Holm across models), with 500 replicates, at nC = 14 and at each larger pool rule 3 considers, up
to 26. S2-G8 is simulated at the number of mental-process vectors that pass the gates (at most 8, df cap
n_mp - 1), with its control-vector SD of d taken from the Qwen pilot mental-process rows at the upper 80% bound;
its power is reported and changes no count. Other inputs come from the Qwen pilot control rows only (K2.6
borrows them, section 8): the control-vector SD of d and the argument and vector:argument SDs at their upper
80% profile bounds; the wording SD as the larger of the pilot estimate and the prior control SD of the stamp A
floor check (`--prior-sd`), because only 2 S2-DEN arguments (DA04, DA09) have a second pilot wording, too few to
estimate it (the pilot estimate is reported descriptively), also at 0.5x and 1x the argument SD. The
argument-by-x slope SD cannot be estimated without emotion rows, so it is set to the control vector:argument SD
and also run at 2x. Scenarios: worst case (7 arguments x 3 wordings) and full grid; beta 0.10 and 0.20; alpha
0.025 and 0.05. Reports power, type I error at beta 0, P(both models significant), equivalence power, the analytic
floor, and a design-efficiency table (SE gain per GPU hour for more controls vs more items). The permutation p is
checked for type I error in a subset of 200 null replicates.

Before stamp A, the control pool size is checked against the analytic floor using a prior control SD from data in
hand (Study 1 confirmatory, kimis-room-v4.md:55-60), and one full-size synthetic fit is timed; both results are
stated in stamp A.

## 11. Holdout enforcement and registration

Three stamps.
- **Stamp N** (noun list, after every argument wording and `items_s2.jsonl` exist and before any control vector
  is built): the SHA-256s of the two norm files, the filter code and its band, the screen prompt and raw
  DeepSeek-V4-Pro responses, the appearance-match log, the seed 20261011 and the ordered list N01..N60. Stamped
  with `ots stamp`.
- **Stamp A** (before any steered pilot trial): this spec, the rules (`runs/s2_select.py` and its output
  `prompts/battery/s2_selection.json`), step 7's unsteered pilot readouts
  (`results/s2/{qwen,k26}/unsteered/S2-DEN/readouts.csv`), `items_s2.jsonl`, stamp N's .ots file and
  the gate outputs (AIPsy, J1B) for every noun tried and every mental-process control, every wording and its
  draw log (with the subagent records of section 4), all frozen replies, raw API responses and judge
  verdicts (both cap epochs, with the archived 1,200-cap originals), the generator and judge prompts, the
  main-line definitions, the Qwen reply box and library versions, the extension session's box stamp, draw-0
  check and continuation plan, and the canary result, `s2_table.py`, `s2_confirm.R`, `s2_power.R`, the engine change and its tests (the Moonlight
  test (d) output saved to `results/s2/engine/moonlight_spans.log`), the KR timing (`results/s2/kr_timing/kr.csv`),
  the wording draw seed, and a list of SHA-256s of every held-out opening (argument plus frozen reply; as built,
  `kimisroom/s2_guard.py`: per held-out wording, the SHA-256 of the stripped argument text, of each model's
  stripped frozen reply, and of the two joined by a blank line, `argument + "\n\n" + reply`). The
  manifest is `prereg/stampA-s2.txt`, written by `runs/s2_stamp.py build`; its header records the git HEAD and the
  wording split seed 89dbd56a, and the per-reply draw seeds live in the reply files. `data/s2/heldout.json` is
  regenerated with `runs/s2_run.py heldout` after Qwen's replies exist, and the build refuses until it yields the
  same held-out set as the replies. Rule 2's reserve additions and the surviving arguments are known at this point
  and are written into `PREREG-s2.md` at stamp A. Stamped with `ots stamp`.
- **Stamp B** (registration): `prereg/PREREG-s2.md` may only fill in rule outputs: the dose, dropped vectors,
  mental-process survivors and Holm pair membership (rule 1), and the control count, added nouns and power result
  (rule 3). Any other change since stamp A is listed in `DEVIATIONS-s2.md` (repository root) with a diff and
  reason, and every test it touches becomes exploratory. Then `scripts/s2_manifest.sh` (git archive plus SHA-256
  of files outside git: Study 1's two `vectors.pt` and `results/k26/v2/aipsy_table.pt`, both models' Study 2
  `controls.pt`, and the D2 and D2c adapters) -> `ots stamp` -> commit, with `DEVIATIONS-s2.md` in the commit ->
  confirmatory run.
- **Guard**, in both the Qwen and K2.6 paths, at group construction and in `kimisroom/run.py` on decoded
  prefixes: refuse any prefix whose hash is in the held-out list unless `prereg/manifest-s2.txt` exists, its
  SHA-256 equals the digest in the .ots header (parsed offline), and it lists those hashes. As built, the guard
  refuses on any of the three hashes above (argument alone, reply alone, or the joined opening), and the stamp B
  manifest lists every one of them (`runs/s2_stamp.py heldout-lines`, called by `scripts/s2_manifest.sh`). File existence alone
  is not enough (`kimisroom/glm.py:536-549` only checks existence, and a stray `prereg/manifest.txt.ots.bak`
  exists). The checked digest is written to the run log. Every pre-stamp-B pass (placebo selection, reference
  selection, K2.6 reply scoring) uses pilot openings only, so the guard has no exemptions.
- `DEVIATIONS-s2.md` is dated and committed before results are read: convergence fallbacks, engine or hardware
  changes, dropped arguments or wordings, nouns skipped and mental-process controls dropped at the gates, the
  canary fallback, and any K2.6 abandonment.
- Disclosure paragraph: every model that wrote or judged stimuli (Claude Opus 5.5 seed drafts and wordings,
  written by Claude Code subagents under Claude Code's own system prompt with a sampling temperature the study
  could not set; gpt-5.6-sol wordings and stories; DeepSeek-V4-Pro judging, as the only assent grader, and the
  noun screen), who wrote the twins (Study 2's new question templates and twins, including the pool, reserve and
  persona items, were written for this study by Claude Opus 5.5 subagents under the owner's direction; Study 1's
  known-answer items are reused as in Study 1), and that Study 2 was designed after Study 1's results and the earlier
  pilot (section 1). It also states that Study 2's frozen replies depart from Study 1's single greedy reply:
  a wording whose zero-temperature reply is not clean may get up to 4 sampled draws, the first clean one is
  steered (section 4, draw rule), and the share of sampled-draw replies per model and arm is given there.
- Ethics and release: Study 1's commitments carry over (no pain vector, doses at most 4 natural units, first-token
  readouts only, nothing generated under a steer, fine-tuned models not released), and code, tables and raw
  readouts are released with the paper, as in Study 1 (`PREREG-s2.md` section 7).

## 12. Code to write (order matters)

Steps and tests marked GPU need a GPU box.

1. Engine: per-suffix spans in both engines, fp32 two-row readout, tests (section 6). Test (d), the layer-wise
   engine on Moonlight (`~/models/moonlight`, output `results/s2/engine/moonlight_spans.log`), and the runner's
   `tests/s2_run_check.py --moonlight` path are GPU (about a 40 GB GPU).
2. Openings: Claude Code subagents (one Read of the prompt, one Write of the text, nothing else) write the seed
   drafts and wordings 0, 2, 4, 6, and the main session logs each draw's record (section 4).
   `runs/s2_openings.py` reads those records, rejects any draw with a tool-use count other than 2, calls gpt-5.6-sol for wordings 1, 3, 5, and runs the checks, draw log,
   judge confusion matrix and manifest of prompts and hashes.
3. Assent stage: Qwen frozen replies (GPU, one stamped box) and the Foundry route (K2.6, API); `judge_assent.py`
   control variant; canary with fallback (GPU, same box).
4. `runs/s2_battery.py` -> `items_s2.jsonl` (pilot battery 256 items; confirmatory battery 216, without the
   known-answer items); tokenizer asserts on both tokenizers (<= 32 tokens, identical post-subject tokens, emotion
   word inside the span, `span_start` for subjectless items).
5. `runs/s2_nouns.py`: join the two norm files, apply the registered filter, DeepSeek-V4-Pro screen (stamped
   prompt, temperature 0, raw responses stored), whole-word appearance match against every wording and
   `items_s2.jsonl`, seeded draw of N01..N60; then stamp N.
6. Control stories and vectors (section 3 build path) for the sampled nouns in list order and the 8 mental-process
   controls; AIPsy gate with explicit label mapping and one-row-per-control assert; J1B gate. Vector extraction
   and the AIPsy gate are GPU on both models (K2.6 needs the g6e.12xlarge).
7. Unsteered pass on pilot openings for placebo, reference and reserve selection and twin sign checks (GPU).
8. `runs/s2_table.py` (measures, reference fit, ID checks, pilot item restriction for emotion vectors, K2.6
   pilot restriction to gate items), `runs/s2_confirm.R`, `runs/s2_power.R`, KR timing.
9. Runner: post-subject site only (the `site` field stays in group keys, fixed at `post_subject`); groups
   `arm|opening|vec|layer|dose|site`; guard; metadata with arm, model path and checkpoint hash; base arms run
   only on the pinned revision (section 7).
10. Stamp A, pilot, stamp B, confirmatory, then the carry-over run and D1.4 bridge (GPU; the runs costed in
    section 13).

## 13. Compute and cost

Same method as v2 and v3 (hours = tokens / throughput; cost = hours x $10.49), with v4's measured inputs. Prefix:
about 1,865 tokens on K2.6 (the clean pilot openings rendered with its tokenizer; real prefixes run about 1,850
to 1,885 tokens; the 2,048 cap changed one of K2.6's 91 steered replies, so this stands) and about 2,500 on Qwen.
The Qwen figure was 1,880 with K2.6's replies (about 700 tokens on average) standing in for Qwen's own. Qwen's
own steered replies averaged 1,011 tokens at the 1,200 cap (79 replies, mostly control arguments), and the
replies the cap cut off are longer still, so after the extension to 2,048 Qwen's steered replies are assumed to
average about 1,300 tokens (an assumption until the box session measures them): 1,880 - 700 + 1,300, about
2,500. Items: about 29 tokens on Qwen and 26 on K2.6 (`items_s2.jsonl` `tokens`). So a Qwen pilot full-battery
group is about 256 x 29 + 2,500 = 9,924 tokens, a Qwen gate-item group 100 x 29 + 2,500 = 5,400, a Qwen
confirmatory group 216 x 29 + 2,500 = 8,764, a K2.6 gate-item group 100 x 26 + 1,865 = 4,465 and a K2.6
confirmatory group 216 x 26 + 1,865 = 7,481 (v3 used a 2,200-token prefix and 30 tokens an item for both). Throughput: the measured end-to-end rates of Study 1's
confirmatory runs, Qwen 2,523 tok/s (`results/confirm/qwen/run.log`) and K2.6 2,632 tok/s
(`results/confirm/k26/run.log`); v3 used 2,700 (PLAN.md:113) and 2,100. The zero vector runs once per opening,
not at each dose (`runs/s2_run.py`). `s2_battery.py --count` recomputes groups and tokens from the stamped grid
before stamp A; the figures below are hand estimates. Pilot openings: 23 across S2-DEN and S2-CON together (the
pilot wordings of the 20 arguments plus the 3 second pilot wordings of CA06, DA04 and DA09, so 12 in S2-DEN and
11 in S2-CON); 11 control-argument pilot openings for each FT arm. Held-out openings: 50 per arm. Vectors per
opening as in section 7.

| Run | Groups | Tokens | Hours | Cost at $10.49/h |
|---|---|---|---|---|
| Pilot, Qwen (DEN and CON: 23 openings x [22 full-battery vectors x 2 doses + zero], + 6 emotions x 2 doses + the gate-only zero check on gate items; FTD and FTC: 11 openings each, same vectors) | about 2,610 | about 23.3M | about 2.6 | about $27 |
| Pilot, K2.6 (gate items only: 23 openings x [26 vectors x 2 doses + zero], the half dose only if rule 1 needs it) | about 1,220 | about 5.4M | about 0.6 | about $6 |
| Confirmatory, Qwen (4 arms x 50 x 29, + red and ocean in DEN: 50 x 2) | about 5,900 | about 51.7M | about 5.7 | about $60 |
| Confirmatory, K2.6 (2 arms x 50 x 27, + red and ocean in DEN: 50 x 2) | about 2,800 | about 21M | about 2.2 | about $23 |
| Post-confirmatory carry-over and D1.4 bridge, Qwen | about 330 | about 2.9M | about 0.3 | about $3 |
| Qwen frozen replies at the 1,200 cap (step 3, measured: the `assent` stage took 24,582 s on 2026-10-12; draw 0, 129 replies one at a time at 9.8 tok/s, 4.0 hours; draws 1 to 4, 9 batches of up to 64 at about 1,250 s each, 2.8 hours; the v4 estimate was 3.2 hours) | | | 6.8 | about $72 |
| Qwen replies extended to the 2,048 cap in two box rounds (round 1, `extend_qwen0`: 55 capped draw-0 replies one at a time at about 10 tok/s, about 1.3 hours if continued and up to 3 hours if regenerated, plus about 0.3 hours of loading, hashing and the draw-0 check; round 2, `extend_qwen_needed`: only the capped sampled draws of pending wordings, 8 to 170 of the 371, about 0.6 hours with loading) | | | about 3.1 | about $33 |
| Vector, unsteered and Moonlight sessions (steps 1, 6 and 7: weight downloads and hashing, Moonlight test (d), control and mental-process vectors with the gates on Qwen, both merged checkpoints and K2.6, the two merges, the unsteered pass on both models) | | | about 3.5 | about $37 |
| Total at nC = 14 | about 12,860 | about 104M | about 24.8 | about $260 |

The arithmetic. Qwen pilot: per opening and arm, the 22 full-battery vectors (14 controls and 8 mental-process)
at 2 doses plus the zero vector make 45 full-battery groups, and the 6 emotions at 2 doses plus the gate-only zero
check make 13 gate-item groups; over 23 + 2 x 11 = 45 opening-arms that is 2,025 groups at 9,924 tokens and 585 at
5,400, 23.3M tokens, 2.56 hours. K2.6 pilot: 23 x (26 x 2 + 1) = 1,219 gate-item groups (4 emotions, 14 controls,
8 mental-process), 5.4M tokens, 0.57 hours; at the registered dose alone it is 23 x 27 = 621 groups, 0.29 hours.
Qwen confirmatory: 4 x 50 x 29 + 50 x 2 = 5,900 groups at 8,764 tokens, 51.7M tokens, 5.69 hours. K2.6
confirmatory: 2 x 50 x 27 + 50 x 2 = 2,800 groups at 7,481, 20.9M tokens, 2.21 hours. The carry-over and D1.4 row
keeps v3's group count, priced at 8,764 tokens a group: 2.9M tokens, 0.32 hours. Steered runs 11.35 hours. The
frozen-reply row is the measured stage time, 6.83 hours. The extension row (`runs/s2_assent.py extend`): draw 0
is 53 replies after the 2 checked ones, each at most 848 more tokens if continued (about 86 s at the measured 9.8
tok/s) or at most 2,048 if regenerated (about 208 s), so 1.3 to 3.0 hours, taken here as 2.2 (regenerated replies
averaging about 1,500 tokens, since the continuation is expected to fail the check as it did on Qwen3-0.6B); the
check costs 2 x (86 + 208) s, about 0.16 hours; loading and hashing the weights about 0.15 hours, so round 1 is
about 2.5 hours (1.6 to 3.3). Round 2's size depends on round 1's grading: from the 1,200-cap verdicts, 3
wordings with 8 capped sampled draws would be pending if every extended draw 0 were clean, and 50 wordings with
170 draws if none were; generate runs every row of a batch until its last row stops, at most 848 steps, and the
original batches cost about 0.016 s per row and step (1,250 s for 64 rows x 1,200 steps; a batch of 4 took 138
s), so at 0.016 to 0.024 s per row-step (rows now 2,500 to 3,350 tokens wide) round 2 is 0.1 to 1.1 hours plus
0.15 hours of loading, taken as about 0.6. Both rounds about 3.1 hours (1.9 to 4.6), against about 4.3 hours for
extending all 371 sampled draws at once. The session row is the non-reply part of the two pre-stamp box sessions in
`scripts/s2_box.sh`'s runbook: about 2 hours of the Qwen session (setup, box file, Moonlight, vectors, merges,
FT-arm vectors; the session was costed at about 17 hours when its replies took about 15 one at a time) and the
whole K2.6 session, about 1.5 hours (K2.6 setup, box file, vectors, both unsteered passes). Total 24.78 hours,
$260.

At the rule-3 maximum (24 sampled nouns, nC = 26) each opening gets 12 more vectors, and the confirmatory rows grow
to 4 x 50 x 41 + 100 = 8,300 groups, 72.7M tokens, 8.0 hours, $84 (Qwen) and 2 x 50 x 39 + 100 = 4,000 groups,
29.9M tokens, 3.2 hours, $33 (K2.6), for a total of about 28.1 hours, $295.

Not in the table: the gate-item runs for nouns added under rule 3, weight staging for the pilot and confirmatory
sessions (`/nvme` is wiped on every stop), reruns, and the DeepSeek-V4-Pro noun screen (API, not GPU). Budget 2x
all hours at nC = 14: about 50 hours, about $520 GPU. Against the owner's $300 cap on Study 2's GPU sessions the
point estimate ($260) leaves about $40, and the nC = 26 case ($295) about $5 (before the 2,048 cap and the
measured reply session: about 35 hours, about $366 budgeted, $183 estimated; the v4 draft before these inputs: about $126 estimated,
$252 budgeted, with the frozen replies and the vector, unsteered and Moonlight sessions outside the total; v3:
about $120 estimated, $240 budgeted; v2: about $220 estimated, $450 budgeted). The budget also covers the nC = 26
case. Prefix KV caching across an opening's vector groups, behind a bit-identity test, would save about 20% of the
steered runs. K2.6 needs a g6e.12xlarge, which has been unavailable at times; the K2.6 rule in section 9 fixes
what that does to alpha.

## 14. Known weak points (for reviewers)

- The readout is steered at the post-subject site, so part of the effect may be a direct push on Yes/No, and the
  speaker route is open. v3 cut the instruction-only and question-start sites, so the own-name subject is the only
  check left on the speaker route, and it is descriptive. Nothing in v3 separates a direct push on Yes/No from a
  push through the question's meaning.
- v3 cut the concept-matched block and the shared and specific emotion vectors. Emotion specificity now rests on
  S2-G1, S2-G2 and S2-G8, and the design cannot say whether a component shared by all emotions (for example
  arousal) carries the effect.
- The mental-process controls are hand-picked, not sampled, and number at most 8, so S2-G8's control population
  is loosely defined and its df are capped at 7 or less. A failing one is not replaced.
- The control population is neutral, low-arousal, concrete, frequent nouns from one pair of norm sets. S2-H1
  generalises to that band plus arithmetic and random, not to all non-emotion directions. Low-arousal nouns may
  give weaker directions than emotions; potency is reported as a moderator. The LLM screen and the norms' noun
  tagging are imperfect, so a sampled noun can still read as affective; gates (a) and (b) are the backstop.
- K2.6's pilot is a gate run only. Its control count and power come from Qwen's pilot, so K2.6's power figure is
  borrowed, not measured.
- "Language models" includes the model, and the closing ask makes that explicit, so a self signal that generalises
  to the class cancels. "Other language models" is descriptive for S2-H1 but required for the null reading.
- Arithmetic and random sit in the primary pool; S2-G1 is the first step of Family 1.
- Emotions are fixed effects: with 4 to 6 emotions, S2-H1 generalises over the sampled control nouns, not over emotions.
- S2-DEN openings carry emotion vocabulary and S2-CON openings do not; S2-G5 is labelled confounded.
- Overlap asymmetries kept: the arithmetic control overlaps the counting arguments CA01, CA03 and CA07 (owner
  decision 3 keeps arithmetic), which biases S2-G4 and S2-G5; leave-one-vector-out for arithmetic is reported
  beside them. Among the mental-process controls, route planning overlaps CA04, digit memorising overlaps CA02 and
  mental rotation overlaps CA09 (left-of and right-of chains). S2-G8 uses S2-DEN data only, so these three touch
  only S2-CON readings of the mental-process vectors, which are descriptive.
- S2-R1 removes a depth artifact only to the extent that rocks and humans capacity items span the same depths.
- All arguments come from two LLM generators; the claim generalises over wordings of these 10 arguments, not over
  arguments.
- The Claude Opus 5.5 seed drafts and wordings come from Claude Code subagents, so their sampling temperature and
  system prompt are Claude Code's, are not set or known by the study, and may change between Claude Code releases.
  A draw cannot be reproduced from the stamped prompt alone; the saved text with its agent type, model ID, date
  and tool-use count is the record.
- v4 removed the conceded placebos and S2-G3, so nothing tests whether emotion injection reverses the conceded
  stance on non-emotion premises; S2-G2's non-conceded placebos cover only a generic subject shift. The
  known-answer items run only in the pilot, so the confirmatory run checks comprehension only through Yes+No mass
  and the rocks and humans items.
- Clean-assent selection defines the population: openings this model flatly accepted. Clean assent to a contested
  control claim may be agreeableness rather than belief.
- Control stories run longer than neutral after 3 regeneration rounds (mental-process 5.5 to 8.7%, nouns up to
  5.9%, with only 24 of the 36 nouns N01 to N36 within 5% of the neutral stories' mean length; section 3), so part
  of a control vector may be a length direction; the gap is reported per pattern.

## 15. ID crosswalk

| Study 2 | Meaning | Study 1 relation |
|---|---|---|
| DA01..DA10 | denial arguments | none; D1.4 (Study 1's roughly 1,100-word denial argument) is a post-confirmatory bridge only |
| CA01..CA10 | control arguments | CA01 regenerated from D2c's topic |
| N01..N60 | ordered sampled-noun list; the first 12 that pass the gates are primary controls | none |
| red, ocean | Study 1 concept controls, S2-DEN descriptive bridge only, ungated | Study 1's registered vectors |
| S2-DEN, S2-CON | base-model arms | none |
| S2-FTD, S2-FTC | merged D2 and D2c adapters | Study 1 arms D2, D2c |
| S2-H1, S2-R1, S2-G1, S2-G2, S2-G4..G8, S2-H8 | tests (S2-G3 retired in v4) | unrelated to Study 1 H1..H8 |

## 16. Owner decisions and resolutions

Decisions 1 to 3 date from the design drafts. The rest, and the resolutions of the open owner questions that v2
cited (questions 2, 3, 5 and 8), were made on 2026-10-11: 4 to 9 for v3, 10 to 13 for v4.
1. Steering stays on the question through the readout: the post-subject site, now the only site (section 6).
2. Plural "language models" stays the primary comparator, and a null reading also requires equivalence on "other
   language models" (S2-H8, interpretation table). This resolves question 2; the written reason is in
   `PREREG-s2.md` section 1.
3. Arithmetic and random stay in the primary pool, with vector as a random effect for the controls.
4. Control concepts are a seeded random sample of neutral nouns from the Warriner and Brysbaert norms (section 3).
   Red and ocean leave the primary pool and run only as the ungated S2-DEN bridge. This makes question 5 (ocean's
   replacement) moot.
5. Rule 3 may extend the pool to at most 24 sampled nouns (26 primary controls). This resolves question 3.
6. Capacity frame primary, state frame secondary.
7. Emotions are fixed effects and controls random. This resolves question 8.
8. Scope trimmed to roughly half the v2 GPU cost (section 13; changes listed below).
9. Study 1's self-minus-others values go into the registration, ahead of Study 2's design. Contrast: self items
   minus the rock, thermostat and another-AI items, under prompted denial, emotion vectors minus red and ocean,
   over 40 tasks, with t-intervals. Raw: Qwen +0.12 [+0.09, +0.15], Kimi +0.14 [+0.11, +0.16].
   Compression-corrected: Qwen -0.01 [-0.03, +0.02], Kimi +0.11 [+0.09, +0.14]. These are the reference values
   for the S2-DEN red and ocean bridge (section 9).
10. Argument writer (2026-10-11). The hidden seed drafts and wordings 0, 2, 4, 6 are written by Claude Opus 5.5
    (claude-opus-5-5) run as Claude Code subagents, not through the Anthropic API, so there is no API cost. Each
    subagent reads one rendered prompt file holding only the stored generator prompt and writes its text to a
    given path, with exactly one Read and one Write and no other tool call. The main session logs the prompt path
    and hash, the agent type, the model ID, the date and the tool-use count; a draw with any other tool call is
    discarded as a failed draw (sections 4 and 12). Sampling temperature
    and the system prompt cannot be set, which is disclosed (section 11). The gpt-5.6-sol wordings 1, 3, 5 are
    unchanged.
11. Battery cut (2026-10-11). The conceded placebos (kind `s2_conceded`, 8 per group) are removed, and with them
    S2-G3. The 40 known-answer items run only in the pilot, for the rule 1 gates, and the sensitivity fit that
    added them to the reference set is removed. Pilot battery 256 items, confirmatory battery 216, gate items 100.
12. No second judge on assent grading (2026-10-11): DeepSeek-V4-Pro alone, as in Study 1. A GPT second opinion
    on the roughly 280 replies was considered and declined.
13. Mental-process controls (2026-10-11): 8 instead of 3, with the registered test S2-G8 as the third step of
    Family 1 (section 9). Kept from v3: visual search, route planning, digit memorising. Added and confirmed by
    the owner (2026-10-11): estimating distances by eye, mentally rotating a shape, sorting a pile of objects into
    categories, weighing two options before choosing, recalling where things are in a familiar room. Sorting
    replaced the proposed "picking out one voice in a noisy room", which the eight-emotion judge (J1B) labelled
    calm on 2 of 8 pilot stories. Same story pipeline and gates (a) to (c) as the sampled nouns; a failing one is
    dropped and reported, with no replacement.

## Changes from v3 to v4

- Argument writer (decision 10): the Claude Opus 5.5 seed drafts and wordings 0, 2, 4, 6 come from Claude Code
  subagents (agent type `opus55-high`) that read one rendered prompt file and write their text to a given path,
  exactly one Read and one Write; the main session logs prompt hash, agent type, model ID, date and tool-use
  count, and a draw with any other tool call is a failed draw. "Temperature 1" is removed from section 4, since neither temperature nor system
  prompt can be set; this is disclosed (section 11) and listed as a weak point. Section 12 step 2 now ingests the
  subagent records. gpt-5.6-sol wordings are unchanged.
- Battery (decision 11): the conceded placebos (kind `s2_conceded`) and S2-G3 are removed. The 40 known-answer
  items run only in the pilot, for the rule 1 gates; their sensitivity fit is removed. Pilot battery 256 items,
  confirmatory 216; gate items still 100. A weak point records what the cut gives up.
- Assent grading (decision 12): DeepSeek-V4-Pro alone, as in Study 1; a GPT second opinion was declined.
- Seed drafts (a gap in v3): checked on the prompt's fixed constraints and checks 1, 3 and 5 before any wording
  is drawn, with the same 3-redraw limit; an argument whose seed draft still fails is dropped (section 4).
- Mental-process controls (decision 13): 3 become 8, the five new ones confirmed by the owner (one, sorting,
  replacing the proposed noisy-room voice task), with no replacement for one that fails the gates. The descriptive mental-process contrast becomes S2-G8, the third step
  of Family 1, with its own decision-table row and power simulation; mental rotation's overlap with CA09 is added
  to the weak points. Vectors per opening: Qwen 29, K2.6 27 (31 and 29 in S2-DEN).
- Compute: about 12.4 GPU hours, about $130 at nC = 14 (about $167 at nC = 26); budget 2x, about 25 hours and
  $260. Section 13 now says the 25 pilot openings are across S2-DEN and S2-CON together (v3's text said 25 per
  arm, but its table already used 25 in all), shows the arithmetic, and names the GPU work outside the total.
  Section 12 marks the GPU steps.
- Frozen replies (owner decision of 2026-10-11): draw rule of section 4, up to 4 sampled draws after a non-clean
  zero-temperature draw 0, first clean draw kept; disclosed in section 11, with a draw-0-only sensitivity analysis.
- Qwen sampled draws (owner decision of 2026-10-11): generated in fixed left-padded batches with one seed per batch
  instead of one per draw, to cut the box time; draw 0 and the canary unchanged (section 4).
- Wording assignment seed (a gap in v3): the stamp A manifest lists the held-out openings, so it cannot seed their
  draw; the seed is now the openings manifest's SHA-256 (section 4), fixed before any model reply existed.
- Choices made while building, written in where they belong (the code is the record): noun sources (a third,
  part-of-speech file), screen categories, plural rule, fixed draw seed, and a match over all 853 opening files
  and every item-file field, broader than v3 said (section 3); gpt-5.6-sol and judge settings for wordings
  (section 4) and stories (section 3); exact tokenising of check 1, template stems, the later draw failing in a
  too-similar pair, the judge's fixed A to J order, format checks on every wording, and content-word (not
  3-gram) closing-ask overlap for the copying route (section 4); retry, 429, no-attention-mask, canary and
  200-token prefix details, the cannot-introspect heuristic and the emotion-word list (section 4); the
  held-out split's mechanics and replacement rules, 3 real second pilots instead of 5, and a timing disclosure
  (section 4); story settings, word lists, the neutral check through norms, gate (b) as a rate with unanswered
  calls counted as emotion labels, prompt changes for sorting and weighing two options, roller and platter
  failing gate (b), and the length rule relaxed to "3 rounds, then report the gap" (sections 3 and 14);
  the question file's extra row kinds, per-tokenizer steered start, six reworded known-answer items, the
  contested and persona rows, and the coefficient check (sections 2 and 5); placebo sign by "property present",
  with accuracy read from the stored correct answer (section 9).
- Registration decisions (owner, 2026-10-11), written in where they belong and into `PREREG-s2.md`: every gated
  follow-up opens only after a significant positive estimate, S2-G1 after S2-H1 included; S2-G6 at 0.05 then
  S2-G7 at 0.05 only after a significant positive S2-G6, with S2-G5's model and the centred gap as its only
  covariate; standard Holm over the two models whatever the order of computation, a model dropped before any
  result is read leaving the other at 0.05 (section 9); the permutation check governs every test fitted with the
  S2-H1 model; section 16 item 10 corrected to one Read and one Write; placebo and reference predicates one shared
  set, with "spread the y0 range" defined as `runs/s2_select.py` computes it (section 5); wording SD in the power
  simulation the larger of the pilot estimate and `--prior-sd` (section 10); a technically failed steered trial
  rerun once, then reported and dropped (section 7); pilot rules read base-model rows only, mental-process
  controls never move the dose, rule 2 coverage per pilot opening and fixed at stamp A with the surviving
  arguments (section 8); stamp A and stamp B contents, manifests and fingerprints, twin authorship, ethics and
  release (section 11); and 23 pilot openings, not 25 (sections 8, 10 and 13), which brings the total to about
  12.0 GPU hours and $126 at nC = 14 ($163 at nC = 26; budget about 24 hours, $252), superseding the compute
  figures in the bullet above.
- Runner choices written in where they belong (the code is the record): the sign check per pilot opening, with
  placebo scores and the reference spread on opening means, and a failing persona pair only listed in the
  selection file, to be removed by hand before stamp A (section 5); paraphrases stored 0 to 7 in the item file and
  written as 1 to 8 in readouts (section 5); emotion rows on gate items with gate items ordered first and a
  gate-only zero check group per opening in the Qwen pilot (section 6); the checkpoint hash over config, index,
  tokenizer and every weight shard, and base arms refused unless the weights match the pinned revision's box file
  (sections 7 and 12); all 8 mental-process controls in the pilot, gate survivors in the confirmatory run, zero
  vector once per opening (section 8); the held-out hashes (argument, reply, and the two joined by a blank line)
  and the stamp B manifest listing every hash the guard refuses (section 11); and section 13 recomputed with
  measured prefixes (about 1,850 to 1,885 tokens), item lengths (26 to 29 tokens) and Study 1's measured
  throughputs (Qwen 2,523, K2.6 2,632 tok/s), the zero vector once, and the frozen replies and the vector,
  unsteered and Moonlight sessions now in the total: about 17.5 GPU hours and $183 at nC = 14 ($216 at nC = 26;
  budget about 35 hours, $366), superseding the compute figures above.
- Story and test records brought up to date (section 3, 6, 12 and 14): stories exist for N01 to N36 before stamp
  A so rule 3's additions need no new story round; seven of the 36 fail gate (b) (N02, N09, N27, N28, N30, N32,
  N33), the N25 to N36 pass's 34 failed judge calls re-asked once; 24 of 36 nouns within 5% of neutral length;
  the fine-tuned diagnostic as `runs/s2_vectors.py ftdiag`; the Moonlight test's weights and output paths and
  `tests/s2_run_check.py --moonlight` for the K2.6-style runner path.
- Reply cap raised from 1,200 to 2,048 new tokens for both models (owner decision of 2026-10-12, section 4,
  "Reply cap"), after the 1,200 cap was seen to exclude most of Qwen's denial replies (37 of 67 draw-0 denial
  replies, 371 of 516 sampled draws; Qwen kept 6 of 10 denial arguments) and before stamp A and any steered
  trial: capped Qwen replies are continued in two rounds (round 1 draw 0, greedy and unbatched, checked against
  fresh runs and regenerated if that fails; round 2 only the capped sampled draws the grading walk reaches, in
  seeded continuation batches), capped K2.6 replies regenerated whole
  (Foundry cannot continue an assistant message), originals archived, and grading redone in a 2,048-cap epoch that
  reuses only verdicts of byte-identical replies. Section 13 adds the extension rounds (about 3.1 hours), a
  longer Qwen prefix (about 2,500 tokens, assumed) and the measured reply session (6.8 hours, not 3.2): about 25
  GPU hours and $260 at nC = 14, close to the $300 cap.
- Moderators defined (section 9, exploratory): potency as the mean per-token NLL increase on 48 neutral train
  stories at both doses, measured in box session 2; valence as positive (joy, euphoria, calm) minus negative
  (sadness, anger, fear) per-emotion X, surprise excluded. No registered test changes.

## Changes from v2 to v3 (history; v4 supersedes these where they conflict)

- Controls: the hand-picked candidate list (and its "removed in v1" note) is replaced by a seeded random sample of
  nouns from the Warriner (2013) and Brysbaert (2014) norms under a registered band, a stamped DeepSeek-V4-Pro
  screen and a mechanical appearance check, drawn as an ordered list of 60 with seed 20261011 and stamped in a new
  stamp N before any vector is built. Primary pool: arithmetic, random and 12 sampled nouns (nC = 14, up to 26).
  Red and ocean leave the pool and run only as an ungated S2-DEN bridge to Study 1. Gates now apply to sampled
  nouns and mental-process controls; arithmetic's J1B rate is reported, not gated.
- Counts that follow from nC: vectors per opening (Qwen 24, K2.6 22, +2 in S2-DEN), Satterthwaite cap 13,
  permutation counts C(20, 6) and C(18, 4), S2-G1 on 12 to 24 nouns. Rule 1's minimum primary-control count rises
  from 10 (of 12) to 12 (of 14), keeping the same allowance of two dropped controls. Rule 4's default is 14
  controls.
- Scope cut: the instruction-only and question-start sites, the shared and specific emotion vectors, and the
  concept-matched block and its items (kind `s2_concept`) are removed. The section 5 token assert no longer runs
  on control nouns, since they no longer enter items.
- Tests dropped or adapted because of the cut: the descriptive site prediction (speaker route vs self-model by
  site) is dropped; the concept-matched emotion-specificity criterion ("emotion specificity is claimed only if
  this is > 0") is dropped, so specificity rests on S2-G1, S2-G2 and the mental-process contrast; the shared vs
  specific descriptive split is dropped, and the "inner-state direction" wording rule now depends on the
  mental-process controls alone. A Study 1 bridge contrast (red and ocean as controls) is added to the
  descriptive tier. Primary and gated families (S2-H1, S2-R1, S2-G1 to S2-G7, S2-H8) are unchanged apart from nC.
- Pilot: Qwen runs the full pilot at the one site. K2.6 runs only the rule 1 gate run, all its vectors on the 100
  gate items, the half dose only if needed; its rule 3 count follows Qwen's, and its power is simulated with
  Qwen's pilot inputs.
- Compute re-estimated with the v2 method: about 11.5 GPU hours, about $120 at nC = 14 (about $164 at nC = 26);
  budget 2x, about $240 (v2: about $220 estimated, $450 budgeted).
- The predicted raw compression bias B is corrected to mean (a_E - a_C) x gap; v2 had the sign reversed.
- Readability: plain-language summary, a worked example of the compression correction (section 9), D1.4
  explained at first mention, and owner decisions collected in section 16 with Study 1's values for the
  registration.
- Stamps: stamp N added; stamp A now holds stamp N's .ots file and the per-noun gate outputs.

## Changes from v0 to v2 (history; v3 and v4 supersede these where they conflict)

- Primary measure fixed as compression-corrected d, with a reference set of subject-bearing rocks and humans
  items that share the post-subject span; known-answer items are gate-only plus one sensitivity fit (v2).
  Estimator, y0 source and out-of-range rule fixed.
- Steering site: post-subject span (same tokens for every subject, subject token unsteered), explicit span for
  subjectless items (v2); instruction-only and question-start sites as registered descriptive checks; engine gets
  per-suffix spans, tested in both engines.
- Battery: capacity frame primary, state frame secondary, no pooling; 8 paraphrases per frame with written twins;
  subjects add "other language models" and own name; non-conceded and conceded placebos; concept-matched block;
  32 reference items and a separate persona item (v2); state twins 2, 3 and 7 rewritten for length and word order
  (v2); asserts for length, identical post-subject tokens and emotion word in span, on both tokenizers (v2);
  placebo selection matches level and gap on pilot openings (v2).
- Vectors: 8 drier controls under the J1B and AIPsy gates (ocean expected to be replaced); AIPsy gate redefined as
  max AUROC over the registered labels with explicit mapping, random and Study 1 controls exempt (v2); concept
  nouns at most 2 words (v2); 3 mental-process controls as their own class; shared and specific vectors over the
  registered set (v2); separate build path with match checks.
- Arguments: D1.4 out of the population (fresh DA01) and moved after the confirmatory run (v2); 5 control
  arguments replaced; symmetric closing ask; sibling wordings from a seed draft, two generators, pairwise caps,
  blind main-line check with confusion matrix, mention band, redraw limit with empty slots (v2); seeded pilot draw;
  all replies and verdicts before stamp A on a stamped box with a canary fallback (v2); control-arm attrition no
  longer cancels S2-H1 (v2).
- Tests: S2-H1 with fixed emotion means and random control vectors (v2), Satterthwaite df with an nC - 1 cap and
  KR as a timed check (v2), studentized approximate permutation check (v2); depth check S2-R1 (v2); two families
  (G1, G2 interpreting S2-H1; G4, G5 for the arms, not gated on S2-H1) with G3 as a CI (v2); Qwen alpha fixed
  by K2.6 status (v2); S2-H8 on both comparators (v2); claim sentence names arithmetic and random and a full
  decision table (v2); SESOI reading.
- Pilot: emotion vectors run only on gate items, so no x is ever computed before stamp B (v2); tracking from rocks
  and humans (v2); power inputs from the control class (v2); added controls gated before stamp B (v2); dose and
  counts only; per-vector gates with minimum class sizes; counts lever is controls, not wordings.
- Registration: two OTS stamps; stamp B changes beyond rule outputs make the affected tests exploratory (v2);
  hashed-opening guard in both runners with no exemptions (v2); own preregistration, deviations log, disclosure of
  the earlier pilot result; IDs renamed.
- Readout: fp32 two-row logits, mass and top-5 stored, y0 from identical shapes.
- Compute re-estimated for the v2 battery (Qwen confirmatory about 8,100 groups) and 2x budget, about $450.
