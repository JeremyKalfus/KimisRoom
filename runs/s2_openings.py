"""Study 2 openings, everything after the argument draws (drafts/study2-spec.md section 4, section 12
step 2). Logic lives in kimisroom/s2_openings.py; this is the CLI.

    python runs/s2_openings.py check-seed DA01 [DA02 ...|all] [--force]
    python runs/s2_openings.py gpt DA01 [--slot w1 --slot w3] [--redraw] [--prompt FILE] [--dry-run]
    python runs/s2_openings.py check DA01 [DA02 ...|all]
    python runs/s2_openings.py report
    python runs/s2_openings.py assign                     # pilot / held-out draw, after `report`

check-seed  Every seed draft data/s2/openings/<ID>/seed_d<n>.md, in draw order: valid record
            (Claude log line, tool_uses == 2, model claude-opus-5-5, agent opus55-high, prompt hash),
            the prompt's fixed constraints (900-1,100 words, 4-6 numbered steps "1." .. "n.", the closing
            ask verbatim as the last paragraph with the argument's claim) and checks 1, 3 and 5. The
            first passing draw is copied to <ID>/seed.md. Prints the next draw to make, or that the
            argument is dropped after 4 failed draws (the first plus 3 redraws).
gpt         Wordings 1, 3, 5 from gpt-5.6-sol on Foundry with the argument's rendered wording prompt
            (the Claude wordings' logged prompt, else prompts/s2/openings/rendered/wording_<ID>.md, else
            rendered with render.py's own functions; it must contain seed.md). Default: slots with no
            draw yet. --redraw: the gpt slots that the last `check` marked for a redraw. Stores
            <slot>_d<n>.md, the raw request and response in <slot>_d<n>.gpt.json, and a line in
            gpt_log.jsonl with model_version, prompt path and SHA-256, params and finish_reason.
check       Checks 1-5 on every draw of w0..w6 (validity and fixed constraints first), check 2 greedily
            in draw order against the accepted set, then per slot: accepted, needs_redraw (next draw
            number), excluded (4 failed draws) or missing. The accepted draw of each slot is copied to
            <ID>/<slot>.md (what runs/s2_assent.py reads); a stale <slot>.md of a slot no longer
            accepted is removed. Writes <ID>/check.json, appends every draw and its reasons to
            check_log.jsonl.
report      Draw status, failure rates by argument and generator, main-line confusion matrix per kind,
            pairwise overlap matrices, mention counts, each battery item's 3-gram overlap with each
            closing ask, and a SHA-256 manifest of every prompt, text, raw response and log, into
            results/s2/openings/{report.md,report.json,manifest.txt}.
assign      Section 4's seeded pilot / held-out / spare draw over the accepted wordings
            (data/s2/openings/assignment.json). The seed is the first 8 hex digits of the SHA-256 of
            results/s2/openings/manifest.txt (the openings manifest `report` wrote before any model
            reply existed; spec v4, which replaced the circular stamp A seed). It is derived, not
            passed, and the draw refuses to run if a check.json or accepted <slot>.md no longer matches
            its manifest hash. Never rerun `report` after the assignment: assignment.json would then
            enter the manifest and change the seed.

Rules (orchestrator, 2026-10-11):
1. Check 1 compares the draw, closing ask removed, with every row of prompts/battery/items_s2.jsonl
   (placebo, reference and reserve candidates included), each item's full user-turn text with
   " Answer with one word, Yes or No." as in Study 1 (kimisroom/distill.py:176-183): the share of the
   item's 3-grams found in the draw is at most 0.25. Banned phrases: the generator prompts' "word
   sequences to avoid" (read from wording.md) and each item's question with its subject left open.
   It runs on seed drafts and wordings.
2. Check 2 (pairwise 3-gram Jaccard <= 0.15 among an argument's wordings) removes the closing ask
   from both texts.
3. Check 5 counts the whole text including the closing ask: "language model" and "language models"
   >= 6, and you, your, yours, yourself, you're (case-insensitive whole words) at most 5 times that.
4. The 900-1,100 word band counts whitespace-separated tokens of the whole text.
5. A Claude draw is valid only with tool_uses == 2 (one Read of its prompt file, one Write of its text).
Checks 3 (main line) and 4 (step survival, wordings only, against seed.md's numbered steps) are
DeepSeek-V4-Pro at temperature 0; templates, rendered prompts and raw responses are files under
data/s2/openings/judge/, and a stored answer is reused, so reruns cost nothing.

Foundry access goes through kimisroom/foundry.py (the resource key is read with the `az` login at
call time and held in memory only). --root points everything at another tree (the tests use it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom import s2_openings as so  # noqa: E402
from kimisroom.texts import ROOT  # noqa: E402


def ids_of(ctx: so.Ctx, args: list[str]) -> list[str]:
    return sorted(ctx.lines) if args == ["all"] else args


def show_rows(rows: list[dict]) -> None:
    for r in sorted(rows, key=lambda r: (r["slot"], r["draw"])):
        tag = r.get("status", "")
        why = "; ".join(r.get("reasons", [])) or "-"
        print(f"  {r['slot']}_d{r['draw']:<2d} {r['generator']:6s} {tag:8.8s} {why}")


def cmd_check_seed(ctx: so.Ctx, a) -> None:
    for i in ids_of(ctx, a.ids):
        o = so.check_seed(ctx, i, force=a.force)
        print(f"{i} seed: {o['status']}")
        show_rows(o["draws"])
        st = o["slot"]
        if st["status"] == "accepted":
            print(f"  -> {o['seed_md']} ({o['seed_sha256'][:12]})")
        elif st["status"] in ("needs_redraw", "missing"):
            print(f"  -> draw next: data/s2/openings/{i}/seed_d{st['next_draw']}.md "
                  f"from prompts/s2/openings/rendered/seed_{i}.md")
        elif st["status"] == "excluded":
            print(f"  -> {i} DROPPED: {st['n_draws']} seed drafts failed")


def cmd_gpt(ctx: so.Ctx, a) -> None:
    for i in a.ids:
        t = so.gpt_targets(ctx, i, a.slot, a.redraw)
        pp = so.wording_prompt(ctx, i, Path(a.prompt) if a.prompt else None)
        print(f"{i}: prompt {ctx.paths.rel(pp)} ({so.file_sha(pp)[:12]}); draws {t or 'none needed'}")
        if not t or a.dry_run:
            continue
        for r in so.generate_gpt(ctx, i, t, pp):
            w = len((ctx.paths.resolve(r["path"]).read_text() if r.get("text_sha256") else "").split())
            print(f"  {r['slot']}_d{r['draw']}: {r.get('error') or r['finish_reason']}, {w} words, "
                  f"model_version {r.get('model_version')}, usage {(r.get('usage') or {}).get('completion_tokens')}")


def cmd_check(ctx: so.Ctx, a) -> None:
    for i in ids_of(ctx, a.ids):
        o = so.check_wordings(ctx, i)
        print(f"{i}: {o['n_accepted']}/7 accepted {o['accepted']}")
        show_rows(o["draws"])
        for s, st in o["slots"].items():
            if st["status"] in ("needs_redraw", "missing"):
                how = (f"Claude subagent -> data/s2/openings/{i}/{s}_d{st['next_draw']}.md" if s in so.CLAUDE_SLOTS
                       else f"runs/s2_openings.py gpt {i} --redraw")
                print(f"  {s}: {st['status']}, draw d{st['next_draw']} ({how})")
            elif st["status"] == "excluded":
                print(f"  {s}: EXCLUDED after {st['n_draws']} failed draws")
        for w in o["warnings"]:
            print(f"  warning: {w}")


def cmd_report(ctx: so.Ctx, a) -> None:
    o = so.report(ctx, [Path(so.__file__), Path(__file__)])
    d = ctx.paths.report_dir
    print(f"{ctx.paths.rel(d / 'report.md')}, report.json, manifest.txt ({len(o['manifest'])} files)")
    for i, v in o["status"].items():
        print(f"  {i}: seed {v['seed']}, accepted {v['accepted']}")


def cmd_assign(ctx: so.Ctx, a) -> None:
    o = so.assign(ctx)
    out = ctx.paths.data / "assignment.json"
    out.write_text(json.dumps(o, indent=1))
    print(f"{ctx.paths.rel(out)}; seed {o['seed_hex']} ({o['seed_source']}); second pilot wording for "
          f"{o['second_pilot_arguments']}")
    for i, v in o["arguments"].items():
        print(f"  {i} ({v['survivors']} survivors): " + ", ".join(f"{s}={r}" for s, r in sorted(v["roles"].items())))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT), help="tree holding prompts/ and data/ (default: this repo)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check-seed")
    p.add_argument("ids", nargs="+")
    p.add_argument("--force", action="store_true", help="replace an existing, different seed.md")
    p = sub.add_parser("gpt")
    p.add_argument("ids", nargs="+")
    p.add_argument("--slot", action="append", choices=so.GPT_SLOTS)
    p.add_argument("--redraw", action="store_true")
    p.add_argument("--prompt")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("ids", nargs="+")
    sub.add_parser("report")
    sub.add_parser("assign")
    a = ap.parse_args(argv)
    ctx = so.Ctx(so.Paths(Path(a.root)))
    {"check-seed": cmd_check_seed, "gpt": cmd_gpt, "check": cmd_check, "report": cmd_report,
     "assign": cmd_assign}[a.cmd](ctx, a)


if __name__ == "__main__":
    main()
