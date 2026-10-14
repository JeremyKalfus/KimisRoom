"""Append one Claude Code subagent draw to data/s2/openings/claude_log.jsonl (drafts/study2-spec.md section 4).

    python runs/s2_claude_log.py <ID> <slot> <draw> <tool_uses> [<ID> <slot> <draw> <tool_uses> ...]

slot is seed, w0, w2, w4 or w6; tool_uses is the count Claude Code reported for that subagent.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/s2/openings/claude_log.jsonl"
args = sys.argv[1:]
assert args and len(args) % 4 == 0, __doc__
with LOG.open("a") as f:
    for i in range(0, len(args), 4):
        aid, slot, draw, uses = args[i], args[i + 1], int(args[i + 2]), int(args[i + 3])
        prompt = ROOT / f"prompts/s2/openings/rendered/{'seed' if slot == 'seed' else 'wording'}_{aid}.md"
        out = ROOT / f"data/s2/openings/{aid}/{slot}_d{draw}.md"
        rec = {"id": aid, "slot": slot, "draw": draw, "path": str(out.relative_to(ROOT)),
               "prompt_path": str(prompt.relative_to(ROOT)),
               "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
               "text_sha256": hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None,
               "agent_type": "opus55-high", "model": "claude-opus-5-5",
               "date": datetime.date.today().isoformat(), "tool_uses": uses}
        f.write(json.dumps(rec) + "\n")
        print(aid, slot, draw, "tool_uses", uses, "text" if out.exists() else "NO TEXT")
