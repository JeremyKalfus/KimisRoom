"""Checks for the registered decision rules in runs/s2_confirm.R and the wording-SD floor in runs/s2_power.R.

    .venv/bin/python tests/s2_rules_check.py [--no-power] [--work DIR]

1. S2-H1 alpha (PREREG-s2.md section 1): standard Holm over the pair whatever the order of computation; a model
   dropped before any result is read (or descriptive under rule 1) leaves the other at 0.05; a pair model with no
   result counts as p = 1. Holm over the models whose step ran (S2-G4, S2-G5). `--k26 qwen-first` is refused.
2. Gating: a follow-up test opens only after a significant positive estimate.
3. Power (runs/s2_power.R): --prior-sd is required with a pilot table; the wording SD used is the larger of the
   pilot estimate and --prior-sd; floor-only mode still works.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FAILS = []


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


R_LOGIC = r"""
suppressMessages(library(lmerTest))
for (e in head(parse(file.path("%s", "runs/s2_confirm.R")), -1)) eval(e, globalenv())
a <- function(p, ...) { x <- h1_alphas(p, ...); paste(names(x), format(x), sep = "=", collapse = ",") }
cat("holm_k26_smaller", a(c(qwen = 0.03, k26 = 0.01)), "\n")
cat("holm_qwen_smaller", a(c(qwen = 0.01, k26 = 0.03)), "\n")
cat("holm_order_free", a(c(k26 = 0.03, qwen = 0.01)), "\n")
cat("holm_first_fails", a(c(qwen = 0.03, k26 = 0.04)), "\n")
cat("holm_missing_k26", a(c(qwen = 0.03)), "\n")
cat("holm_nofit_qwen", a(c(qwen = NA, k26 = 0.02)), "\n")
cat("holm_dropped_k26", a(c(qwen = 0.03), dropped = "k26"), "\n")
cat("holm_descriptive_k26", a(c(qwen = 0.03, k26 = 0.2), descriptive = "k26"), "\n")
cat("holm_dropped_qwen", a(c(k26 = 0.03), dropped = "qwen"), "\n")
r <- holm_ran(c(NA, 0.03), c("no data", "maximal"), c("qwen", "k26")); cat("ran_nodata", format(r[["k26"]]), "\n")
r <- holm_ran(c(NA, 0.03), c("no fit", "maximal"), c("qwen", "k26")); cat("ran_nofit", format(r[["k26"]]), "\n")
cat("opens", opens_next(TRUE, 0.2), opens_next(TRUE, -0.2), opens_next(FALSE, 0.2), opens_next(NA, 0.2), opens_next(TRUE, NA), "\n")
"""


def r_logic(work: Path) -> None:
    script = work / "logic.R"
    script.write_text(R_LOGIC % ROOT)
    r = subprocess.run(["Rscript", str(script)], capture_output=True, text=True)
    ok(r.returncode == 0, "decision-rule helpers ran" + ("" if r.returncode == 0 else ": " + r.stderr[-1500:]))
    got = {ln.split()[0]: " ".join(ln.split()[1:]) for ln in r.stdout.splitlines() if ln.strip()}
    want = {
        "holm_k26_smaller": "qwen=0.050,k26=0.025",    # smaller p (K2.6) at 0.025, passes; Qwen then at 0.05
        "holm_qwen_smaller": "qwen=0.025,k26=0.050",
        "holm_order_free": "k26=0.050,qwen=0.025",     # same alphas when the models come in the other order
        "holm_first_fails": "qwen=0.025,k26=0.025",    # smaller p fails at 0.025: the other is not tested at 0.05
        "holm_missing_k26": "qwen=0.025",              # K2.6 in the pair without a result counts as p = 1
        "holm_nofit_qwen": "qwen=0.050,k26=0.025",     # no fit counts as p = 1 (its 0.05 is never met)
        "holm_dropped_k26": "qwen=0.05",               # dropped before any result was read
        "holm_descriptive_k26": "qwen=0.05,k26=NA",    # left the pair under rule 1
        "holm_dropped_qwen": "k26=0.05",
        "ran_nodata": "0.05",                          # a step that did not run leaves the pair
        "ran_nofit": "0.025",                          # a step that ran without a fit counts as p = 1
        "opens": "TRUE FALSE FALSE FALSE FALSE",
    }
    for k, v in want.items():
        g = got.get(k, "missing").replace(" ", "")
        ok(g == v.replace(" ", ""), f"{k}: {got.get(k)} (want {v})")
    r = subprocess.run(["Rscript", str(ROOT / "runs/s2_confirm.R"), str(work / "qf"), str(work / "none.csv"), "--k26", "qwen-first"],
                       capture_output=True, text=True)
    ok(r.returncode != 0 and "qwen-first was removed" in r.stderr, "--k26 qwen-first is refused")


def read_inputs(path: Path) -> dict:
    return {(x["source"], x["component"]): x for x in csv.DictReader(path.open())}


def power(work: Path) -> None:
    pil = work / "pilot.csv"
    subprocess.run([PY, str(ROOT / "tests/s2_synth.py"), str(pil), "--phase", "pilot", "--models", "qwen",
                    "--n-args", "4", "--n-wordings", "2", "--seed", "7"], check=True, capture_output=True)
    tab = work / "pilot_table.csv"
    subprocess.run([PY, str(ROOT / "runs/s2_table.py"), "--phase", "pilot", "--out", str(tab), str(pil)],
                   check=True, capture_output=True)
    base = ["Rscript", str(ROOT / "runs/s2_power.R")]
    quick = ["--reps", "2", "--nc", "14", "--design", "7x3", "--perm-null", "2", "--cores", "1", "--quick"]
    r = subprocess.run(base + [str(work / "p0"), str(tab)] + quick, capture_output=True, text=True)
    ok(r.returncode != 0 and "--prior-sd is required" in r.stderr, "power refuses a pilot table without --prior-sd")
    vals = {}
    for tag, prior in (("big", "5"), ("zero", "0")):
        out = work / f"p_{tag}"
        r = subprocess.run(base + [str(out), str(tab), "--prior-sd", prior] + quick, capture_output=True, text=True)
        ok(r.returncode == 0, f"power ran with --prior-sd {prior}" + ("" if r.returncode == 0 else ": " + r.stderr[-1500:]))
        if r.returncode:
            return
        vals[tag] = read_inputs(out / "inputs.csv")
    est = float(vals["zero"][("primary controls", "aw")]["point"])
    used_big = float(vals["big"][("primary controls", "aw_used")]["point"])
    used_zero = float(vals["zero"][("primary controls", "aw_used")]["point"])
    print(f"     pilot wording SD {est:.4f}; used with prior 5: {used_big}; with prior 0: {used_zero:.4f}")
    ok(used_big == 5.0 and abs(used_zero - est) < 1e-12 and est < 5,
       "wording SD used = max(pilot estimate, --prior-sd), pilot estimate reported")
    r = subprocess.run(base + [str(work / "p_floor"), "--prior-sd", "0.08", "--nc", "14"], capture_output=True, text=True)
    ok(r.returncode == 0 and (work / "p_floor/floor.csv").exists(), "floor-only mode (no pilot table) still runs")


def main() -> None:
    work = Path(sys.argv[sys.argv.index("--work") + 1]) if "--work" in sys.argv else Path(tempfile.mkdtemp(prefix="s2rules"))
    work.mkdir(parents=True, exist_ok=True)
    r_logic(work)
    if "--no-power" not in sys.argv:
        power(work)
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
