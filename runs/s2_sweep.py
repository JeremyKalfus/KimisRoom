"""Study 2 exploratory dose sweep, laptop summary (post-study, NOT registered; drafts/s2-dose-sweep.md).

    .venv/bin/python runs/s2_sweep.py [--sweep results/s2/qwen/sweep] [--out results/s2/qwen/sweep/summary.csv]

Inputs (scripts/s2_box.sh sweep_qwen, then scripts/s2_fetch.sh):
  SWEEP/S2-DEN/readouts.csv       the confirmatory S2-DEN design at the sweep doses (18 vectors, --test --unit 47.99)
  SWEEP/gate/{S2-DEN,S2-CON}/     rule 1's gate-item run for the same vectors and doses (the nouns stages' path)
  --confirm (results/s2/qwen/confirm/S2-DEN)   dose 1: the confirmatory readouts, the same 18 vectors taken from them
  --pilot-gates (results/s2/pilot/qwen_gates.csv)   dose 1's rule 1 statistics (the pilot's gate rows, both arms)
  --pilot (results/s2/qwen/pilot)  the pilot's dose 2 readouts, to check the sweep's gate run against them

Measures go through runs/s2_table.py's own code (frozen in stamp A; imported, never edited): T.load, T.check,
T.groups_of and T.group_measures for d, T.gates for rule 1's statistics. T.check accepts only the registered dose
(2) and its half, so each sweep dose is put through it alone with its steered rows relabelled to the registered
dose; the true dose is kept here. Overlap measures are left out (T.OVERLAP None), as runs/s2_carry.py does.

Per dose (summary.csv, one row per dose; at dose 1 a second row with all 26 confirmatory primary controls):
  per vector, d = mean over its openings of the mean of d over the 8 capacity paraphrases (the S2-H1 measure);
  X = mean d over the emotions minus mean d over the primary controls (arithmetic, random, the sampled nouns);
  sd_e, sd_c the sample SDs of per-vector d; se = sqrt(sd_e^2/n_e + sd_c^2/n_c); x_over_se = X / se;
  cap_mass_e, cap_mass_c: mean P(Yes)+P(No) on the capacity you and language-models readouts (table measure mass);
  rule 1's gate statistics over the 18 vectors (ka_acc, gate_mass, track_rocks, track_humans: mean and min), the
  number that pass the per-vector gates (ka >= 0.90, mass >= 0.90, rocks and humans tracking >= 0.95), all_pass
  (every vector passes: what rule 1 asks of each vector at the dose it settles on) and rule1_strict (all_pass and
  mean known-answer accuracy >= 0.95: the condition rule 1 puts to the registered dose before falling back to half;
  the confirmatory dose 1 itself does not meet it, mean 0.935 in the pilot).
Also SWEEP/vectors.csv (per dose and vector: d, its SD over openings, mass, gate statistics) and SWEEP/checks.json:
the sweep runs' s2_meta.json against the confirmatory run's (unit, layer, items, vector files, openings, weights),
the unsteered y0 of the sweep against the confirmatory run's on the same openings and items, and the sweep's
dose 2 gate run against the pilot's dose 2 rows (readout by readout and per-vector gate statistics).
Memory: one dose at a time (about 0.2M readouts); the large readout files are streamed into per-dose temporary
files first. Missing sweep files are reported and skipped, so this also runs before the sweep (dose 1 only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import s2_table as T  # noqa: E402  (frozen in stamp A; its measure code only)

MODEL = "qwen"
R = T.CFG[MODEL]["dose"]  # registered dose (2 units): the label T.check accepts
EMO = T.CFG[MODEL]["emo"]
# the sweep's vectors in scripts/s2_box.sh SWEEP_VECS order (the runner keeps --vecs order in s2_meta.json)
SWEEP_VECS = ["anger", "euphoria", "fear", "joy", "sadness", "surprise", "N01", "N03", "N04", "N05", "N06", "N07",
              "N08", "N10", "N11", "N12", "arithmetic", "random"]
assert sorted(EMO) == SWEEP_VECS[:6]
csv.field_size_limit(1 << 24)


def need(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"s2_sweep: {msg}")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def contrast(per_vec: dict[str, tuple[str, float]]) -> dict:
    """X, the vector-level SDs and SE, X/SE from {vec: (vclass, mean d)}."""
    E = [m for c, m in per_vec.values() if c == "emotion"]
    C = [m for c, m in per_vec.values() if c in T.PRIMARY]
    se_ = math.sqrt(sd(E) ** 2 / len(E) + sd(C) ** 2 / len(C)) if len(E) > 1 and len(C) > 1 else float("nan")
    x = mean(E) - mean(C)
    return {"n_emotions": len(E), "n_controls": len(C), "mean_d_emotions": mean(E), "mean_d_controls": mean(C),
            "X": x, "sd_emotions": sd(E), "sd_controls": sd(C), "se": se_,
            "x_over_se": x / se_ if se_ and se_ == se_ else float("nan")}


def split_by_dose(path: Path, tmp: Path, keep, tag: str) -> dict[float, Path]:
    """Stream PATH's readouts into one CSV per steered dose (zero-vector rows go to every file), steered rows
    relabelled to the registered dose so T.check takes them. keep(row) filters rows. Returns {true dose: file}."""
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        zero, files, writers = [], {}, {}
        for r in rd:
            if not keep(r):
                continue
            if r["vclass"] == "none":
                zero.append(r)
                for w in writers.values():
                    w.writerow(r)
                continue
            d = float(r["dose"])
            if d not in writers:
                p = tmp / f"{tag}_{d:g}.csv"
                files[d] = (p, p.open("w", newline=""))
                writers[d] = csv.DictWriter(files[d][1], fieldnames=cols)
                writers[d].writeheader()
                writers[d].writerows(zero)  # the zero rows read so far; later ones are written as they come
            writers[d].writerow(dict(r, dose=repr(R)))
    for p, fh in files.values():
        fh.close()
    return {d: p for d, (p, _) in files.items()}


def dose_measures(path: Path, phase: str = "confirm") -> tuple[dict, dict, dict]:
    """Through runs/s2_table.py's check and measures: {vec: (vclass, [per-opening mean d])}, {vec: [per-opening
    capacity mass]}, and the zero-vector y0 {(opening, item): y0}."""
    rows = T.load([path])
    T.check(rows, phase)
    steered, zero = T.groups_of(rows, phase)
    d, mass = defaultdict(list), defaultdict(list)
    cls = {}
    for key, g in sorted(steered.items()):
        ms, _ = T.group_measures(g, zero[key[:3]])
        dv = [v for p, m, v in ms if m == "d"]
        need(len(dv) == 8, f"{key}: {len(dv)} capacity paraphrases with d, expected 8")
        vec = key[3]
        cls[vec] = next(iter(g.values()))["vclass"]
        d[vec].append(mean(dv))
        mass[vec].append(mean([v for p, m, v in ms if m == "mass"]))
    y0 = {(o, it): r["y0"] for (_, _, o), z in zero.items() for it, r in z.items()}
    return {v: (cls[v], d[v]) for v in d}, dict(mass), y0


def gate_stats(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Rule 1's per-vector gate statistics with T.gates, after T.check on each dose relabelled. Returns
    {(vec, dose): stats} and the rows themselves (for the readout comparison)."""
    rows = [r for r in T.load(paths) if r["vec"] in SWEEP_VECS or r["vclass"] == "none"]
    for dose in sorted({r["dose"] for r in rows if r["vclass"] != "none"}):
        rel = [dict(r, dose=R) if r["vclass"] != "none" else dict(r) for r in rows
               if r["vclass"] == "none" or r["dose"] == dose]
        T.check(rel, "pilot")
        T.groups_of(rel, "pilot")
    G, _ = T.gates(rows)
    return {(g["vec"], g["dose"]): g for g in G if g["model"] == MODEL}, rows


def rule1(stats: list[dict]) -> dict:
    """Rule 1 at one dose over the given vectors. all_pass: every vector passes the per-vector gates (what rule 1
    asks of each vector at the dose it settles on); rule1_strict: that and mean known-answer accuracy >= 0.95 (the
    condition rule 1 puts to the registered dose before falling back to half)."""
    out = {"gate_n": len(stats), "gate_n_pass": sum(bool(s["passes"]) for s in stats)}
    for k in ("ka_acc", "gate_mass", "track_rocks", "track_humans"):
        vals = [float(s[k]) for s in stats]
        out[f"{k}_mean"], out[f"{k}_min"] = mean(vals), (min(vals) if vals else float("nan"))
    out["all_pass"] = bool(stats) and out["gate_n_pass"] == len(stats)
    out["rule1_strict"] = out["all_pass"] and out["ka_acc_mean"] >= T.GATE["ka_mean"]
    out["gate_fails"] = " ".join(sorted(s["vec"] for s in stats if not s["passes"]))
    return out


def meta_checks(sweep: dict, conf: dict) -> list[str]:
    """Differences between the sweep's S2-DEN s2_meta.json and the confirmatory one that would break comparability."""
    bad = []
    for k in ("model", "arm", "phase", "layer", "natural_unit", "site", "items_sha256", "emotion_file_sha256",
              "controls_sha256", "pool_sha256", "hf_revision", "ckpt_sha256", "targets"):
        if sweep.get(k) != conf.get(k):
            bad.append(f"{k}: sweep {sweep.get(k)!r} vs confirm {conf.get(k)!r}")
    so = [(o["opening"], o["sha256"]) for o in sweep["openings"]]
    co = [(o["opening"], o["sha256"]) for o in conf["openings"]]
    if so != co:
        bad.append(f"openings differ ({len(so)} vs {len(co)})")
    if not sweep["guard"]["stamp_b"]:
        bad.append("sweep guard did not report stamp B")
    if [v["vec"] for v in sweep["vectors"]] != SWEEP_VECS:
        bad.append(f"sweep vectors {[v['vec'] for v in sweep['vectors']]} != {SWEEP_VECS}")
    if not sweep.get("test"):
        bad.append("sweep s2_meta.json is not marked test (expected: the sweep's doses need --test)")
    return bad


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", type=Path, default=ROOT / "results/s2/qwen/sweep")
    ap.add_argument("--confirm", type=Path, default=ROOT / "results/s2/qwen/confirm/S2-DEN")
    ap.add_argument("--pilot-gates", type=Path, default=ROOT / "results/s2/pilot/qwen_gates.csv")
    ap.add_argument("--pilot", type=Path, default=ROOT / "results/s2/qwen/pilot")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or a.sweep / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.OVERLAP = None
    checks: dict = {}
    per_dose: dict[float, dict] = {}   # dose -> {"d": {vec: (cls, [..])}, "mass": {...}, "source": str}
    ref26 = None
    conf_meta = json.loads((a.confirm / "s2_meta.json").read_text())
    unit = conf_meta["natural_unit"]
    with tempfile.TemporaryDirectory(prefix="s2_sweep_") as td:
        tmp = Path(td)
        # dose 1: the confirmatory readouts, emotions and every primary control (the 26 for the reference row)
        conf_files = split_by_dose(a.confirm / "readouts.csv", tmp,
                                   lambda r: r["vclass"] in ("none", "emotion") + T.PRIMARY, "confirm")
        need(list(conf_files) == [1.0], f"{a.confirm}: steered doses {sorted(conf_files)}, expected [1.0]")
        d1, m1, y0_conf = dose_measures(conf_files[1.0])
        conf_files[1.0].unlink()
        ref26 = contrast({v: (c, mean(x)) for v, (c, x) in d1.items()})
        ref26_mass = {c: mean([mean(m1[v]) for v, (cc, _) in d1.items() if (cc == "emotion") == (c == "e")])
                      for c in ("e", "c")}
        need(set(SWEEP_VECS) <= set(d1), f"{a.confirm}: missing sweep vectors {sorted(set(SWEEP_VECS) - set(d1))}")
        per_dose[1.0] = {"d": {v: d1[v] for v in SWEEP_VECS}, "mass": {v: m1[v] for v in SWEEP_VECS},
                         "source": "confirmatory S2-DEN"}
        print(f"dose 1: {a.confirm}/readouts.csv, {len(d1)} vectors ({len(SWEEP_VECS)} in the sweep set)", flush=True)
        sdir = a.sweep / "S2-DEN"
        if (sdir / "readouts.csv").exists():
            sm = json.loads((sdir / "s2_meta.json").read_text())
            bad = meta_checks(sm, conf_meta)
            checks["sweep_meta_vs_confirm"] = bad or "same unit, layer, items, vector files, openings, weights"
            need(not bad, "sweep run is not comparable with the confirmatory run:\n  " + "\n  ".join(bad))
            files = split_by_dose(sdir / "readouts.csv", tmp, lambda r: True, "sweep")
            need(sorted(files) == sorted(sm["doses"]), f"{sdir}: doses {sorted(files)} vs s2_meta {sm['doses']}")
            y0_diff = None
            for dose in sorted(files):
                dd, mm, y0 = dose_measures(files[dose])
                files[dose].unlink()
                need(sorted(dd) == sorted(SWEEP_VECS), f"{sdir} dose {dose}: vectors {sorted(dd)}")
                per_dose[dose] = {"d": dd, "mass": mm, "source": "sweep S2-DEN"}
                if y0_diff is None:
                    common = set(y0) & set(y0_conf)
                    need(len(common) == len(y0), f"{len(y0) - len(common)} sweep zero readouts not in the confirm run")
                    y0_diff = max(abs(y0[k] - y0_conf[k]) for k in common)
                    checks["y0_sweep_vs_confirm_max_abs"] = y0_diff
                    checks["y0_compared"] = len(common)
                print(f"dose {dose:g}: {sdir}/readouts.csv, {len(dd)} vectors", flush=True)
            if y0_diff is not None and y0_diff > 1e-3:
                print(f"WARNING: unsteered y0 differs from the confirmatory run's by up to {y0_diff:.3g}", flush=True)
        else:
            print(f"{sdir}/readouts.csv missing: sweep doses skipped (dose 1 only)", flush=True)

    # rule 1's gate statistics: dose 1 from the pilot's gates file, the sweep doses from the sweep's gate run
    gates: dict[tuple[str, float], dict] = {}
    gsrc: dict[float, str] = {}
    with a.pilot_gates.open(newline="") as f:
        pilot_g = {(r["vec"], float(r["dose"])): r for r in csv.DictReader(f) if r["model"] == MODEL}
    for k, r in pilot_g.items():
        if k[0] in SWEEP_VECS and k[1] == 1.0:
            gates[k] = dict(r, passes=r["passes"] == "True")
            gsrc[1.0] = f"pilot ({a.pilot_gates.name})"
    gfiles = [a.sweep / "gate" / arm / "readouts.csv" for arm in ("S2-DEN", "S2-CON")]
    if all(p.exists() for p in gfiles):
        gs, grows = gate_stats(gfiles)
        for (v, dose), g in gs.items():
            if dose != 1.0:
                gates[(v, dose)] = g
                gsrc[dose] = "sweep gate run"
        # the sweep's dose 2 against the pilot's dose 2: the same vectors, openings and gate items
        if any(dose == R for (_, dose) in gs):
            cmp = {}
            for c in ("ka_acc", "gate_mass", "track_rocks", "track_humans"):
                cmp[c] = max(abs(float(gs[(v, R)][c]) - float(pilot_g[(v, R)][c])) for v in SWEEP_VECS)
            cmp["passes_agree"] = all(bool(gs[(v, R)]["passes"]) == (pilot_g[(v, R)]["passes"] == "True")
                                      for v in SWEEP_VECS)
            sw = {(r["arm"], r["opening"], r["vec"], r["item"]): r for r in grows
                  if r["vclass"] != "none" and r["dose"] == R}
            n, dy, dm, top = 0, 0.0, 0.0, 0
            for arm in ("S2-DEN", "S2-CON"):
                with (a.pilot / arm / "readouts.csv").open(newline="") as f:
                    for r in csv.DictReader(f):
                        k = (r["arm"], r["opening"], r["vec"], r["item"])
                        if r["vec"] in SWEEP_VECS and float(r["dose"]) == R and k in sw:
                            s = sw[k]
                            n += 1
                            dy = max(dy, abs(float(r["y"]) - s["y"]))
                            dm = max(dm, abs(float(r["mass"]) - s["mass"]))
                            top += r["top1"].strip() != s["top1"]
            need(n == len(sw), f"{len(sw) - n} sweep dose-2 gate readouts have no pilot counterpart")
            cmp.update(readouts=n, y_max_abs=dy, mass_max_abs=dm, top1_differ=top)
            checks["gate_dose2_sweep_vs_pilot"] = cmp
    else:
        print(f"{gfiles[0].parent.parent}: gate readouts missing: breakage measures at dose 1 only", flush=True)

    rows, vrows = [], []
    for dose in sorted(set(per_dose) | set(gsrc)):
        row = {"dose": dose, "coef": round(dose * unit, 4), "control_set": "sweep (10 nouns, arithmetic, random)"}
        p = per_dose.get(dose)
        if p:
            n_op = {len(x) for _, x in p["d"].values()}
            need(len(n_op) == 1, f"dose {dose}: vectors with different opening counts {n_op}")
            c = contrast({v: (cl, mean(x)) for v, (cl, x) in p["d"].items()})
            mass_e = mean([mean(p["mass"][v]) for v, (cl, _) in p["d"].items() if cl == "emotion"])
            mass_c = mean([mean(p["mass"][v]) for v, (cl, _) in p["d"].items() if cl in T.PRIMARY])
            row.update(d_source=p["source"], n_openings=n_op.pop(), **c, cap_mass_e=mass_e, cap_mass_c=mass_c)
        st = [gates[(v, dose)] for v in SWEEP_VECS if (v, dose) in gates]
        if st:
            need(len(st) == len(SWEEP_VECS), f"dose {dose}: gate statistics for {len(st)} of {len(SWEEP_VECS)} vectors")
            row.update(gate_source=gsrc[dose], **rule1(st))
        rows.append(row)
        for v in SWEEP_VECS:
            vr = {"dose": dose, "vec": v}
            if p:
                cl, x = p["d"][v]
                vr.update(vclass=cl, mean_d=mean(x), sd_d_openings=sd(x), n_openings=len(x), cap_mass=mean(p["mass"][v]))
            g = gates.get((v, dose))
            if g:
                vr.update({k: g[k] for k in ("ka_acc", "gate_mass", "track_rocks", "track_humans", "passes")})
            vrows.append(vr)
        if dose == 1.0 and ref26:
            rows.append({"dose": 1.0, "coef": round(unit, 4), "control_set": "confirmatory (all 26 primary controls)",
                         "d_source": "confirmatory S2-DEN", "n_openings": row.get("n_openings"), **ref26,
                         "cap_mass_e": ref26_mass["e"], "cap_mass_c": ref26_mass["c"]})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    vcols = list(dict.fromkeys(k for r in vrows for k in r))
    with (out.parent / "vectors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vcols)
        w.writeheader()
        w.writerows(vrows)
    (out.parent / "checks.json").write_text(json.dumps(checks, indent=1, default=str))

    show = ["dose", "control_set", "n_openings", "X", "sd_emotions", "sd_controls", "se", "x_over_se", "cap_mass_e",
            "cap_mass_c", "ka_acc_mean", "ka_acc_min", "gate_mass_min", "track_rocks_min", "track_humans_min",
            "gate_n_pass", "all_pass", "rule1_strict", "gate_fails"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    table = [[c.replace("control_set", "controls") for c in show]] + [
        [fmt(r.get(c)) if c != "control_set" else ("26" if r[c].startswith("confirmatory") else "12") for c in show]
        for r in rows]
    wid = [max(len(t[i]) for t in table) for i in range(len(show))]
    print("\nEXPLORATORY dose sweep (not registered): emotion minus control d per dose, and rule 1's gates")
    for t in table:
        print("  ".join(s.rjust(wid[i]) for i, s in enumerate(t)))
    print(f"\n{out}: {len(rows)} rows; {out.parent / 'vectors.csv'}: {len(vrows)} rows; checks: "
          f"{json.dumps(checks, default=str)}")


if __name__ == "__main__":
    main()
