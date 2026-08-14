"""M46 A0 — the 780+-band-weighted ship gate (docs/M46-plan.md Track A).

The m44 live post-mortem measured WHERE the ELO decision happens: the 780+
band is grim .39 / mirror .23 / wall+lucario+stall+dragapult ~.21, our pooled
wr there is 0.30, and both M44 gates CONTAINED the live verdict at bed level
and averaged it away in the 25-bed pool. This instrument is the registered
answer: the pool is weighted by the measured band mix, and a single family
kill can no longer be overruled by a pooled positive.

Pre-registered decision rule (band spec `bars`, hash-covered):
    PASS  = weighted delta >= bars.pass  AND  no family z <= bars.family_z
            AND the verdict is weight-stable (bootstrap agreement >=
            bootstrap.stability_floor over B multinomial redraws of the n=83
            live-game weights)
    KILL  = weighted delta <= bars.kill  OR  any family z <= bars.family_z
    else INCONCLUSIVE — including "PASS-shaped but WEIGHT-UNSTABLE".
`bars.family_z` must be CALIBRATED before first candidate use: a
control-vs-control null panel (spec flag `null_calibration`) measures the
null spread of the per-family z's (z <= -2 over ~9 families fires ~19% of
the time under a true null — the M26 uncalibrated-kill failure).

Architecture: a WRAPPER around scripts/gate_spec.py, which stays byte-frozen.
One committed BAND spec (weights INSIDE, as integer live-game counts) derives
one gate_spec-compatible spec per seed; the band hash is embedded in the
derived spec's NAME, which gate_spec hashes and `run` stamps into every cell
header — so editing the weights (or any decision field) after cells exist
turns every decode into a refusal. Single source of truth: this instrument
reads NO other weighting table (scripts/band_decode.py's two-tables warning;
data/m39_live_mix.json is not consulted).

Advisory beds (`"advisory": true, "family": null` — tuned/iono) are decoded
in their own block and are STRUCTURALLY outside the weighted verdict (Q2:
anchors can neither adopt nor kill).

    m46_band_gate.py hash   docs/specs/m46_X.json
    m46_band_gate.py run    docs/specs/m46_X.json --out runs/m46_X --workers 8
    m46_band_gate.py decode docs/specs/m46_X.json --out runs/m46_X

Exit codes, PER MODE: gate mode 0=PASS 1=KILL/INCONCLUSIVE 2=refusal;
null_calibration 0=no family fired at the registered bar, 1=fired (re-set
`bars.family_z` from the printed recommendation BEFORE any candidate cell
runs), 2=refusal; acceptance mode (spec block `acceptance`) 0=criterion MET,
1=instrument failed validation or underpowered, 2=refusal.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.gate_spec as gs  # noqa: E402  (frozen cell runner, reused)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BAND_REQUIRED = ("name", "arm", "control", "beds", "n_per_cell", "seeds",
                 "weights", "bars", "bootstrap", "coverage_floor")
BAND_OPTIONAL_HASHED = ("acceptance", "null_calibration")
SPREAD_WARN = 0.10          # G-13 draw-spread WIDE flag (m40 precedent)
NULL_MINZ_B = 1000          # game-level bootstrap draws for the null min-z


def band_hash(spec: dict) -> str:
    """Canonical hash of every DECISION field, weights included. `question`/
    `note` stay outside (gate_spec:64-74 rationale); `acceptance` and
    `null_calibration` are decision content, so they are inside."""
    payload = {k: spec[k] for k in BAND_REQUIRED}
    payload.update({k: spec[k] for k in BAND_OPTIONAL_HASHED if k in spec})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_band_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in BAND_REQUIRED if k not in spec]
    if missing:
        raise SystemExit(f"band spec missing required field(s): {missing}")
    w = spec["weights"]
    if set(w) != {"n_games", "source", "games"}:
        raise SystemExit("weights must carry exactly n_games/source/games")
    if sum(w["games"].values()) != w["n_games"]:
        raise SystemExit(
            f"weights.games sum {sum(w['games'].values())} != n_games "
            f"{w['n_games']} — integer live-game counts are the contract")
    seeds = spec["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) for s in seeds)):
        raise SystemExit("seeds must be a non-empty list of ints")
    bars = spec["bars"]
    for k in ("pass", "kill", "family_z"):
        if k not in bars:
            raise SystemExit(f"bars must pre-register {k!r}")
    if bars["kill"] > bars["pass"]:
        raise SystemExit("kill bar above the pass bar — unreadable")
    if bars["family_z"] >= 0:
        raise SystemExit("family_z must be negative (it is a kill bar)")
    boot = spec["bootstrap"]
    for k in ("B", "seed", "stability_floor"):
        if k not in boot:
            raise SystemExit(f"bootstrap must pre-register {k!r}")
    fams_with_beds = set()
    for bed in spec["beds"]:
        if "name" not in bed or "spec" not in bed:
            raise SystemExit(f"every bed needs a name and a spec: {bed}")
        fam, advisory = bed.get("family"), bed.get("advisory", False)
        if advisory:
            if fam is not None:
                raise SystemExit(
                    f"advisory bed {bed['name']} carries a family — advisory "
                    "beds are structurally OUTSIDE the weighted verdict")
        else:
            if fam not in w["games"]:
                raise SystemExit(
                    f"weighted bed {bed['name']} family {fam!r} is not in "
                    "weights.games — every weighted cell must carry weight")
            fams_with_beds.add(fam)
    bedless = set(w["games"]) - fams_with_beds
    if bedless:
        raise SystemExit(
            f"weights name families with no bed: {sorted(bedless)} — weight "
            "mass without a cell is silent coverage loss")
    return spec


def derive_spec(spec: dict, seed: int) -> dict:
    """The per-seed gate_spec spec. The band hash rides inside `name`, which
    is in gate_spec's REQUIRED set — so gate_spec.spec_hash covers it and
    `run` stamps it into every cell: the tamper chain that makes a post-hoc
    weights edit a refusal, not a kinder verdict."""
    return {
        "name": f"{spec['name']}.b{band_hash(spec)[:8]}.s{seed}",
        "question": spec.get("question", ""),
        "arm": spec["arm"],
        "control": spec["control"],
        "beds": spec["beds"],           # extra keys ride along harmlessly
        "n_per_cell": spec["n_per_cell"],
        "seed": seed,
        "bars": {"pass": spec["bars"]["pass"], "kill": spec["bars"]["kill"]},
    }


def seed_dir(out: Path, seed: int) -> Path:
    return out / f"s{seed}"


def cmd_hash(spec_path: Path) -> int:
    spec = load_band_spec(spec_path)
    digest = band_hash(spec)
    print(f"band {digest}")
    for seed in spec["seeds"]:
        print(f"  s{seed} derived {gs.spec_hash(derive_spec(spec, seed))}")
    return 0


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def cmd_run(spec_path: Path, out: Path, workers: int) -> int:
    # M43 box law (docs/M43.md:258): 8 matchrunner workers x default BLAS
    # thread pools thrash the 16-thread box to ~1.5-2 games/s; single-thread
    # pinning is 15x. Set BEFORE the pool spawns — workers inherit the env.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    spec = load_band_spec(spec_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "band_spec.json").write_text(
        spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    # bed-weights manifest: the spec hash pins PATHS, not weights (the m43
    # box-migration lesson) — record what the paths held when the cells ran.
    manifest = {}
    for side in ("arm", "control"):
        for part in spec[side].split(":"):
            if part.endswith(".pt") and (ROOT / part).exists():
                manifest[part] = _md5(ROOT / part)
    for bed in spec["beds"]:
        for part in bed["spec"].split(":"):
            if part.endswith(".pt") and (ROOT / part).exists():
                manifest[part] = _md5(ROOT / part)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")
    print(f"band gate {spec['name']}  band {band_hash(spec)}  "
          f"seeds {spec['seeds']}  n/cell {spec['n_per_cell']}")
    for seed in spec["seeds"]:
        derived = derive_spec(spec, seed)
        sdir = seed_dir(out, seed)
        sdir.mkdir(parents=True, exist_ok=True)
        dpath = out / f"s{seed}.spec.json"
        dpath.write_text(json.dumps(derived, indent=1), encoding="utf-8")
        rc = gs.cmd_run(dpath, sdir, workers)
        if rc:
            return rc
    print(f"done — decode with: m46_band_gate.py decode {spec_path} "
          f"--out {out}")
    return 0


# --------------------------------------------------------------- decode ----

def _collect(spec: dict, out: Path):
    """(cells, refusals): cells[side][bed_name] = per-seed-concatenated
    results. Refusal-first, gate_spec discipline — no number is read into a
    verdict unless EVERY cell exists, is stamped with THIS band's derived
    hash, and holds the pre-registered n."""
    refusals: list[str] = []
    cells = {"arm": {}, "control": {}}
    for seed in spec["seeds"]:
        derived = derive_spec(spec, seed)
        digest = gs.spec_hash(derived)
        sdir = seed_dir(out, seed)
        for side in ("arm", "control"):
            for bed in spec["beds"]:
                path = gs.cell_path(sdir, side, bed["name"])
                if not path.exists():
                    refusals.append(f"missing cell s{seed}/{path.name}")
                    continue
                results, header = gs.read_cell(path)
                stamped = (header.get("extra") or {}).get("spec_hash")
                if stamped != digest:
                    refusals.append(
                        f"s{seed}/{path.name} ran under spec {stamped!r}, "
                        f"not {digest!r} — the band spec changed after the run")
                if len(results) != spec["n_per_cell"]:
                    refusals.append(
                        f"s{seed}/{path.name} holds {len(results)} games, "
                        f"spec pre-registered {spec['n_per_cell']}")
                cells[side].setdefault(bed["name"], []).extend(results)
    return cells, refusals


def _family_table(spec: dict, cells: dict):
    """fam -> list of per-bed (fa, na, fc, nc); plus per-bed rates for the
    G-13 spread read. Weighted beds only."""
    fams: dict[str, list] = {}
    bed_rates: dict[str, list] = {}
    for bed in spec["beds"]:
        if bed.get("advisory", False):
            continue
        a = cells["arm"][bed["name"]]
        c = cells["control"][bed["name"]]
        fam = bed["family"]
        fams.setdefault(fam, []).append((gs.wr(a), len(a), gs.wr(c), len(c)))
        bed_rates.setdefault(fam, []).append((bed["name"], gs.wr(a), gs.wr(c)))
    return fams, bed_rates


def _family_rates(cells_list):
    """m40_decide.py:251-254 law: family rate = n-weighted mean of its
    cells, arm and control separately."""
    na = sum(n for _, n, _, _ in cells_list)
    nc = sum(n for _, _, _, n in cells_list)
    fa = sum(p * n for p, n, _, _ in cells_list) / na
    fc = sum(p * n for _, _, p, n in cells_list) / nc
    return fa, na, fc, nc


def _weighted_delta(fams: dict, shares: dict):
    """m40_decide.py:248-300 arithmetic, replicated verbatim (the original is
    frozen and reads another cell layout; tests pin this copy to it
    numerically). Returns (delta, se, w_a/w_sum, w_c/w_sum)."""
    w_a = w_c = w_sum = 0.0
    for fam, cells_list in fams.items():
        share = shares.get(fam, 0.0)
        fa, _, fc, _ = _family_rates(cells_list)
        w_a += share * fa
        w_c += share * fc
        w_sum += share
    var = 0.0
    for fam, cells_list in fams.items():
        w = shares.get(fam, 0.0) / w_sum
        fa, na, fc, nc = _family_rates(cells_list)
        var += w * w * (fa * (1 - fa) / na + fc * (1 - fc) / nc)
    se = math.sqrt(var)
    return (w_a - w_c) / w_sum, se, w_a / w_sum, w_c / w_sum


def _delta_class(delta: float, bars: dict) -> str:
    if delta <= bars["kill"]:
        return "KILL"
    if delta >= bars["pass"]:
        return "PASS"
    return "INCONCLUSIVE"


def _weight_bootstrap(spec: dict, fams: dict):
    """Multinomial redraw of the n=83 live-game weights, family rates held
    FIXED: this isolates weight-sampling uncertainty (the SE already carries
    game-sampling uncertainty). Deterministic under bootstrap.seed. Returns
    (stability, lo, hi) — agreement fraction with the point delta-class and
    the 2.5/97.5 percentile delta band."""
    w = spec["weights"]
    boot = spec["bootstrap"]
    fam_names = sorted(fams)
    rates = {f: _family_rates(fams[f]) for f in fam_names}
    probs = np.array([w["games"][f] for f in fam_names], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(boot["seed"])
    draws = rng.multinomial(w["n_games"], probs, size=boot["B"])
    deltas = np.empty(boot["B"])
    for i, counts in enumerate(draws):
        num = den = 0.0
        for f, cnt in zip(fam_names, counts):
            fa, _, fc, _ = rates[f]
            num += cnt * (fa - fc)
            den += cnt
        deltas[i] = num / den if den else 0.0
    point = _delta_class(float(_weighted_delta(fams, {
        f: w["games"][f] / w["n_games"] for f in fam_names})[0]), spec["bars"])
    agree = np.mean([_delta_class(float(d), spec["bars"]) == point
                     for d in deltas])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(agree), float(lo), float(hi)


def _null_min_z(spec: dict, fams: dict):
    """Game-level bootstrap of the null min-family-z: resample each family's
    pooled arm/control results with replacement, recompute every family z,
    take the min — the distribution one 10-minute null run buys instead of a
    single draw of min-z. 5th percentile = the recommended calibrated bar."""
    rng = np.random.default_rng(spec["bootstrap"]["seed"] + 1)
    pooled = {}
    for fam, cells_list in fams.items():
        fa, na, fc, nc = _family_rates(cells_list)
        pooled[fam] = (fa, na, fc, nc)
    mins = np.empty(NULL_MINZ_B)
    for i in range(NULL_MINZ_B):
        zmin = math.inf
        for fam, (fa, na, fc, nc) in pooled.items():
            ra = rng.binomial(na, fa) / na
            rc = rng.binomial(nc, fc) / nc
            zmin = min(zmin, gs.two_proportion_z(ra, na, rc, nc))
        mins[i] = zmin
    return float(np.percentile(mins, 5))


def cmd_decode(spec_path: Path, out: Path) -> int:
    spec = load_band_spec(spec_path)
    w = spec["weights"]
    shares = {f: g / w["n_games"] for f, g in w["games"].items()}
    is_null = bool(spec.get("null_calibration"))
    acceptance = spec.get("acceptance")
    bars = spec["bars"]

    print(f"=== band gate {spec['name']} ===")
    print(f"band hash   {band_hash(spec)}")
    if spec.get("question"):
        print(f"question    {spec['question']}")
    print(f"weights     n={w['n_games']} live games — "
          + ", ".join(f"{f} {g}" for f, g in sorted(
              w["games"].items(), key=lambda kv: -kv[1])))
    print(f"bars        pass >= {bars['pass']:+.4f}  kill <= "
          f"{bars['kill']:+.4f}  family z <= {bars['family_z']:+.2f}")

    cells, refusals = _collect(spec, out)
    if refusals:
        print("\nREFUSED — this run does not match its pre-registered spec:")
        for r in refusals:
            print(f"  - {r}")
        print("\nNo verdict is emitted. That is the point: a bar that moves "
              "after the numbers land is not a bar.")
        return 2

    # --- per-cell table --------------------------------------------------
    n_seeds = len(spec["seeds"])
    print(f"\n{'cell':<16}{'fam':<12}{'arm':>18}{'control':>18}"
          f"{'delta':>10}{'z':>8}")
    for bed in spec["beds"]:
        if bed.get("advisory", False):
            continue
        a, c = cells["arm"][bed["name"]], cells["control"][bed["name"]]
        p_a, p_c = gs.wr(a), gs.wr(c)
        z = gs.two_proportion_z(p_a, len(a), p_c, len(c))
        print(f"{bed['name']:<16}{bed['family']:<12}"
              f"{p_a:>10.4f}+-{gs.ci95(p_a, len(a)):<6.4f}"
              f"{p_c:>10.4f}+-{gs.ci95(p_c, len(c)):<6.4f}"
              f"{p_a - p_c:>+10.4f}{z:>8.2f}")

    fams, bed_rates = _family_table(spec, cells)

    # --- family aggregation + kills + G-13 spread ------------------------
    print(f"\n{'family':<12}{'share':>7}{'arm':>9}{'control':>9}"
          f"{'delta':>10}{'z':>8}  draws")
    kills = []
    for fam in sorted(fams, key=lambda f: -shares[f]):
        fa, na, fc, nc = _family_rates(fams[fam])
        z = gs.two_proportion_z(fa, na, fc, nc)
        if z <= bars["family_z"]:
            kills.append((fam, z))
        arm_rates = [ra for _, ra, _ in bed_rates[fam]]
        spread = max(arm_rates) - min(arm_rates)
        note = (f"spread {spread:.3f}"
                + (" WIDE" if spread > SPREAD_WARN else "")
                if len(arm_rates) > 1 else "SINGLE-DRAW (G-13 caveat)")
        print(f"{fam:<12}{shares[fam]:>7.3f}{fa:>9.4f}{fc:>9.4f}"
              f"{fa - fc:>+10.4f}{z:>8.2f}  {note}")

    # --- coverage (G-3) --------------------------------------------------
    covered = sum(shares[f] for f in fams)
    print(f"\nroster coverage: {covered:.1%} of band mass "
          f"(floor {spec['coverage_floor']:.0%})")
    if covered < spec["coverage_floor"]:
        print("VERDICT: INVALID — coverage below the floor; a roster this "
              "thin measures a different meta than the one that decides ELO.")
        return 2

    # --- pools -----------------------------------------------------------
    uw_a = [r for bed in spec["beds"] if not bed.get("advisory", False)
            for r in cells["arm"][bed["name"]]]
    uw_c = [r for bed in spec["beds"] if not bed.get("advisory", False)
            for r in cells["control"][bed["name"]]]
    print(f"\nunweighted pool : arm {gs.wr(uw_a):.4f}  control "
          f"{gs.wr(uw_c):.4f}  delta {gs.wr(uw_a) - gs.wr(uw_c):+.4f}"
          "   (continuity only)")
    delta, se, wa, wc = _weighted_delta(fams, shares)
    z = delta / se if se else 0.0
    mde = 2.8 * se
    print(f"**WEIGHTED pool : arm {wa:.4f}  control {wc:.4f}  "
          f"delta {delta:+.4f} +/-{1.96 * se:.4f}  z={z:+.2f}**"
          "   <- THE gate number")
    print(f"   minimum detectable delta at this roster's n: +/-{mde:.4f} "
          f"({mde * 100:.1f}pp)")

    # --- advisory block --------------------------------------------------
    advisory = [b for b in spec["beds"] if b.get("advisory", False)]
    if advisory:
        print("\nADVISORY (never in the verdict — Q2: anchors can neither "
              "adopt nor kill):")
        for bed in advisory:
            a, c = cells["arm"][bed["name"]], cells["control"][bed["name"]]
            p_a, p_c = gs.wr(a), gs.wr(c)
            print(f"  {bed['name']:<14}arm {p_a:.4f}  control {p_c:.4f}  "
                  f"delta {p_a - p_c:+.4f}")

    # --- null-calibration mode -------------------------------------------
    if is_null:
        rec = _null_min_z(spec, fams)
        fired = [f"{fam} z={fz:+.2f}" for fam, fz in kills]
        print(f"\nNULL CALIBRATION (arm == control):")
        print(f"  families firing at the registered bar "
              f"{bars['family_z']:+.2f}: "
              + (", ".join(fired) if fired else "none"))
        print(f"  bootstrap null min-z 5th percentile: {rec:+.2f}  "
              "<- recommended calibrated family_z")
        print("  (regenerate the A0v/gate specs with --family-z at or below "
              "this BEFORE any candidate cell runs)")
        return 1 if fired else 0

    # --- family-kill scan + verdict --------------------------------------
    if kills:
        for fam, fz in kills:
            print(f"\nFAMILY-KILL: {fam} z={fz:+.2f} <= "
                  f"{bars['family_z']:+.2f} — a pooled positive can no "
                  "longer overrule a band kill (m44 post-mortem lesson 3.1)")
        verdict = "KILL"
    else:
        verdict = _delta_class(delta, bars)

    stability, lo, hi = _weight_bootstrap(spec, fams)
    print(f"\nweight bootstrap (B={spec['bootstrap']['B']}, n="
          f"{w['n_games']} games): delta 95% band [{lo:+.4f}, {hi:+.4f}]  "
          f"verdict agreement {stability:.3f} "
          f"(floor {spec['bootstrap']['stability_floor']})")
    if (verdict == "PASS"
            and stability < spec["bootstrap"]["stability_floor"]):
        print("  PASS-shaped but WEIGHT-UNSTABLE — the verdict flips under "
              "plausible redraws of the 83-game band sample. Downgraded to "
              "INCONCLUSIVE.")
        verdict = "INCONCLUSIVE"

    rc = {"PASS": 0, "KILL": 1, "INCONCLUSIVE": 1}[verdict]
    print(f"\nVERDICT: {verdict}   weighted delta {delta:+.4f} vs pass "
          f"{bars['pass']:+.4f} / kill {bars['kill']:+.4f}"
          + (f"   family kills: {[f for f, _ in kills]}" if kills else ""))
    if verdict == "INCONCLUSIVE" and not kills:
        print("  between the bars — the pre-registered outcome is 'we do "
              "not know', not a rounded-up pass.")

    # --- acceptance mode (A0v) -------------------------------------------
    if acceptance:
        expect, max_z = acceptance["expect"], acceptance["max_z"]
        met = verdict == expect and z <= max_z
        if met:
            print(f"\nA0V ACCEPTANCE: MET — verdict {verdict} at z={z:+.2f} "
                  f"(required {expect} with z <= {max_z:+.2f}). The panel "
                  "reproduces the live 810-vs-735 ordering.")
            return 0
        if delta < 0 and z > max_z:
            print(f"\nA0V ACCEPTANCE: directionally correct but UNDERPOWERED "
                  f"(z={z:+.2f} > {max_z:+.2f}) — escalate, do not "
                  "self-certify.")
            return 1
        print(f"\nA0V ACCEPTANCE: FAILED — INSTRUMENT INVALID (verdict "
              f"{verdict}, z={z:+.2f}; live truth says the control wins). "
              "Panel demoted to advisory; guards-only fallback; escalate "
              "to Piotr.")
        return 1

    print("\nQC: a qc_battery sweep by any >=20%-share family (grim, mirror) "
          "blocks a ship regardless of this verdict — enforced in the ship "
          "ritual, not here.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash", help="band hash + per-seed derived hashes")
    h.add_argument("spec", type=Path)
    r = sub.add_parser("run", help="run every seed x side x bed cell")
    r.add_argument("spec", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=8)
    d = sub.add_parser("decode", help="verify every cell, emit the verdict")
    d.add_argument("spec", type=Path)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "hash":
        return cmd_hash(a.spec)
    if a.cmd == "run":
        if a.workers > 8:                      # the standing parallelism cap
            raise SystemExit("--workers 8 is the proven-stable ceiling")
        return cmd_run(a.spec, a.out, a.workers)
    return cmd_decode(a.spec, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
