"""M39: weighted-pool gate decoder — the G-3 instrument.

Why this exists (docs/M39-plan.md finding #2): M38's gate was honest and
still wrong. Its +6.1pp pooled delta was carried by the iono and dragapult
beds (+14/+17pp) which are ~5% of live games between them, while the beds
matching the real loss mass were flat, stale, or absent. Re-weighted by the
live opponent mix the delta was ~0 — and that re-weighting was possible at
gate time. From M39 on the WEIGHTED pool is THE gate number.

Three hard behaviours, all pre-registered:

  G-3  the weighted pool is the verdict; unweighted is printed for
       continuity only, and never as the headline.
  G-3  a roster covering < 80% of live-mix mass is INVALID — this script
       refuses to emit a verdict rather than emitting a soft one.
  G-9  every rate carries a 95% CI, and per-cell z is vs the SAME-BATTERY
       control (never a frozen historical pin).

Decode law (matchrunner checkpoints): 0 = side-a WIN, 1 = side-b win,
2 = draw and counts half.

Usage:
    uv run python scripts/m39_decide.py --arm cont3 --control champion
    uv run python scripts/m39_decide.py --arm cont3 --prefix m39_gate
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIX = ROOT / "data/m39_live_mix.json"
COVERAGE_FLOOR = 0.80
PANEL_SPREAD_WARN = 0.10   # G-13: draws disagreeing by >10pp = a lottery cell

# gate bed -> live family it stands in for. Two beds may share a family;
# their cells pool inside it and the family's live share weights the result.
# Kept in sync with BED_FAMILY in scripts/m39_live_mix.py.
BED_FAMILY = {
    "tuned": "lucario",
    "mirror": "mirror",
    "m28": "mirror",
    "grim": "grim",
    "wall": "wall",
    "archaludon": "archaludon",
    "dragapult": "dragapult",
    "iono": "iono",
    "rocket": "rocket",
    # NOTE no garchomp bed: checkpoints/m37_bc_garchomp.pt is a pre-v4 net
    # (state_enc input 1580 vs the current 1708) and will not load against
    # today's encoders. Garchomp is 4.8% of the live mix and sits in the
    # uncovered remainder; the roster still clears the G-3 floor without it.
    # Rebuild it (287 seats cached) if coverage ever needs the headroom.
    #
    # --- G-13 PANEL DRAWS (added 2026-08-02) ----------------------------
    # D1 found bed strength is a training lottery: three clones of the same
    # grim list from near-identical data scored .671/.623/.546 against the
    # same net at n=2400 each. A single-clone cell samples that lottery once
    # and reports the draw as if it were the family. Panel draws are extra
    # BEDS inside the SAME family, so the existing family pooling below
    # (n-weighted mean over the family's cells) already computes the panel
    # mean; what this section adds is the draws, and `draw_spread` below
    # adds the report G-13 requires.
    "wall_d1": "wall", "wall_d2": "wall", "wall_d3": "wall",
    "arch_d1": "archaludon", "arch_d2": "archaludon", "arch_d3": "archaludon",
    "grim_d1": "grim", "grim_d2": "grim", "grim_d3": "grim",
}

# Panel cells reported SEPARATELY and never pooled into the weighted number:
# the 900+ instrument is not part of the live mix, it is the slot-3 signal
# (plan revision 2). Three draws for the same G-13 reason.
TOP_BEDS = ("top_d1", "top_d2", "top_d3")

# Families whose cell is a single draw, for the G-13 caveat line. mirror/m28
# run our own champion lineage (a historical artifact, not a corpus we can
# re-draw); tuned/dragapult/iono are rule agents with no training draw at
# all; rocket's M30 corpus is not on this box.
SINGLE_DRAW_BEDS = ("tuned", "mirror", "m28", "dragapult", "iono", "rocket")


def decode(paths) -> tuple[float, int] | None:
    res = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if "results" in d:
                    res += d["results"]
    if not res:
        return None
    wins, draws = res.count(0), res.count(2)
    return (wins + 0.5 * draws) / len(res), len(res)


def cell(prefix: str, arm: str, bed: str):
    return decode(sorted(glob.glob(f"runs/{prefix}_{arm}_{bed}_s*.jsonl")))


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def two_prop_z(p1: float, n1: int, p2: float, n2: int) -> float:
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pool * (1 - pool), 1e-9) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", default="champion")
    ap.add_argument("--prefix", default="m39_gate")
    args = ap.parse_args()

    if not MIX.exists():
        print(f"missing {MIX} - run scripts/m39_live_mix.py first")
        return 1
    mix = json.loads(MIX.read_text())
    shares = {f: d["share"] for f, d in mix["families"].items()}

    print(f"=== M39 weighted gate: {args.arm} vs {args.control} "
          f"(mix from subs {mix['source_subs']}, n={mix['n_games']}) ===\n")

    # ---- per-cell table ------------------------------------------------
    rows, fam_cells = [], {}
    for bed, fam in BED_FAMILY.items():
        a, c = cell(args.prefix, args.arm, bed), cell(args.prefix, args.control, bed)
        if a is None or c is None:
            rows.append((bed, fam, None, None, None, None))
            continue
        (pa, na), (pc, nc) = a, c
        z = two_prop_z(pa, na, pc, nc)
        rows.append((bed, fam, pa, na, pc, z))
        fam_cells.setdefault(fam, []).append((pa, na, pc, nc))

    print(f"{'bed':<12}{'family':<12}{'share':>8}{'arm':>18}{'control':>10}"
          f"{'delta':>9}{'z':>8}")
    print("-" * 77)
    for bed, fam, pa, na, pc, z in rows:
        share = shares.get(fam, 0.0)
        if pa is None:
            print(f"{bed:<12}{fam:<12}{share:>8.3f}{'(pending)':>18}")
            continue
        pm = ci95(pa, na)
        print(f"{bed:<12}{fam:<12}{share:>8.3f}"
              f"{pa:>11.3f}+/-{pm:.3f}{pc:>10.3f}{pa - pc:>+9.3f}{z:>+8.2f}")

    # ---- panel spread (G-13) --------------------------------------------
    # The number that says whether a family's cell means anything. D1's
    # lottery had a 12.5pp spread against a 2.0pp binomial CI; a family whose
    # draws disagree by more than ~10pp is telling you its cell is measuring
    # the draw, not the family.
    panels = {}
    for bed, fam, pa, na, pc, z in rows:
        if pa is not None and bed.rsplit("_d", 1)[0] != bed:
            panels.setdefault(fam, []).append((bed, pa, pc))
    if panels:
        print(f"\n--- G-13 panel spread (>{PANEL_SPREAD_WARN * 100:.0f}pp "
              f"means the cell is measuring the draw, not the family) ---")
        for fam, draws in sorted(panels.items()):
            arm_sp = max(p for _, p, _ in draws) - min(p for _, p, _ in draws)
            ctl_sp = max(p for _, _, p in draws) - min(p for _, _, p in draws)
            flag = "  <- WIDE" if max(arm_sp, ctl_sp) > PANEL_SPREAD_WARN else ""
            per = " ".join(f"{b.rsplit('_', 1)[1]}={p:.3f}" for b, p, _ in draws)
            print(f"  {fam:<12} arm {per}   spread arm {arm_sp:.3f} / "
                  f"control {ctl_sp:.3f}{flag}")
        print("  (deltas are largely insulated - the same fixed draw serves "
              "both arms, so it cancels; ABSOLUTE cell values are not.)")
    singles = [b for b in SINGLE_DRAW_BEDS
               if any(r[0] == b and r[2] is not None for r in rows)]
    if singles:
        print(f"  single-draw cells (no panel): {', '.join(singles)}")

    # ---- coverage (G-3) -------------------------------------------------
    covered = sum(shares.get(f, 0.0) for f in fam_cells)
    total_share = sum(shares.values())
    coverage = covered / total_share if total_share else 0.0
    print(f"\nroster coverage: {coverage:.1%} of live-mix mass "
          f"(floor {COVERAGE_FLOOR:.0%})")
    uncovered = sorted(((s, f) for f, s in shares.items() if f not in fam_cells),
                       reverse=True)
    if uncovered:
        print("  uncovered: " + ", ".join(f"{f} {s:.1%}" for s, f in uncovered))

    # ---- pooled numbers -------------------------------------------------
    uw_a = uw_n = uw_c = 0.0
    w_a = w_c = w_sum = 0.0
    for fam, cells in fam_cells.items():
        share = shares.get(fam, 0.0)
        fa = sum(p * n for p, n, _, _ in cells) / sum(n for _, n, _, _ in cells)
        fc = sum(p * n for _, _, p, n in cells) / sum(n for _, _, _, n in cells)
        for p, n, pc, nc in cells:
            uw_a += p * n
            uw_c += pc * nc
            uw_n += n
        w_a += share * fa
        w_c += share * fc
        w_sum += share

    print(f"\nunweighted pool : arm {uw_a / uw_n:.4f}  control {uw_c / uw_n:.4f}"
          f"  delta {(uw_a - uw_c) / uw_n:+.4f}   (continuity only)")
    if w_sum:
        # G-9/G-12: a weighted delta printed WITHOUT an interval is exactly
        # the error this whole gate exists to stop. Var of a share-weighted
        # mean of independent per-family rates = sum(w^2 * p(1-p)/n), and the
        # arm/control batteries are independent samples, so their variances
        # add. Delta is what carries the decision, so delta gets the CI.
        var = 0.0
        for fam, cells in fam_cells.items():
            w = shares.get(fam, 0.0) / w_sum
            na = sum(n for _, n, _, _ in cells)
            nc = sum(n for _, _, _, n in cells)
            fa = sum(p * n for p, n, _, _ in cells) / na
            fc = sum(p * n for _, _, p, n in cells) / nc
            var += w * w * (fa * (1 - fa) / na + fc * (1 - fc) / nc)
        se = math.sqrt(var)
        delta = (w_a - w_c) / w_sum
        z = delta / se if se else 0.0
        print(f"**WEIGHTED pool : arm {w_a / w_sum:.4f}  control {w_c / w_sum:.4f}"
              f"  delta {delta:+.4f} +/-{1.96 * se:.4f}  z={z:+.2f}**"
              f"   <- THE gate number")
        mde = 2.8 * se   # ~80% power at alpha .05
        print(f"   minimum detectable delta at this roster's n: "
              f"+/-{mde:.4f} ({mde * 100:.1f}pp)")
        if abs(z) < 1.96:
            print(f"   => NO DETECTABLE DIFFERENCE (|z|<1.96). Report as "
                  f"'no detectable change', NOT as a gain or a regression "
                  f"(G-9). Raise n if a delta this size must be resolved.")

    # ---- the 900+ instrument, reported apart from the gate (G-8) ---------
    top = [(b, cell(args.prefix, args.arm, b), cell(args.prefix, args.control, b))
           for b in TOP_BEDS]
    top = [(b, a, c) for b, a, c in top if a and c]
    if top:
        pa = sum(p * n for _, (p, n), _ in top) / sum(n for _, (p, n), _ in top)
        pc = sum(p * n for _, _, (p, n) in top) / sum(n for _, _, (p, n) in top)
        na = sum(n for _, (_, n), _ in top)
        nc = sum(n for _, _, (_, n) in top)
        print(f"\n--- 900+ CEILING PANEL (NOT pooled into the gate) ---")
        for b, (p, n), (q, m) in top:
            print(f"  {b:<10} arm {p:.3f}+/-{ci95(p, n):.3f}   control {q:.3f}"
                  f"   delta {p - q:+.3f}")
        print(f"  PANEL MEAN arm {pa:.3f}  control {pc:.3f}  "
              f"delta {pa - pc:+.4f}  z={two_prop_z(pa, na, pc, nc):+.2f}")
        print(f"  gap vs parity: arm {pa - 0.5:+.3f}   "
              f"(plan P0.8 readings: <=10pp reachable / 10-25pp needs "
              f"best-response / >25pp corpus cannot teach the band)")
        print("  This is the slot-3 selection signal, never a Ship B gate "
              "(G-8): the top bed is not in the live mix.")

    if coverage < COVERAGE_FLOOR:
        print(f"\nVERDICT: **INVALID** - roster covers {coverage:.1%} < "
              f"{COVERAGE_FLOOR:.0%} of live mix (G-3). Add a bed for the "
              f"uncovered families above, or the gate says nothing.")
        return 2
    print("\nVERDICT: valid roster. Weighted delta above is the gate number; "
          "read it against the arm's pre-registered bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
