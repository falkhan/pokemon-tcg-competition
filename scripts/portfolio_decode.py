"""E1a meta-weighted portfolio decode over a battery's JSONLs (M36 P2.5).

Aggregates per-bed pooled WRs into ONE composite weighted by the FROZEN live
meta shares, so battery movement can be read against the field we actually
face. The [[no-validated-live-predictor]] answer: with the wall bed the
portfolio covers ~77% of the live field — the first offline number with a
defensible claim to track live movement.

Caveat pre-registered ([[measurement-power-discipline]]): weighting adds NO
statistical power — per-bed n still governs; the composite is a decision aid,
never a significance bar. Per-bed bars are reported alongside.

Weights: FROZEN 2026-07-27 from the sub-55011605 live cache at n=57
(scripts/live_deck_race.py archetype counts; meta_v3 snapshot era):
mirror 12, archaludon 11, lucario 8, wall family 7 (crustle/tusk 6 + hop 1),
grim 5, rocket 1; uncovered by any bed: dragapult 5, starmie 3, barbaracle 2,
cynthia 2, buneary 1 (13/57 = 23%). kyogre bed = collapse guard, weight 0.

Decode law: results arrays code 0 = side-A WIN, 1 = loss, 2 = draw (half).

Usage: uv run python scripts/portfolio_decode.py [--prefix runs/m36_]
                                                 [--arms gacf gacfv]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# bed -> frozen live-game count it represents (n=57, see docstring)
BED_WEIGHTS = {
    "mirror": 12, "arch": 11, "luc": 8, "wall": 7, "grim": 5, "rocket": 1,
    "kyo": 0,
}
LIVE_N = 57
UNCOVERED = LIVE_N - sum(BED_WEIGHTS.values())   # dragapult/starmie/... tail


def pooled_wr(prefix: Path, arm: str, bed: str):
    """(wr, n) pooled over every seed file runs/<prefix><arm>_<bed>_s*.jsonl."""
    res = []
    for path in sorted(prefix.parent.glob(f"{prefix.name}{arm}_{bed}_s*.jsonl")):
        for line in path.open():
            d = json.loads(line)
            res += d.get("results", [])
    if not res:
        return None, 0
    w, dr = res.count(0), res.count(2)
    return (w + 0.5 * dr) / len(res), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="runs/m36_")
    ap.add_argument("--arms", nargs="+", default=["gacf", "gacfv"])
    args = ap.parse_args()
    prefix = ROOT / args.prefix if not Path(args.prefix).is_absolute() \
        else Path(args.prefix)

    covered_w = sum(w for w in BED_WEIGHTS.values() if w)
    print(f"frozen weights (live n={LIVE_N}): "
          f"{ {b: w for b, w in BED_WEIGHTS.items() if w} } | "
          f"covered {covered_w}/{LIVE_N} = {covered_w / LIVE_N:.0%} "
          f"(uncovered tail {UNCOVERED})")

    for arm in args.arms:
        num = den = var = 0.0
        rows = []
        for bed, w in BED_WEIGHTS.items():
            wr, n = pooled_wr(prefix, arm, bed)
            rows.append((bed, w, wr, n))
            if wr is None or not w:
                continue
            num += w * wr
            den += w
            var += (w ** 2) * wr * (1 - wr) / n
        print(f"\narm {arm}:")
        for bed, w, wr, n in rows:
            tag = "guard" if not w else f"w={w}"
            print(f"  {bed:7s} {tag:6s} " +
                  ("(no runs)" if wr is None else f"wr={wr:.4f} n={n}"))
        if den:
            comp = num / den
            ci = 1.96 * math.sqrt(var) / den
            print(f"  PORTFOLIO (over {den:.0f}/{LIVE_N} live games): "
                  f"{comp:.4f} ± {ci:.4f} (95% CI, binomial propagation)")


if __name__ == "__main__":
    main()
