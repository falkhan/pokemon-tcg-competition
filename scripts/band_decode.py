"""600-band-weighted portfolio decode (M37 — the band-trap answer).

Same decode law as scripts/portfolio_decode.py (0 = side-A WIN, 2 = draw
counts half), but weighted by the 600-BAND composition the M36 sub actually
faced, not the 750-band meta_v3 shares. Rationale (docs/m36-post-mortem.md):
a new sub seeds at ~600 and must beat the 600-band mix (40% stall/grim) to
climb; the band composite is therefore the M37 ship-gate number.

Weights: FROZEN 2026-07-28 from the sub-55030954 live cache, n=52 ladder
games (scripts/pm_extra.py ledger): mirror 12, grim 8, lucario 8,
crustle/tusk wall 5, starmie 4, other 4, rocket 3, hop stall 2, garchomp 2,
dragapult 2, archaludon 1, lucario-solrock 1. Beds existing locally cover
38/52 = 73% (uncovered: rocket 3 + starmie 4 + dragapult 2 + solrock 1 +
other 4 = 14).

Caveat ([[measurement-power-discipline]]): weighting adds NO statistical
power — per-bed n still governs; the composite is a decision aid and a
pre-registered gate INPUT, never a significance bar on its own.

Usage: uv run python scripts/band_decode.py [--prefix runs/m37_strawman_]
                                            [--arms ""]
  With --arms "" the files are runs/<prefix><bed>_s*.jsonl (no arm segment);
  with arm names they are runs/<prefix><arm>_<bed>_s*.jsonl.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# bed -> frozen 600-band live-game count it represents (n=52, see docstring)
BAND_WEIGHTS = {
    "mirror": 12, "grimlive": 8, "luc": 8, "wall": 5, "hop": 2,
    "garchomp": 2, "arch": 1,
}
LIVE_N = 52
UNCOVERED = LIVE_N - sum(BAND_WEIGHTS.values())


def pooled_wr(prefix: Path, arm: str, bed: str):
    seg = f"{arm}_" if arm else ""
    res = []
    for path in sorted(prefix.parent.glob(f"{prefix.name}{seg}{bed}_s*.jsonl")):
        for line in path.open():
            d = json.loads(line)
            res += d.get("results", [])
    if not res:
        return None, 0
    w, dr = res.count(0), res.count(2)
    return (w + 0.5 * dr) / len(res), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="runs/m37_strawman_")
    ap.add_argument("--arms", nargs="*", default=[""])
    args = ap.parse_args()
    prefix = ROOT / args.prefix if not Path(args.prefix).is_absolute() \
        else Path(args.prefix)

    covered_w = sum(BAND_WEIGHTS.values())
    print(f"600-band frozen weights (live n={LIVE_N}): {BAND_WEIGHTS} | "
          f"covered {covered_w}/{LIVE_N} = {covered_w / LIVE_N:.0%} "
          f"(uncovered tail {UNCOVERED})")

    for arm in args.arms:
        num = den = var = 0.0
        rows = []
        for bed, w in BAND_WEIGHTS.items():
            wr, n = pooled_wr(prefix, arm, bed)
            rows.append((bed, w, wr, n))
            if wr is None:
                continue
            num += w * wr
            den += w
            var += (w ** 2) * wr * (1 - wr) / n
        print(f"\narm {arm or '(none)'}:")
        for bed, w, wr, n in rows:
            print(f"  {bed:9s} w={w:<2d} " +
                  ("(no runs)" if wr is None else f"wr={wr:.4f} n={n}"))
        if den:
            comp = num / den
            ci = 1.96 * math.sqrt(var) / den
            print(f"  BAND COMPOSITE (over {den:.0f}/{LIVE_N} live games): "
                  f"{comp:.4f} ± {ci:.4f} (95% CI, binomial propagation)")


if __name__ == "__main__":
    main()
