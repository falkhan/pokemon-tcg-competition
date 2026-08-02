"""600-band-weighted portfolio decode (M37 — the band-trap answer).

SUPERSEDED for M39+ ship gates by `scripts/m39_decide.py`, which reads the
committed `data/m39_live_mix.json` instead of an in-code table and enforces
G-3's 80% coverage floor (a roster below it emits no verdict at all). Kept
for decoding historical M36/M37 batteries, and re-weighted below onto the
current sample so any legacy use is not two metas stale — but do NOT add a
second weighting table to the M39 gate path; two sources of truth drift,
which is exactly the class of error M39 exists to stop.

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

# bed -> live-game count it represents.
# RE-FROZEN 2026-08-01 (M39 P0.6) on the pooled M37 55065484 + M38 55146658
# sample, n=147 ladder games — the same source as data/m39_live_mix.json, so
# the two cannot disagree about the meta even though only the json is the
# gate input. Previous freeze was the sub-55030954 600-band cache (n=52),
# two metas stale. The headline shift: archaludon 1 -> 18 (it was a rounding
# error at the 600 band and is now the third-largest family), stall 2 -> 4.
BAND_WEIGHTS = {
    "luc": 29, "mirror": 23, "wall": 20, "arch": 18, "grimlive": 18,
    "dragapult": 7, "garchomp": 7, "rocket": 5, "hop": 4, "iono": 1,
}
LIVE_N = 147
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
