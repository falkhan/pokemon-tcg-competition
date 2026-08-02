"""M40 X5 decode — a CI on the compression factor.

docs/M40-plan.md §1's keystone: M39's D1 put the 1000-band grim clone over the
750-band clone at 0.623, read as ~86 ELO of separation from demonstrators ~250
ELO apart, i.e. clones retain "roughly a third to a half" of demonstrator edge.
That came from ONE draw-pair in a regime G-13 prices at ~+/-6pp per draw.

This decodes the full 3x3 panel cross. Two variance components matter and they
answer different questions:

  WITHIN-cell (binomial)  how precisely we measured one pairing
  ACROSS-cell (draw)      how much the answer depends on which clones we drew

If the across-cell spread swamps the within-cell CI, the retention factor is
not a property of the band -- it is a property of the training lottery, and no
single pairing (including D1's) can estimate it.

Usage: uv run python scripts/m40_x5_decode.py
"""
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The plan's stated demonstrator separation between the two seat bands.
BAND_GAP_ELO = 250.0
D1_PIN = 0.623          # the single pairing §1 rests on


def elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def cell(h: str, lo: str):
    f = ROOT / "runs" / f"m40_x5_hi{h}_lo{lo}.jsonl"
    if not f.exists():
        return None
    res = []
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if "results" in d:
            res += d["results"]
    if not res:
        return None
    return (res.count(0) + 0.5 * res.count(2)) / len(res), len(res)


def main() -> int:
    draws = ("d1", "d2", "d3")
    grid, flat = {}, []
    for h in draws:
        for lo in draws:
            c = cell(h, lo)
            if c:
                grid[(h, lo)] = c
                flat.append(c[0])
    if not flat:
        print("no X5 cells - run scripts/m40_x5_compression.sh")
        return 2

    print("X5 — high-band (>=1000) vs low-band (700-850) grim clones, same list")
    print("rows = topgrim draw, cols = grim draw; cell = high-band's WR\n")
    print(f"  {'':<10}" + "".join(f"{'grim_'+d:>12}" for d in draws) + f"{'row mean':>11}")
    for h in draws:
        row = [grid[(h, lo)][0] for lo in draws if (h, lo) in grid]
        cells = "".join(f"{grid[(h,lo)][0]:>12.3f}" if (h, lo) in grid else f"{'-':>12}"
                        for lo in draws)
        print(f"  {'topgrim_'+h:<10}{cells}{(sum(row)/len(row)):>11.3f}")
    print(f"  {'col mean':<10}" + "".join(
        f"{sum(grid[(h,lo)][0] for h in draws if (h,lo) in grid)/len([1 for h in draws if (h,lo) in grid]):>12.3f}"
        for lo in draws))

    n_tot = sum(n for _, n in grid.values())
    mean = sum(p * n for p, n in grid.values()) / n_tot
    lo_, hi_ = min(flat), max(flat)
    # Across-cell SD (the construction lottery), and the SE of the panel mean
    # treating the 9 cells as the sampling unit -- which is the honest unit,
    # because the thing that varies between replications is the DRAW, not the
    # coin flips inside a fixed pairing.
    sd = math.sqrt(sum((p - sum(flat) / len(flat)) ** 2 for p in flat) / (len(flat) - 1))
    se_draw = sd / math.sqrt(len(flat))
    se_binom = math.sqrt(mean * (1 - mean) / n_tot)

    print(f"\n  cells {len(flat)}   games {n_tot}")
    print(f"  panel mean WR of the HIGH band     : {mean:.4f}")
    print(f"  range across draws                 : {lo_:.3f} .. {hi_:.3f} "
          f"(spread {hi_-lo_:.3f})")
    print(f"  within-cell (binomial) SE          : {se_binom:.4f}  "
          f"-> 95% CI +/-{1.96*se_binom:.4f}")
    print(f"  ACROSS-draw SE (the honest one)    : {se_draw:.4f}  "
          f"-> 95% CI +/-{1.96*se_draw:.4f}")
    print(f"  ratio across/within                : {se_draw/se_binom:.1f}x")

    ci_lo, ci_hi = mean - 1.96 * se_draw, mean + 1.96 * se_draw
    print(f"\n  HIGH-band WR  {mean:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]"
          f"   (D1's single pairing pinned {D1_PIN:.3f})")
    print(f"  implied ELO separation of the CLONES: {elo(mean):+.1f} "
          f"[{elo(ci_lo):+.1f}, {elo(ci_hi):+.1f}]")
    r, r_lo, r_hi = (elo(mean) / BAND_GAP_ELO, elo(ci_lo) / BAND_GAP_ELO,
                     elo(ci_hi) / BAND_GAP_ELO)
    print(f"  RETENTION FACTOR vs a {BAND_GAP_ELO:.0f}-ELO demonstrator gap:")
    print(f"      {r:.2f}   95% CI [{r_lo:.2f}, {r_hi:.2f}]")
    print(f"      (plan §1 asserts 'roughly a third to a half', i.e. 0.33-0.50,"
          f" from the single 0.623 pairing)")

    print("\n  READING:")
    if ci_lo <= 0.5 <= ci_hi:
        print("  * The interval SPANS 0.5 — parity between the bands is not"
              " excluded. A 250-ELO")
        print("    demonstrator gap cannot be shown to transmit AT ALL through"
              " BC cloning.")
    if se_draw > 2 * se_binom:
        print(f"  * Draw variance dominates sampling variance"
              f" ({se_draw/se_binom:.1f}x). The retention")
        print("    factor is a property of the TRAINING LOTTERY as much as of"
              " the band, so any")
        print("    single-pairing estimate — including the 0.623 the plan"
              " rests on — is not")
        print("    an estimate of the band effect. G-13 applies to compression"
              " too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
