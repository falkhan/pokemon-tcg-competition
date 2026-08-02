"""Live read with confidence intervals — the G-9 instrument.

Why (docs/VALIDATION.md §0): the campaign has been quoting bare point
estimates off samples far too small to support them. Computed properly, M37
lands at [646, 819] and M38 at [566, 753] — intervals that overlap across
nearly their full width, so "M38 regressed from ~719 to 659" is a working
hypothesis, not a fact. The headlined wall cell (3-6 -> 1-5) is Fisher
p=0.60, i.e. indistinguishable from noise.

G-9 makes a CI mandatory on every live claim, and bars matchup-level gating
from live data at any n. This script exists so the honest number is the easy
number to quote.

    implied ELO ~ avg_opp + 700 * (WR - 0.5)     [reproduces M38: 659]

Usage:
    uv run python scripts/live_ci.py --sub 55146658
    uv run python scripts/live_ci.py --sub 55146658 --vs 55065484   # A/B
"""
import argparse
import math
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402

from m39_live_mix import classify, load_pokemon_names  # noqa: E402

ELO_PER_WR = 700.0
# Minimum n at which a live cell may inform a decision at all (G-9). Matchup
# cells never gate regardless of n — offline beds are that instrument.
GATE_MIN_N = 30


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — correct near 0 and 1, where our matchup cells live
    (a 0-6 cell has no normal-approximation interval worth printing)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on a 2x2 — the right test at our cell sizes."""
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0:
        return 1.0
    p0 = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    tot = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        k = a + c - i
        if k < 0 or k > c + d:
            continue
        p = comb(a + b, i) * comb(c + d, k) / comb(n, a + c)
        if p <= p0 + 1e-12:
            tot += p
    return min(1.0, tot)


def load(sub: int):
    names = load_pokemon_names()
    eps = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")
    od = pl.read_parquet(ROOT / "data/kaggle/opp_decks.parquet")
    fam = {}
    for e, s, deck in zip(od["episode_id"], od["seat"], od["deck"]):
        fam[(e, s)] = classify({names[c] for c in deck if c in names})

    rows = []
    for r in eps.iter_rows(named=True):
        for seat in (0, 1):
            if r[f"submission_id_{seat}"] != sub:
                continue
            opp = 1 - seat
            if r[f"submission_id_{opp}"] == sub:
                continue                      # self-validation
            rew, orew = r[f"reward_{seat}"], r[f"reward_{opp}"]
            osc = r[f"updated_score_{opp}"]
            if rew is None or orew is None or osc is None:
                continue
            rows.append((1 if rew > orew else (0.5 if rew == orew else 0),
                         float(osc), fam.get((r["episode_id"], opp), "unknown")))
    return rows


def report(sub: int, rows) -> tuple[float, int]:
    n = len(rows)
    if not n:
        print(f"sub {sub}: no ladder games in cache")
        return (0.0, 0)
    wins = sum(r[0] for r in rows)
    wr = wins / n
    avg_opp = sum(r[1] for r in rows) / n
    lo, hi = wilson(round(wins), n)
    elo = avg_opp + ELO_PER_WR * (wr - 0.5)
    print(f"\n=== sub {sub}: n={n}  {round(wins)}W-{n - round(wins)}L  "
          f"WR={wr:.3f}  avg_opp={avg_opp:.1f} ===")
    print(f"  implied ELO {elo:.0f}   95% CI [{avg_opp + ELO_PER_WR * (lo - 0.5):.0f}, "
          f"{avg_opp + ELO_PER_WR * (hi - 0.5):.0f}]   "
          f"(half-width +/-{ELO_PER_WR * (hi - lo) / 2:.0f})")
    if n < GATE_MIN_N:
        print(f"  G-9: n < {GATE_MIN_N} — this read may NOT inform a decision.")

    cells = defaultdict(lambda: [0, 0])
    for w, _, f in rows:
        cells[f][0 if w == 1 else 1] += 1
    print(f"\n  {'opponent':<14}{'W-L':>8}{'WR':>7}{'95% CI':>18}   forensic only (G-9)")
    for f, (w, l) in sorted(cells.items(), key=lambda kv: -sum(kv[1])):
        cl, ch = wilson(w, w + l)
        print(f"  {f:<14}{f'{w}-{l}':>8}{w / (w + l):>7.2f}"
              f"{f'[{cl:.2f}, {ch:.2f}]':>18}")
    print("\n  Matchup cells NEVER gate a decision (G-9) — they generate "
          "hypotheses for offline beds to test.")
    return (wr, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--vs", type=int, default=None, help="comparator sub")
    args = ap.parse_args()

    rows_a = load(args.sub)
    report(args.sub, rows_a)
    if args.vs:
        rows_b = load(args.vs)
        report(args.vs, rows_b)
        wa = round(sum(r[0] for r in rows_a))
        wb = round(sum(r[0] for r in rows_b))
        p = fisher(wa, len(rows_a) - wa, wb, len(rows_b) - wb)
        print(f"\n=== {args.sub} vs {args.vs}: Fisher two-sided p = {p:.3f} ===")
        print("  " + ("DETECTABLE difference." if p < 0.05 else
                      "NOT a detectable difference at these sample sizes — "
                      "report as 'no detectable change', not as a regression."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
