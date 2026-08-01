"""M39 P0 census — what beds and corpora the cached harvest can actually build.

Two questions, one pass over `data/kaggle/opp_decks.parquet`:

1. **Bed buildability.** Seats per live family per score band. A clone bed
   needs seats AT the band its live pilots play at — the M38 lesson (the
   grim bed was an M26-era 600-band clone while live grim pilots sat at
   738-838, and it lied: offline +4pp, live 1-6).

2. **The vs-loss census** (M39 P3 corpus A, and the deck-decision
   falsification test): how many 800+ winner seats on OUR deck exist
   AGAINST each opponent family. Plenty -> the losing matchups are winnable
   with better lines. Nearly none -> strong pilots lose or dodge them too.

Family classification reuses `scripts/m39_live_mix.py` so the census and the
gate weights can never drift apart.

Usage:
    uv run python scripts/m39_census.py
    uv run python scripts/m39_census.py --our-hash 9294d9d8
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402

from m39_live_mix import classify, load_pokemon_names  # noqa: E402

OPP_DECKS = ROOT / "data/kaggle/opp_decks.parquet"

# Score bands named for what they mean to us, not round numbers: <700 is
# below our own live band (a bed built here is a strawman), 700-850 is where
# the loss mass lives, 850+ is the ceiling we cannot currently measure.
BANDS = ((0, 700, "<700 strawman"), (700, 850, "700-850 LIVE"),
         (850, 1000, "850-1000 high"), (1000, 9999, "1000+ ceiling"))

MIN_BED_SEATS = 100   # m38_bc_wall shipped on 138; 100 is the working floor


def band_of(score) -> str:
    """Seats whose leaderboard score the replay never carried land in
    `no score` — they are unusable for a band-faithful bed, so they must be
    visible in the census rather than silently bucketed."""
    if score is None:
        return "no score"
    for lo, hi, name in BANDS:
        if lo <= score < hi:
            return name
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-hash", default="9294d9d8",
                    help="deck-hash prefix of the list we pilot")
    ap.add_argument("--min-score", type=float, default=800.0,
                    help="vs-loss census: winner-seat score floor")
    args = ap.parse_args()

    if not OPP_DECKS.exists():
        print(f"missing {OPP_DECKS} - run `python -m rl.kaggle_ingest harvest`")
        return 1

    names = load_pokemon_names()
    df = pl.read_parquet(OPP_DECKS)

    # family per seat, from the seat's own decklist
    fams, pokemon_sets = [], []
    for deck in df["deck"].to_list():
        pk = {names[c] for c in deck if c in names}
        pokemon_sets.append(pk)
        fams.append(classify(pk))
    df = df.with_columns(pl.Series("family", fams),
                         pl.Series("band", [band_of(s) for s in df["score"]]))

    print(f"=== M39 census over {len(df)} cached seats "
          f"({df['episode_id'].n_unique()} episodes) ===\n")

    # ---- 1. bed buildability -------------------------------------------
    print("--- BED BUILDABILITY: seats per family per band ---")
    band_names = [b[2] for b in BANDS]
    grid: dict = defaultdict(Counter)
    for fam, band in zip(df["family"], df["band"]):
        grid[fam][band] += 1
    header = f"{'family':<14}" + "".join(f"{b:>16}" for b in band_names) + f"{'TOTAL':>8}"
    print(header)
    print("-" * len(header))
    for fam, counts in sorted(grid.items(), key=lambda kv: -sum(kv[1].values())):
        row = f"{fam:<14}" + "".join(f"{counts[b]:>16}" for b in band_names)
        total = sum(counts.values())
        live = counts["700-850 LIVE"]
        flag = "  <- BUILDABLE at band" if live >= MIN_BED_SEATS else ""
        print(row + f"{total:>8}" + flag)

    # ---- 2. vs-loss census ---------------------------------------------
    print(f"\n--- VS-LOSS CENSUS: our-deck ({args.our_hash}) winner seats "
          f"@>={args.min_score:.0f}, by OPPONENT family ---")
    ours = df.filter(
        pl.col("deck_hash").str.starts_with(args.our_hash)
        & pl.col("won")
        & (pl.col("score") >= args.min_score))
    by_ep = {e: f for e, f in zip(df["episode_id"], df["family"])}
    # opponent family = the other seat of the same episode
    opp_fam: Counter = Counter()
    seat_lookup = defaultdict(dict)
    for e, s, f in zip(df["episode_id"], df["seat"], df["family"]):
        seat_lookup[e][s] = f
    for e, s in zip(ours["episode_id"], ours["seat"]):
        other = seat_lookup[e].get(1 - s)
        opp_fam[other or "unknown"] += 1
    total = sum(opp_fam.values())
    print(f"{'opponent':<14}{'winner seats':>14}")
    print("-" * 28)
    for fam, n in opp_fam.most_common():
        print(f"{fam:<14}{n:>14}")
    print(f"{'TOTAL':<14}{total:>14}")
    if by_ep:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
