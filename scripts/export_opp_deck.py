"""Export a harvested opponent deck to a decks-format CSV, reproducibly.

Every previous opponent-deck export (rocket_3394cd30, grimmsnarl_3121746f,
garchomp_m37) was an ad-hoc parquet filter in a REPL — the deck a bed plays
was chosen by hand and recorded nowhere. This makes the choice a command:

    # the family's most-played hash among seats at or above the band floor
    uv run python scripts/export_opp_deck.py --family dragapult --min-score 700

    # or pin an exact hash prefix (what the M30 rocket export did by hand)
    uv run python scripts/export_opp_deck.py --hash 3394cd30

Writes data/kaggle/<family>_<hash8>_deck.csv (decks format: one id per line)
after the same legality hard-fail as build_meta_field — an exported deck feeds
real games, so an illegal parse must stop the pipeline, not warp a bed.

Family names come from m39_live_mix.FAMILY_SIGNATURES so the census, the live
mix and this exporter can never disagree about what "dragapult" means.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402

from m39_live_mix import classify, load_pokemon_names  # noqa: E402
from rl.kaggle_ingest import _write_deck_csv, validate_deck  # noqa: E402

OPP_DECKS = ROOT / "data/kaggle/opp_decks.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--family", help="live family per m39_live_mix signatures")
    sel.add_argument("--hash", help="deck_hash prefix, min 8 hex chars")
    ap.add_argument("--min-score", type=float, default=700.0,
                    help="seat score floor for the most-played-hash vote")
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/kaggle/<family>_<hash8>_deck.csv")
    a = ap.parse_args()

    if a.hash and len(a.hash) < 8:
        print(f"--hash prefix too short ({a.hash!r}): 8+ hex chars, or "
              "collisions become silent")
        return 2

    names = load_pokemon_names()
    df = pl.read_parquet(OPP_DECKS).filter(~pl.col("is_ours").fill_null(False))

    votes: Counter[str] = Counter()
    decks: dict[str, list[int]] = {}
    fam_of_hash: dict[str, str] = {}
    for r in df.iter_rows(named=True):
        h = r["deck_hash"]
        if a.hash:
            if not h.startswith(a.hash):
                continue
        else:
            fam = fam_of_hash.get(h)
            if fam is None:
                fam = classify({names[c] for c in r["deck"] if c in names})
                fam_of_hash[h] = fam
            if fam != a.family:
                continue
            if r["score"] is None or r["score"] < a.min_score:
                continue
        votes[h] += 1
        decks.setdefault(h, list(r["deck"]))

    if not votes:
        target = a.hash or f"{a.family} @>={a.min_score:.0f}"
        print(f"no seats match {target} in {OPP_DECKS}")
        return 1

    top_hash, n = votes.most_common(1)[0]
    ids = decks[top_hash]
    family = a.family or classify({names[c] for c in ids if c in names})

    legal, reasons = validate_deck(ids)
    if not legal:
        print(f"harvested deck {top_hash[:12]} is ILLEGAL: {reasons}")
        return 1

    out = a.out or ROOT / f"data/kaggle/{family}_{top_hash[:8]}_deck.csv"
    _write_deck_csv(out, ids)
    others = ", ".join(f"{h[:8]}x{c}" for h, c in votes.most_common(6)[1:])
    print(f"{out}  <- hash {top_hash[:12]} ({n} seats"
          + (f"; runners-up {others}" if others else "") + ")")
    pokemon = sorted({names[c] for c in ids if c in names})
    print(f"  family {family}: {', '.join(pokemon)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
