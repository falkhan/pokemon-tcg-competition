"""Deck legality + mutation-bandit deck search (ARCHITECTURE.md §3.3, §5.3).

Phase 0 uses a fixed deck; this module already provides the legality checker
(mirrors the one validated in deck_analysis.ipynb) and the mutation primitive
for Phase 1.
"""
import random
from collections import Counter
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECK_SIZE = 60
MAX_COPIES = 4

_cards = pl.read_parquet(DATA_DIR / "cards_features.parquet")
_ft = {r["card_id"]: r for r in _cards.iter_rows(named=True)}
ALL_IDS = list(_ft)


def validate_deck(ids: list[int]) -> tuple[bool, list[str]]:
    """Check a deck against the construction rules. Returns (is_legal, reasons)."""
    reasons = []
    if len(ids) != DECK_SIZE:
        reasons.append(f"deck has {len(ids)} cards, must be exactly {DECK_SIZE}")

    unknown = [i for i in ids if i not in _ft]
    if unknown:
        reasons.append(f"unknown card ids: {sorted(set(unknown))}")
    known = [i for i in ids if i in _ft]

    copies = Counter(_ft[i]["name"] for i in known if not _ft[i]["is_basic_energy"])
    over = {n: c for n, c in copies.items() if c > MAX_COPIES}
    if over:
        reasons.append(f">4 copies of: {over}")

    if not any(_ft[i]["is_basic"] and _ft[i]["is_pokemon"] for i in known):
        reasons.append("no Basic Pokémon (cannot legally start a game)")

    n_ace = sum(_ft[i]["is_ace_spec"] for i in known)
    if n_ace > 1:
        reasons.append(f"{n_ace} ACE SPEC cards (max 1)")

    return (len(reasons) == 0, reasons)


def mutate(deck: list[int], n_swaps: int | None = None,
           candidate_weights: dict[int, float] | None = None) -> list[int]:
    """Swap 1-4 random cards for new ones; always returns a *legal* deck.

    candidate_weights: optional card_id -> weight (e.g. learned card-impact
    scores) biasing which replacements get tried.
    """
    weights = None
    if candidate_weights:
        weights = [candidate_weights.get(i, 0.01) for i in ALL_IDS]

    for _ in range(200):  # retry until legal
        d = list(deck)
        for _ in range(n_swaps or random.randint(1, 4)):
            d[random.randrange(DECK_SIZE)] = random.choices(ALL_IDS, weights=weights)[0]
        if validate_deck(d)[0]:
            return d
    raise RuntimeError("could not produce a legal mutation in 200 tries")


# TODO(Phase 1): population loop — evaluate each deck with the current play
# policy (rl.eval.play_games), rate with openskill, replace the bottom quartile
# with mutations of the top quartile.
