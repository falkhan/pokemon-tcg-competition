"""Deck legality checker (ARCHITECTURE.md §3.3) — the ship-gate's validator.

Trimmed to the live surface in the M25 cleanup: the mutation primitives and
the self-play search loops (M4 — mutate / mutate_flex / rate_population /
hill_climb / evolve) were retired with the rest of the deck-search dead code;
the production copies live in ``rl/deck_search.py``.

This twin must stay importable under ``tcg``: ``tcg.shipping.deck_check``
runs after the gate has loaded the submission bundle's ``main.py``, which
puts the bundle first on ``sys.path`` — from then on ``rl`` resolves to the
bundle's stripped ``rl/`` package, so ``rl.deck_search`` is unreachable in
that process. Parity with the rl twin is pinned in tests/test_deck_search.py.
"""
from collections import Counter
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECK_SIZE = 60
MAX_COPIES = 4              # per card NAME, basic energy exempt

_cards = pl.read_parquet(DATA_DIR / "cards_features.parquet")
# One parquet row (as a plain dict) per card id: name/is_pokemon/is_basic/... flags.
CARD_ROWS: dict[int, dict] = {row["card_id"]: row
                              for row in _cards.iter_rows(named=True)}
ALL_CARD_IDS = list(CARD_ROWS)


def validate_deck(ids: list[int]) -> tuple[bool, list[str]]:
    """Check a deck against the construction rules. Returns (is_legal, reasons)."""
    reasons = []
    if len(ids) != DECK_SIZE:
        reasons.append(f"deck has {len(ids)} cards, must be exactly {DECK_SIZE}")

    unknown = [i for i in ids if i not in CARD_ROWS]
    if unknown:
        reasons.append(f"unknown card ids: {sorted(set(unknown))}")
    known = [i for i in ids if i in CARD_ROWS]

    copies = Counter(CARD_ROWS[i]["name"] for i in known
                     if not CARD_ROWS[i]["is_basic_energy"])
    over = {name: count for name, count in copies.items() if count > MAX_COPIES}
    if over:
        reasons.append(f">4 copies of: {over}")

    if not any(CARD_ROWS[i]["is_basic"] and CARD_ROWS[i]["is_pokemon"] for i in known):
        reasons.append("no Basic Pokémon (cannot legally start a game)")

    n_ace = sum(CARD_ROWS[i]["is_ace_spec"] for i in known)
    if n_ace > 1:
        reasons.append(f"{n_ace} ACE SPEC cards (max 1)")

    return (len(reasons) == 0, reasons)
