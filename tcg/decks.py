"""Deck-file loading, shared by every module that needs a deck list.

The same one-line reader used to be re-implemented in ``rl/collector.py``
(``_deck``), ``rl/bc.py`` (``_load_deck``), ``rl/value_train.py`` (``_deck``),
``rl/ppo.py``, ``rl/gate.py`` and ``rl/teacher.py`` (inline). Deck files are
whitespace-separated card ids (one per line by convention).

Kept dependency-free on purpose: deck files are read by the torch-side
training loops AND by the engine-only teacher loader.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK_DIR = ROOT / "decks"
ROOT_DECK_PATH = ROOT / "deck.csv"  # the deck the submission ships


def load_deck_file(path: Path) -> list[int]:
    """Read a deck file into a list of card ids."""
    return [int(token) for token in path.read_text().split() if token.strip()]


def load_deck(name: str) -> list[int]:
    """Read ``decks/<name>.csv`` (e.g. "lucario", "kyogre", "iono")."""
    return load_deck_file(DECK_DIR / f"{name}.csv")
