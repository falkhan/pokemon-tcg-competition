"""Deck-agnostic rule-based pilot for the Kaggle Pokémon TCG AI Battle (cabt).

Readable, dataclass-based refactor of the pilot that previously lived in
``rl/combat.py`` + ``rl/generic_pilot.py`` (see docs/M6.md). Pure Python with
``cg`` (the bundled game engine) as the only dependency — no numpy / polars /
torch — so the whole package can ship inside a Kaggle submission bundle.

Package layout:

    models.py     Card / Attack dataclasses built from the engine's card database
    library.py    the card database: CARDS / ATTACKS lookup tables + card-id sets
    constants.py  every tuned score tier and game constant, with the rationale
    combat.py     damage math: can_afford() / best_damage()
    pilot.py      make_generic_pilot() — scores every legal option each turn

This package intentionally does not import its submodules here: building the
card tables requires the ``cg`` engine, and ``import tcg`` alone should stay
side-effect free.

The training-side ``rl/`` package and the submission pipeline still use the old
modules; switching them over to ``tcg`` is a follow-up (docs/M6.md).
"""
