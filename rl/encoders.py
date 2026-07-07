"""Observation / option encoders: game objects -> numpy vectors.

numpy-only on purpose — the Kaggle submission ships these encoders plus .npz
weights and never imports torch (see ARCHITECTURE.md §7.1).

Feature source: data/cards_features.parquet, produced by deck_analysis.ipynb.
"""
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Card ids are 1..1267; row 0 of the matrix is an all-zeros "no card" padding row.
_cards = pl.read_parquet(DATA_DIR / "cards_features.parquet")
_numeric = _cards.drop(["card_id", "name"]).cast(pl.Float32)
FEAT_DIM = _numeric.width
FEAT = np.zeros((_cards["card_id"].max() + 1, FEAT_DIM), dtype=np.float32)
FEAT[_cards["card_id"].to_numpy()] = _numeric.to_numpy()

N_BENCH = 5          # max bench slots we encode
N_CONTEXTS = 64      # SelectContext one-hot size (49 defined today; head-room for new ones)
N_OPTION_TYPES = 17  # OptionType one-hot size

# Per-Pokémon-slot extras beyond the card features: hp, maxHp, n_energies
SLOT_DIM = FEAT_DIM + 3
# global scalars + my hand pool + 6 of my slots + 6 opponent slots
STATE_DIM = 7 + FEAT_DIM + 2 * (1 + N_BENCH) * SLOT_DIM
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM


def _poke_vec(p) -> np.ndarray:
    """Pokémon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if p is None:
        return v
    v[:FEAT_DIM] = FEAT[p.id]
    v[FEAT_DIM:] = (p.hp / 340.0, p.maxHp / 340.0, len(p.energies) / 5.0)
    return v


def encode_state(state) -> np.ndarray:
    """cg.api.State -> STATE_DIM float32 vector."""
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    glob = np.array([
        state.turn / 30.0,
        len(me.prize) / 6.0,
        len(op.prize) / 6.0,
        me.deckCount / 60.0,
        op.handCount / 15.0,
        float(state.energyAttached),
        float(state.supporterPlayed),
    ], dtype=np.float32)

    hand = np.zeros(FEAT_DIM, dtype=np.float32)
    if me.hand:
        hand = FEAT[[c.id for c in me.hand]].sum(axis=0)

    def slots(ps):
        active = ps.active[0] if ps.active else None
        bench = list(ps.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        return [active] + bench

    parts = [glob, hand]
    parts += [_poke_vec(p) for p in slots(me)]
    parts += [_poke_vec(p) for p in slots(op)]
    return np.concatenate(parts)


def encode_option(opt, obs) -> np.ndarray:
    """cg.api.Option -> OPTION_DIM float32 vector (type one-hot + resolved card features)."""
    v = np.zeros(OPTION_DIM, dtype=np.float32)
    v[int(opt.type)] = 1.0

    card_id = opt.cardId
    if card_id is None and opt.index is not None and opt.area is not None:
        card_id = _resolve_card_id(opt, obs)
    if card_id:
        v[N_OPTION_TYPES:] = FEAT[card_id]
    return v


def _resolve_card_id(opt, obs) -> int | None:
    """Follow an option's (playerIndex, area, index) pointer to the card it refers to."""
    state = obs.current
    ps = state.players[opt.playerIndex if opt.playerIndex is not None else state.yourIndex]
    area = {  # AreaType values, see cg/api.py
        1: obs.select.deck, 2: ps.hand, 3: ps.discard, 4: ps.active,
        5: ps.bench, 6: ps.prize, 7: state.stadium, 12: state.looking,
    }.get(int(opt.area))
    try:
        card = area[opt.index]
        return card.id if card is not None else None
    except (TypeError, IndexError):
        return None


def encode_context(context) -> np.ndarray:
    """SelectContext -> N_CONTEXTS one-hot."""
    v = np.zeros(N_CONTEXTS, dtype=np.float32)
    v[int(context)] = 1.0
    return v
