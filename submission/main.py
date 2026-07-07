import os
import numpy as np
from cg.api import (to_observation_class)

## Load artifacts (local + /kaggle_simulations/agent/ on Kaggle)

def _base_dir():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    for d in [here, "/kaggle_simulations/agent", "submission", "."]:
        if d and os.path.exists(os.path.join(d, "policy_weights.npz")):
            return d
    raise FileNotFoundError("agent artifacts not found")

BASE = _base_dir()

POLICY_WEIGHTS = np.load(os.path.join(BASE, "policy_weights.npz"))
DECK = [int(x) for x in open(os.path.join(BASE, 'deck.csv')) if x.strip()]

FEAT = np.load(os.path.join(BASE, "card_features.npy"))

## Constants
FEAT_DIM = FEAT.shape[1]

N_BENCH = 5  # max bench slots we encode
N_CONTEXTS = 64  # SelectContext one-hot size (49 defined today; head-room for new ones)
N_OPTION_TYPES = 17  # OptionType one-hot size

# Per-Pokémon-slot extras beyond the card features: hp, maxHp, n_energies
SLOT_DIM = FEAT_DIM + 3
# global scalars + my hand pool + 6 of my slots + 6 opponent slots
STATE_DIM = 7 + FEAT_DIM + 2 * (1 + N_BENCH) * SLOT_DIM
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM

## Neural Networks

def linear(x, name):                       # torch Linear stores weight as (out, in) -> transpose!
    return x @ POLICY_WEIGHTS[f"{name}.weight"].T + POLICY_WEIGHTS[f"{name}.bias"]

def relu(x):
    return np.maximum(0, x)

def score_options(state_ctx, options):    # state_ctx: (575,) ; options: (N, 53)
    s = relu(linear(relu(linear(state_ctx, "state_enc.0")), "state_enc.2"))      # (256,)
    o = relu(linear(options, "option_enc.0"))                                    # (N, 256)
    so = np.concatenate([np.broadcast_to(s, (len(o), s.size)), o], axis=1)       # (N, 512)
    return linear(relu(linear(so, "score_head.0")), "score_head.2").ravel()      # (N,) scores

## Encoders

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


def encode_context(context) -> np.ndarray:
    """SelectContext -> N_CONTEXTS one-hot."""
    v = np.zeros(N_CONTEXTS, dtype=np.float32)
    v[int(context)] = 1.0
    return v

## Helpers

def _poke_vec(p) -> np.ndarray:
    """Pokémon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if p is None:
        return v
    v[:FEAT_DIM] = FEAT[p.id]
    v[FEAT_DIM:] = (p.hp / 340.0, p.maxHp / 340.0, len(p.energies) / 5.0)
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


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    state_ctx = np.concatenate([encode_state(obs.current),
                                encode_context(obs.select.context)])
    options = np.stack([encode_option(o, obs) for o in obs.select.option])
    scores = score_options(state_ctx, options)
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:obs.select.maxCount]]