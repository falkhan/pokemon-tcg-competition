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

## Constants (MUST mirror rl/encoders.py -- verified by rl/gate.py encoder_parity_check)
FEAT_DIM = FEAT.shape[1]

N_BENCH = 5  # max bench slots we encode
N_CONTEXTS = 64  # SelectContext one-hot size (49 defined today; head-room for new ones)
N_OPTION_TYPES = 17  # OptionType one-hot size
N_ENERGY = 12  # EnergyType enum size
N_STATUS = 5  # poisoned/burned/asleep/paralyzed/confused

BASIC_FIGHTING_ENERGY = 6  # card id

# Per-Pokémon-slot: card features + hp/maxHp/energy-count + energy-type counts + tools pool
SLOT_DIM = FEAT_DIM + 3 + N_ENERGY + FEAT_DIM
# globals + hand pool + 2 discard pools + fighting-in-discard + status x2 + stadium + 12 slots
STATE_DIM = 7 + FEAT_DIM + 2 * FEAT_DIM + 1 + 2 * N_STATUS + FEAT_DIM + 2 * (1 + N_BENCH) * SLOT_DIM
# option-type one-hot + acted card + TARGET card + target-is-active flag
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM + FEAT_DIM + 1

## Neural Networks

def linear(x, name):                       # torch Linear stores weight as (out, in) -> transpose!
    return x @ POLICY_WEIGHTS[f"{name}.weight"].T + POLICY_WEIGHTS[f"{name}.bias"]

def relu(x):
    return np.maximum(0, x)

def score_options(state_ctx, options):    # state_ctx: (STATE_DIM+N_CONTEXTS,) ; options: (N, OPTION_DIM)
    s = relu(linear(relu(linear(state_ctx, "state_enc.0")), "state_enc.2"))      # (256,)
    o = relu(linear(options, "option_enc.0"))                                    # (N, 256)
    so = np.concatenate([np.broadcast_to(s, (len(o), s.size)), o], axis=1)       # (N, 512)
    return linear(relu(linear(so, "score_head.0")), "score_head.2").ravel()      # (N,) scores

## Encoders (copied verbatim from rl/encoders.py -- keep in sync!)

def _pool(cards, scale: float = 1.0) -> np.ndarray:
    """Sum of card features over a list of Card objects (zeros when empty)."""
    if not cards:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    return FEAT[[c.id for c in cards if c is not None]].sum(axis=0) * scale


def _poke_vec(p) -> np.ndarray:
    """Pokémon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if p is None:
        return v
    v[:FEAT_DIM] = FEAT[p.id]
    base = FEAT_DIM
    v[base:base + 3] = (p.hp / 340.0, p.maxHp / 340.0, len(p.energies) / 5.0)
    base += 3
    for e in p.energies:                       # energy TYPE counts, not just the total
        v[base + int(e)] += 1.0 / 3.0
    base += N_ENERGY
    v[base:base + FEAT_DIM] = _pool(p.tools)   # attached tools
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

    hand = _pool(me.hand)
    my_discard = _pool(me.discard, scale=0.1)
    op_discard = _pool(op.discard, scale=0.1)
    fighting_in_discard = np.array(
        [sum(1 for c in me.discard if c.id == BASIC_FIGHTING_ENERGY) / 10.0],
        dtype=np.float32)

    def status(ps):
        return np.array([ps.poisoned, ps.burned, ps.asleep, ps.paralyzed, ps.confused],
                        dtype=np.float32)

    stadium = _pool(state.stadium)

    def slots(ps):
        active = ps.active[0] if ps.active else None
        bench = list(ps.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        return [active] + bench

    parts = [glob, hand, my_discard, op_discard, fighting_in_discard,
             status(me), status(op), stadium]
    parts += [_poke_vec(p) for p in slots(me)]
    parts += [_poke_vec(p) for p in slots(op)]
    return np.concatenate(parts)


def _card_id_at(obs, area, index, player_index):
    """Card id at (area, index) for the given player, or None."""
    state = obs.current
    ps = state.players[player_index]
    zone = {  # AreaType values, see cg/api.py
        1: obs.select.deck, 2: ps.hand, 3: ps.discard, 4: ps.active,
        5: ps.bench, 6: ps.prize, 7: state.stadium, 12: state.looking,
    }.get(int(area))
    try:
        card = zone[index]
        return card.id if card is not None else None
    except (TypeError, IndexError):
        return None


def encode_option(opt, obs) -> np.ndarray:
    """cg.api.Option -> OPTION_DIM float32 vector.

    [OptionType one-hot | acted card features | TARGET Pokémon features | target-is-active]
    """
    v = np.zeros(OPTION_DIM, dtype=np.float32)
    v[int(opt.type)] = 1.0
    your_index = obs.current.yourIndex

    card_id = opt.cardId
    if card_id is None and opt.index is not None and opt.area is not None:
        player = opt.playerIndex if opt.playerIndex is not None else your_index
        card_id = _card_id_at(obs, opt.area, opt.index, player)
    if card_id:
        v[N_OPTION_TYPES:N_OPTION_TYPES + FEAT_DIM] = FEAT[card_id]

    if opt.inPlayArea is not None and opt.inPlayIndex is not None:
        target_id = _card_id_at(obs, opt.inPlayArea, opt.inPlayIndex, your_index)
        if target_id:
            base = N_OPTION_TYPES + FEAT_DIM
            v[base:base + FEAT_DIM] = FEAT[target_id]
        v[-1] = float(int(opt.inPlayArea) == 4)  # AreaType.ACTIVE
    return v


def encode_context(context) -> np.ndarray:
    """SelectContext -> N_CONTEXTS one-hot."""
    v = np.zeros(N_CONTEXTS, dtype=np.float32)
    v[int(context)] = 1.0
    return v


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
