"""Observation / option encoders: game objects -> numpy vectors.

numpy-only on purpose — the Kaggle submission ships these encoders plus .npz
weights and never imports torch (see ARCHITECTURE.md §7.1).

Feature source: data/cards_features.parquet, produced by deck_analysis.ipynb.

Stage B enrichment (docs/M1-plan.md §B1 + the aliasing diagnosis from Stage A):
- options now encode the action's TARGET (which Pokémon an ATTACH/EVOLVE acts on) —
  in Stage A, "attach to Mega Lucario" and "attach to a benchwarmer" were identical
  vectors (337/500 menus had aliased options)
- state now sees: both discard piles, own discarded Basic Fighting Energy count
  (Mega Lucario's attack scales on it), per-slot energy TYPE counts, attached
  tools, status conditions, and the stadium
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
N_ENERGY = 12        # EnergyType enum size (Colorless..Team_Rocket)
N_STATUS = 5         # poisoned/burned/asleep/paralyzed/confused

BASIC_FIGHTING_ENERGY = 6  # card id; teacher's Mega Brave scales on discarded copies

# NOTE: a per-card-id "revealed opponent cards" feature was tried and reverted — a probe
# showed the opponent archetype is ALREADY ~98% identifiable from the pooled features below,
# so it was redundant and only added overfitting. See docs/DECISIONS.md 2026-07-08.

# --- Combat-lookahead tables (from the engine; available in training AND submission) ---
# The rule experts reason with hidden-state combat math (damage / KO / prize race) that BC
# couldn't imitate from raw board features. We compute the SAME quantities and expose them.
from cg.api import all_card_data as _all_card_data, all_attack as _all_attack  # noqa: E402
_ATK = {a.attackId: (a.damage, tuple(int(e) for e in a.energies)) for a in _all_attack()}
_CARD = {c.cardId: (c.weakness, c.resistance, int(c.energyType), c.attacks,
                    3 if c.megaEx else 2 if c.ex else 1)
         for c in _all_card_data()}
COLORLESS = 0
N_COMBAT = 11        # combat-lookahead features (see _combat_features)

# Per-Pokémon-slot: card features + hp/maxHp/energy-count + energy-type counts + tools pool
SLOT_DIM = FEAT_DIM + 3 + N_ENERGY + FEAT_DIM
# globals + my hand pool + 2 discard pools + my discarded-fighting-energy count
# + status x2 + stadium + combat features + 6 of my slots + 6 opponent slots
STATE_DIM = (7 + FEAT_DIM + 2 * FEAT_DIM + 1 + 2 * N_STATUS + FEAT_DIM + N_COMBAT
             + 2 * (1 + N_BENCH) * SLOT_DIM)
# option-type one-hot + acted card features + TARGET card features + target-is-active flag
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM + FEAT_DIM + 1


def _can_afford(energies, cost) -> bool:
    """Do a Pokémon's attached energies cover an attack's cost (colorless=any)?"""
    have = {}
    for e in energies:
        have[e] = have.get(e, 0) + 1
    total = len(energies)
    n_colorless = sum(1 for c in cost if c == COLORLESS)
    used = 0
    for c in cost:
        if c == COLORLESS:
            continue
        if have.get(c, 0) <= 0:
            return False
        have[c] -= 1
        used += 1
    return total - used >= n_colorless


def _best_damage(attacker, target, extra_energy: int = 0) -> int:
    """Max damage `attacker` can deal to `target` this turn (best affordable attack,
    after weakness/resistance vs the attacker's type). extra_energy simulates attaching
    that many of the attacker's own energy (the '+1 attach enables the attack' case)."""
    if attacker is None or target is None or attacker.id not in _CARD:
        return 0
    _, _, atk_type, attacks, _ = _CARD[attacker.id]
    energies = list(attacker.energies) + [atk_type] * extra_energy
    t_weak, t_res, _, _, _ = _CARD.get(target.id, (None, None, 0, [], 1))
    best = 0
    for aid in attacks:
        if aid not in _ATK:
            continue
        dmg, cost = _ATK[aid]
        if dmg <= 0 or not _can_afford(energies, cost):
            continue
        if t_weak is not None and int(t_weak) == atk_type:
            dmg *= 2
        elif t_res is not None and int(t_res) == atk_type:
            dmg = max(0, dmg - 30)
        best = max(best, dmg)
    return best


def _combat_features(state) -> np.ndarray:
    """The tactical quantities the rule experts reason over (damage / KO / prize race)."""
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]
    my_act = me.active[0] if me.active else None
    op_act = op.active[0] if op.active else None

    my_dmg = _best_damage(my_act, op_act)
    op_dmg = _best_damage(op_act, my_act)
    my_ko = float(op_act is not None and my_dmg >= op_act.hp)
    op_ko = float(my_act is not None and op_dmg >= my_act.hp)
    one_from = float(op_act is not None and not my_ko
                     and _best_damage(my_act, op_act, extra_energy=1) >= op_act.hp)
    my_can_attack = float(_best_damage(my_act, op_act) > 0 or
                          (my_act is not None and any(
                              _ATK.get(a, (0, ()))[0] > 0 and _can_afford(my_act.energies, _ATK[a][1])
                              for a in _CARD.get(my_act.id, (None, None, 0, [], 1))[3] if a in _ATK)))
    op_prize = _CARD.get(op_act.id, (None, None, 0, [], 1))[4] / 3.0 if op_act else 0.0
    n_ready = sum(1 for p in ([my_act] + list(me.bench))
                  if p is not None and any(
                      a in _ATK and _ATK[a][0] > 0 and _can_afford(p.energies, _ATK[a][1])
                      for a in _CARD.get(p.id, (None, None, 0, [], 1))[3]))

    return np.array([
        my_dmg / 340.0, my_ko, my_can_attack, one_from,
        op_dmg / 340.0, op_ko,
        (len(me.prize) - len(op.prize)) / 6.0,   # prize race (negative = I'm ahead)
        op_prize,
        n_ready / 6.0,
        (my_act.hp / max(1, my_act.maxHp)) if my_act else 0.0,
        (op_act.hp / max(1, op_act.maxHp)) if op_act else 0.0,
    ], dtype=np.float32)


def _pool(cards, scale: float = 1.0) -> np.ndarray:
    """Sum of card features over a list of Card objects (zeros when empty)."""
    if not cards:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    return FEAT[[c.id for c in cards if c is not None]].sum(axis=0) * scale


def _poke_vec(p) -> np.ndarray:
    """Pokemon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
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
    v[base:base + FEAT_DIM] = _pool(p.tools)   # attached tools (Hero Cape etc.)
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
             status(me), status(op), stadium, _combat_features(state)]
    parts += [_poke_vec(p) for p in slots(me)]
    parts += [_poke_vec(p) for p in slots(op)]
    return np.concatenate(parts)


def _card_id_at(obs, area, index, player_index) -> int | None:
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

    [OptionType one-hot | features of the acted card | features of the TARGET
    Pokémon (for ATTACH/EVOLVE: what it's attached to / evolves onto) | target-is-active]
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

    # The action's target: which Pokémon this ATTACH/EVOLVE acts on.
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
