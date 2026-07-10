"""Observation / option encoders: game objects -> numpy vectors.

Same encodings as the old ``rl/encoders.py``, byte-for-byte (test-pinned),
with the combat lookahead now built on the public ``tcg.combat`` /
``tcg.library`` API instead of ``rl.combat``'s ``_private`` tables.

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

from cg.api import AreaType

from tcg.combat import best_damage, can_afford
from tcg.library import CARDS, known_attacks
from tcg.models import UNKNOWN_CARD
from tcg.pilot import card_at

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
N_COMBAT = 11        # combat-lookahead features (see combat_features)

BASIC_FIGHTING_ENERGY = 6  # card id; teacher's Mega Brave scales on discarded copies

# --- normalization scales ----------------------------------------------------
# Every raw quantity is squashed to roughly [0, 1] before it reaches the net.
# The values are part of the trained encoding: changing one changes every
# checkpoint's input distribution (parity-pinned in tests/test_encoders.py).
HP_SCALE = 340.0               # highest printed HP in the card pool
TURN_SCALE = 30.0              # typical game length upper bound
PRIZE_SCALE = 6.0              # prizes start at 6
DECK_SIZE_SCALE = 60.0         # decks are 60 cards
HAND_SIZE_SCALE = 15.0         # a very full hand
ENERGY_COUNT_SCALE = 5.0       # attached-energy count on one Pokémon
ENERGY_TYPE_STEP = 1.0 / 3.0   # per-type energy count increments
DISCARD_POOL_SCALE = 0.1       # discard piles get large; damp their pooled features
FIGHTING_DISCARD_SCALE = 10.0  # discarded Basic Fighting Energy count
MAX_PRIZE_COUNT = 3.0          # a mega-ex KO gives 3 prizes
READY_ATTACKERS_SCALE = 6.0    # active + 5 bench slots

# NOTE: a per-card-id "revealed opponent cards" feature was tried and reverted — a probe
# showed the opponent archetype is ALREADY ~98% identifiable from the pooled features below,
# so it was redundant and only added overfitting. See docs/DECISIONS.md 2026-07-08.

# Per-Pokémon-slot: card features + hp/maxHp/energy-count + energy-type counts + tools pool
SLOT_DIM = FEAT_DIM + 3 + N_ENERGY + FEAT_DIM
# globals + my hand pool + 2 discard pools + my discarded-fighting-energy count
# + status x2 + stadium + combat features + 6 of my slots + 6 opponent slots
STATE_DIM = (7 + FEAT_DIM + 2 * FEAT_DIM + 1 + 2 * N_STATUS + FEAT_DIM + N_COMBAT
             + 2 * (1 + N_BENCH) * SLOT_DIM)
# Where the M3 combat block sits inside encode_state's output — pre-M3
# checkpoints (bc_v1) were trained without it, and slicing this range out
# reconstructs their input encoding (see matchrunner's model loader).
COMBAT_START = 7 + 4 * FEAT_DIM + 1 + 2 * N_STATUS
COMBAT_SLICE = (COMBAT_START, COMBAT_START + N_COMBAT)
# option-type one-hot + acted card features + TARGET card features + target-is-active flag
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM + FEAT_DIM + 1


def has_affordable_attack(pokemon) -> bool:
    """Can this in-play Pokémon pay for any of its damaging attacks right now?"""
    return any(attack.damage > 0 and can_afford(pokemon.energies, attack.cost)
               for attack in known_attacks(CARDS.get(pokemon.id, UNKNOWN_CARD)))


def combat_features(state) -> np.ndarray:
    """The tactical quantities the rule experts reason over (damage / KO / prize race).

    The rule experts reason with hidden-state combat math (damage / KO / prize
    race) that BC couldn't imitate from raw board features; we compute the SAME
    quantities (via ``tcg.combat`` — available in training AND submission) and
    expose them to the net.
    """
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = me.active[0] if me.active else None
    opponent_active = opponent.active[0] if opponent.active else None

    my_damage = best_damage(my_active, opponent_active)
    opponent_damage = best_damage(opponent_active, my_active)
    my_ko = float(opponent_active is not None and my_damage >= opponent_active.hp)
    opponent_ko = float(my_active is not None and opponent_damage >= my_active.hp)
    one_energy_from_ko = float(
        opponent_active is not None and not my_ko
        and best_damage(my_active, opponent_active, extra_energy=1) >= opponent_active.hp)
    my_can_attack = float(
        best_damage(my_active, opponent_active) > 0
        or (my_active is not None and has_affordable_attack(my_active)))
    opponent_active_prize = (
        CARDS.get(opponent_active.id, UNKNOWN_CARD).prize_count / MAX_PRIZE_COUNT
        if opponent_active else 0.0)
    ready_attackers = sum(1 for pokemon in ([my_active] + list(me.bench))
                          if pokemon is not None and has_affordable_attack(pokemon))

    return np.array([
        my_damage / HP_SCALE, my_ko, my_can_attack, one_energy_from_ko,
        opponent_damage / HP_SCALE, opponent_ko,
        (len(me.prize) - len(opponent.prize)) / PRIZE_SCALE,  # prize race (negative = I'm ahead)
        opponent_active_prize,
        ready_attackers / READY_ATTACKERS_SCALE,
        (my_active.hp / max(1, my_active.maxHp)) if my_active else 0.0,
        (opponent_active.hp / max(1, opponent_active.maxHp)) if opponent_active else 0.0,
    ], dtype=np.float32)


def pooled_features(cards, scale: float = 1.0) -> np.ndarray:
    """Sum of card features over a list of Card objects (zeros when empty)."""
    if not cards:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    return FEAT[[card.id for card in cards if card is not None]].sum(axis=0) * scale


def pokemon_slot_vector(pokemon) -> np.ndarray:
    """Pokemon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
    vector = np.zeros(SLOT_DIM, dtype=np.float32)
    if pokemon is None:
        return vector
    vector[:FEAT_DIM] = FEAT[pokemon.id]
    base = FEAT_DIM
    vector[base:base + 3] = (pokemon.hp / HP_SCALE, pokemon.maxHp / HP_SCALE,
                             len(pokemon.energies) / ENERGY_COUNT_SCALE)
    base += 3
    for energy in pokemon.energies:            # energy TYPE counts, not just the total
        vector[base + int(energy)] += ENERGY_TYPE_STEP
    base += N_ENERGY
    vector[base:base + FEAT_DIM] = pooled_features(pokemon.tools)  # attached tools (Hero Cape etc.)
    return vector


def encode_state(state) -> np.ndarray:
    """cg.api.State -> STATE_DIM float32 vector."""
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]

    global_features = np.array([
        state.turn / TURN_SCALE,
        len(me.prize) / PRIZE_SCALE,
        len(opponent.prize) / PRIZE_SCALE,
        me.deckCount / DECK_SIZE_SCALE,
        opponent.handCount / HAND_SIZE_SCALE,
        float(state.energyAttached),
        float(state.supporterPlayed),
    ], dtype=np.float32)

    hand = pooled_features(me.hand)
    my_discard = pooled_features(me.discard, scale=DISCARD_POOL_SCALE)
    opponent_discard = pooled_features(opponent.discard, scale=DISCARD_POOL_SCALE)
    fighting_in_discard = np.array(
        [sum(1 for card in me.discard if card.id == BASIC_FIGHTING_ENERGY)
         / FIGHTING_DISCARD_SCALE],
        dtype=np.float32)

    def status(player) -> np.ndarray:
        return np.array([player.poisoned, player.burned, player.asleep,
                         player.paralyzed, player.confused], dtype=np.float32)

    stadium = pooled_features(state.stadium)

    def slots(player) -> list:
        active = player.active[0] if player.active else None
        bench = list(player.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        return [active] + bench

    parts = [global_features, hand, my_discard, opponent_discard, fighting_in_discard,
             status(me), status(opponent), stadium, combat_features(state)]
    parts += [pokemon_slot_vector(pokemon) for pokemon in slots(me)]
    parts += [pokemon_slot_vector(pokemon) for pokemon in slots(opponent)]
    return np.concatenate(parts)


LOOKING_AREA = 12  # AreaType.LOOKING — cards revealed by a search/look effect


def card_id_at(observation, area, index, player_index) -> int | None:
    """Card id at (area, index) for the given player, or None.

    ``tcg.pilot.card_at`` resolves the zones the pilot acts on; the encoder
    additionally reads two zones the pilot never selects from — the prize row
    and the "looking" reveal — handled here with the same None-on-miss rules.
    """
    if int(area) == int(AreaType.PRIZE):
        zone = observation.current.players[player_index].prize
    elif int(area) == LOOKING_AREA:
        zone = observation.current.looking
    else:
        card = card_at(observation, area, index, player_index)
        return card.id if card is not None else None
    try:
        card = zone[index]
        return card.id if card is not None else None
    except (TypeError, IndexError):  # hidden zone, or index out of range
        return None


def encode_option(option, observation) -> np.ndarray:
    """cg.api.Option -> OPTION_DIM float32 vector.

    [OptionType one-hot | features of the acted card | features of the TARGET
    Pokémon (for ATTACH/EVOLVE: what it's attached to / evolves onto) | target-is-active]
    """
    vector = np.zeros(OPTION_DIM, dtype=np.float32)
    vector[int(option.type)] = 1.0
    your_index = observation.current.yourIndex

    card_id = option.cardId
    if card_id is None and option.index is not None and option.area is not None:
        player_index = (option.playerIndex if option.playerIndex is not None
                        else your_index)
        card_id = card_id_at(observation, option.area, option.index, player_index)
    if card_id:
        vector[N_OPTION_TYPES:N_OPTION_TYPES + FEAT_DIM] = FEAT[card_id]

    # The action's target: which Pokémon this ATTACH/EVOLVE acts on.
    if option.inPlayArea is not None and option.inPlayIndex is not None:
        target_id = card_id_at(observation, option.inPlayArea, option.inPlayIndex,
                               your_index)
        if target_id:
            base = N_OPTION_TYPES + FEAT_DIM
            vector[base:base + FEAT_DIM] = FEAT[target_id]
        vector[-1] = float(int(option.inPlayArea) == int(AreaType.ACTIVE))
    return vector


def encode_context(context) -> np.ndarray:
    """SelectContext -> N_CONTEXTS one-hot."""
    vector = np.zeros(N_CONTEXTS, dtype=np.float32)
    vector[int(context)] = 1.0
    return vector
