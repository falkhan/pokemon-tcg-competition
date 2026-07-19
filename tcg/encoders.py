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

from cg.api import AreaType, OptionType

from tcg.combat import attack_available, best_damage, can_afford
from tcg.library import ATTACKS, CARDS, known_attacks
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


# --- Encoders v2 (M7.3) — side-by-side with v1; v1 stays byte-identical -----
# v2 returns (numeric, ids): the numeric vector extends v1 with the k-turn race
# block (deferred from M7.2b) and two deck-context pools, while the id vector
# carries card ids at fixed sites for LEARNABLE embeddings — the 36 features
# are nearly blind for trainers, embeddings let a multi-deck pilot learn
# per-card behavior from data (M7-plan §3.1b). Id 0 = "no card" padding,
# aligned with FEAT row 0.
EMBED_DIM = 16
N_CARD_IDS = FEAT.shape[0]          # 1268: ids 1..1267 + padding row 0
N_STATE_IDS = 2 * (1 + N_BENCH)     # my/opp active + bench card ids
N_OPTION_IDS = 2                    # acted card + target card
N_RACE = 8                          # see race_features
RACE_TURN_CAP = 10.0                # race turns normalized /10, capped
STATE_V2_DIM = STATE_DIM + N_RACE + 2 * FEAT_DIM
OPTION_V2_DIM = OPTION_DIM
# M16 option-identity block — twin of rl/encoders.py (see its comment): PLAY
# options resolve their hand card, ATTACK options encode their attack's
# numbers, NUMBER options their count; appended so the v1 slice is unchanged.
N_OPTION_EXTRA = 4   # [atk dmg/300, atk cost/5, atk eff-dmg vs opp active/300, number/10]
OPTION_V3_DIM = OPTION_DIM + N_OPTION_EXTRA


def race_features(state) -> np.ndarray:
    """k-turn prize-race block on the M7.2b combat primitives — the M3 combat
    features are 1-turn only; these are turns-to-ready / turns-to-first-KO for
    both boards plus the race delta (M7-plan §3b L1)."""
    from tcg.combat import turns_to_first_ko, turns_to_ready

    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]

    def board(player_state):
        active = player_state.active[0] if player_state.active else None
        return ([p for p in [active] + list(player_state.bench) if p is not None],
                active)

    my_board, my_active = board(me)
    op_board, op_active = board(opponent)

    def norm(turns) -> float:
        return min(float(turns), RACE_TURN_CAP) / RACE_TURN_CAP

    my_ttfk = min((turns_to_first_ko(p, op_active) for p in my_board),
                  default=RACE_TURN_CAP) if op_active is not None else RACE_TURN_CAP
    op_ttfk = min((turns_to_first_ko(p, my_active) for p in op_board),
                  default=RACE_TURN_CAP) if my_active is not None else RACE_TURN_CAP
    return np.array([
        norm(turns_to_ready(my_active, op_active)) if my_active else 1.0,
        norm(turns_to_first_ko(my_active, op_active)) if my_active and op_active else 1.0,
        norm(my_ttfk),
        sum(1 for p in my_board if turns_to_ready(p, op_active) == 0) / 6.0,
        norm(turns_to_ready(op_active, my_active)) if op_active else 1.0,
        norm(turns_to_first_ko(op_active, my_active)) if op_active and my_active else 1.0,
        norm(op_ttfk),
        (min(float(op_ttfk), RACE_TURN_CAP) - min(float(my_ttfk), RACE_TURN_CAP))
        / RACE_TURN_CAP,                                    # >0: I win the race
    ], dtype=np.float32)


def deck_pools(state, my_deck: list[int]) -> np.ndarray:
    """Deck-context pools (M7-plan §3.1b): FEAT sums of my FULL 60-card list and
    of my REMAINING deck (list minus hand/board/discard — all observable; the 6
    prized cards stay in "remaining" since which ones is hidden). Pooled, NOT a
    per-id count vector — the reverted-feature lesson. Discard-pool 0.1 scale."""
    from collections import Counter
    me = state.players[state.yourIndex]
    remaining = Counter(my_deck)
    seen = [c.id for c in list(me.hand) + list(me.discard) if c is not None]
    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is not None:
            seen.append(p.id)
            seen.extend(t.id for t in p.tools if t is not None)
    for cid in seen:
        if remaining[cid] > 0:
            remaining[cid] -= 1
    full = FEAT[list(my_deck)].sum(axis=0) * 0.1
    rest = (sum((FEAT[cid] * n for cid, n in remaining.items() if n > 0),
                np.zeros(FEAT_DIM)) * 0.1)
    return np.concatenate([full, rest]).astype(np.float32)


def board_ids(state) -> np.ndarray:
    """Card ids of the 12 board slots (my/opp active + bench), 0-padded."""
    ids = []
    for player_state in (state.players[state.yourIndex],
                         state.players[1 - state.yourIndex]):
        active = player_state.active[0] if player_state.active else None
        bench = list(player_state.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        ids += [p.id if p is not None else 0 for p in [active] + bench]
    return np.array(ids, dtype=np.int32)


def encode_state_v2(state, my_deck: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(numeric STATE_V2_DIM f32, board ids N_STATE_IDS i32). Needs my 60-card
    list — observations don't carry it; the pilot closes over its own deck."""
    numeric = np.concatenate([encode_state(state), race_features(state),
                              deck_pools(state, my_deck)])
    return numeric.astype(np.float32), board_ids(state)


def attack_extra(attack_id, observation) -> np.ndarray:
    """[printed dmg/300, cost size/5, effective dmg vs opp active/300] — twin
    of rl/encoders.py _attack_extra; weakness/resistance math mirrors
    tcg.combat.charged_best (change BOTH if the rules change)."""
    vector = np.zeros(3, dtype=np.float32)
    attack = ATTACKS.get(attack_id)
    if attack is None:
        return vector
    vector[0] = attack.damage / 300.0
    vector[1] = len(attack.cost) / 5.0
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    attacker = me.active[0] if me.active and me.active[0] is not None else None
    target = (opponent.active[0]
              if opponent.active and opponent.active[0] is not None else None)
    if attacker is None or target is None or attack.damage <= 0:
        return vector
    board = {p.id for p in [attacker] + list(me.bench or []) if p is not None}
    if not attack_available(attack_id, board):
        return vector
    target_card = CARDS.get(target.id, UNKNOWN_CARD)
    attacker_type = CARDS.get(attacker.id, UNKNOWN_CARD).energy_type
    effective = attack.damage
    if target_card.weakness is not None and int(target_card.weakness) == attacker_type:
        effective *= 2
    elif (target_card.resistance is not None
          and int(target_card.resistance) == attacker_type):
        effective = max(0, effective - 30)
    vector[2] = effective / 300.0
    return vector


def _my_pokemon_at(observation, in_play_area, in_play_index):
    """Twin of rl/encoders.py _my_poke_at — the in-play Pokémon OBJECT."""
    me = observation.current.players[observation.current.yourIndex]
    zone = {int(AreaType.ACTIVE): me.active,
            int(AreaType.BENCH): me.bench}.get(int(in_play_area))
    if zone is None or in_play_index is None or in_play_index >= len(zone):
        return None
    return zone[in_play_index]


def attach_extra(option, observation) -> np.ndarray:
    """[target attached-energy/5, energy gap/5, saturated flag] — M19 twin of
    rl/encoders.py _attach_extra (change BOTH)."""
    from tcg.combat import turns_to_ready

    vector = np.zeros(3, dtype=np.float32)
    target = _my_pokemon_at(observation, option.inPlayArea, option.inPlayIndex)
    if target is None:
        return vector
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    opponent_active = (opponent.active[0]
                       if opponent.active and opponent.active[0] is not None else None)
    board_ids = {p.id for p in list(me.active or []) + list(me.bench or [])
                 if p is not None}
    gap = turns_to_ready(target, opponent_active, board_ids)
    vector[0] = min(len(target.energies or ()), 5) / 5.0
    vector[1] = min(gap, 5) / 5.0
    vector[2] = float(gap == 0)
    return vector


def retreat_extra(observation) -> np.ndarray:
    """[active damage fraction, active prizes-on-KO/3, bench-ready flag] —
    M19 twin of rl/encoders.py _retreat_extra (change BOTH)."""
    from tcg.combat import turns_to_ready

    vector = np.zeros(3, dtype=np.float32)
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    active = me.active[0] if me.active and me.active[0] is not None else None
    if active is None:
        return vector
    opponent_active = (opponent.active[0]
                       if opponent.active and opponent.active[0] is not None else None)
    vector[0] = 1.0 - active.hp / max(1, active.maxHp)
    vector[1] = CARDS.get(active.id, UNKNOWN_CARD).prize_count / 3.0
    vector[2] = float(any(turns_to_ready(pokemon, opponent_active) == 0
                          for pokemon in me.bench or [] if pokemon is not None))
    return vector


def encode_option_v2(option, observation) -> tuple[np.ndarray, np.ndarray]:
    """(numeric OPTION_V3_DIM f32, [acted_id, target_id] i32, 0 = none).

    M16 twin of rl/encoders.py encode_option_v2: PLAY options resolve their
    hand card (FEAT block + embedding id), the appended block encodes attack
    identity and NUMBER counts; M19 adds ATTACH energy-sufficiency and
    RETREAT utility in the same 3 slots (mutually exclusive types).
    Pre-M16 checkpoints: encode_option_v2_legacy."""
    numeric = np.zeros(OPTION_V3_DIM, dtype=np.float32)
    numeric[:OPTION_DIM] = encode_option(option, observation)
    your_index = observation.current.yourIndex
    card_id = option.cardId
    if card_id is None and option.index is not None:
        if option.area is not None:
            player_index = (option.playerIndex if option.playerIndex is not None
                            else your_index)
            card_id = card_id_at(observation, option.area, option.index,
                                 player_index)
        elif option.type == OptionType.PLAY:
            # PLAY carries only a hand index; v1 stays blank by design.
            card_id = card_id_at(observation, AreaType.HAND, option.index,
                                 your_index)
            if card_id:
                numeric[N_OPTION_TYPES:N_OPTION_TYPES + FEAT_DIM] = FEAT[card_id]
    target_id = None
    if option.inPlayArea is not None and option.inPlayIndex is not None:
        target_id = card_id_at(observation, option.inPlayArea, option.inPlayIndex,
                               your_index)
    if option.type == OptionType.ATTACK:
        numeric[OPTION_DIM:OPTION_DIM + 3] = attack_extra(option.attackId,
                                                          observation)
    elif option.type == OptionType.ATTACH and option.inPlayArea is not None:
        numeric[OPTION_DIM:OPTION_DIM + 3] = attach_extra(option, observation)
    elif option.type == OptionType.RETREAT:
        numeric[OPTION_DIM:OPTION_DIM + 3] = retreat_extra(observation)
    elif getattr(option, "number", None) is not None:
        numeric[OPTION_DIM + 3] = min(float(option.number), 10.0) / 10.0
    ids = np.array([card_id or 0, target_id or 0], dtype=np.int32)
    return numeric, ids


def encode_option_v2_legacy(option, observation) -> tuple[np.ndarray, np.ndarray]:
    """Pre-M16 v2 encoding (no option-identity block) — twin of
    rl/encoders.py encode_option_v2_legacy; pinned checkpoints only."""
    your_index = observation.current.yourIndex
    card_id = option.cardId
    if card_id is None and option.index is not None and option.area is not None:
        player_index = (option.playerIndex if option.playerIndex is not None
                        else your_index)
        card_id = card_id_at(observation, option.area, option.index, player_index)
    target_id = None
    if option.inPlayArea is not None and option.inPlayIndex is not None:
        target_id = card_id_at(observation, option.inPlayArea, option.inPlayIndex,
                               your_index)
    ids = np.array([card_id or 0, target_id or 0], dtype=np.int32)
    return encode_option(option, observation), ids
