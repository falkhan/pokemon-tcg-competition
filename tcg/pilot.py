"""The deck-agnostic rule-based pilot.

``make_generic_pilot(deck)`` returns a Kaggle agent that, on every decision,
scores each presented option by general TCG principles (develop the board,
load an attacker, take the KO, don't mill your own deck) plus the combat
lookahead in ``tcg.combat``, and answers with the highest-scoring options.

Behavior is identical to the old ``rl/generic_pilot.py``; the score tiers and
the reasoning behind them live in ``tcg.constants`` and docs/M6.md.
"""
from collections.abc import Callable
from types import SimpleNamespace

from cg.api import AreaType, OptionType, SelectContext, to_observation_class

from tcg import constants
from tcg.combat import (attack_available, best_damage, turns_to_first_ko,
                        turns_to_ready)
from tcg.library import (ATTACKS, CARDS, ENERGY_CARD_IDS,
                         HAND_DISCARD_TRAINER_IDS, POKEMON_CARD_IDS,
                         known_attacks)
from tcg.models import UNKNOWN_ATTACK, UNKNOWN_CARD


def make_generic_pilot(deck: list[int]) -> Callable[[dict], list[int]]:
    """Build a Kaggle agent (``obs_dict -> list[int]``) that pilots ``deck``."""
    def agent(obs_dict: dict) -> list[int]:
        observation = to_observation_class(obs_dict)
        if observation.select is None:  # game start: return the deck list
            return deck
        scores = [score_option(option, observation)
                  for option in observation.select.option]
        # Stable sort: equal scores keep their original (ascending) index order.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [int(i) for i in ranked[:observation.select.maxCount]]
    return agent


def score_option(option, observation) -> float:
    """Score one presented option; higher wins (see the ladder in constants)."""
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = me.active[0] if me.active else None
    opponent_active = opponent.active[0] if opponent.active else None

    option_type = option.type
    if option_type == OptionType.ATTACK:
        return score_attack(option, my_active, opponent_active, opponent.bench)
    if option_type == OptionType.ATTACH:
        return score_attach(option, observation)
    if option_type == OptionType.ABILITY:
        return constants.SCORE_ABILITY
    if option_type == OptionType.EVOLVE:
        return constants.SCORE_EVOLVE
    if option_type == OptionType.PLAY:
        return score_play(option, observation)
    if option_type == OptionType.RETREAT:
        return score_retreat(observation)
    if option_type == OptionType.CARD:
        return score_card(option, observation)
    return constants.FALLBACK_PRIORITY.get(option_type, 0)


def card_at(observation, area, index, player_index):
    """Resolve an option's (area, index) to the card/Pokémon it refers to.

    Returns None when the area is unknown, the zone is hidden, or the index is
    out of range.
    """
    player = observation.current.players[player_index]
    zones = {
        int(AreaType.DECK): observation.select.deck,
        int(AreaType.HAND): player.hand,
        int(AreaType.DISCARD): player.discard,
        int(AreaType.ACTIVE): player.active,
        int(AreaType.BENCH): player.bench,
        int(AreaType.STADIUM): observation.current.stadium,
    }
    zone = zones.get(int(area)) if area is not None else None
    try:
        return zone[index]
    except (TypeError, IndexError):  # zone is None / hidden, or index out of range
        return None


def attacker_quality(card_id: int) -> int:
    """Best raw attack damage a card can deal (0 if not an attacker)."""
    return max((attack.damage for attack in known_attacks(CARDS[card_id])),
               default=0)


def card_usefulness(card_id: int) -> int:
    """How valuable a card is to KEEP / fetch: attackers > energy > other."""
    if card_id in POKEMON_CARD_IDS:
        return constants.USEFULNESS_POKEMON_BASE + min(
            attacker_quality(card_id), constants.ATTACKER_QUALITY_CAP)
    if card_id in ENERGY_CARD_IDS:
        return constants.USEFULNESS_ENERGY
    return constants.USEFULNESS_OTHER


def _pokemon_in_hand(me) -> list:
    return [card for card in (me.hand or [])
            if card is not None and card.id in POKEMON_CARD_IDS]


def _my_pokemon_names(me, hand_pokemon) -> set:
    """Names of my Pokémon in play and in hand — the evolution-basis pool."""
    in_play = [pokemon for pokemon in (list(me.active or []) + list(me.bench or []))
               if pokemon is not None]
    return ({CARDS[p.id].name for p in in_play if p.id in CARDS}
            | {CARDS[c.id].name for c in hand_pokemon if c.id in CARDS})


def hand_holds_keepers(me) -> bool:
    """Any hand Pokémon worth protecting from a hand-discard trainer: a basic,
    an evolution whose basis is in play or hand, or a hard hitter even while
    momentarily dead (the deck's win-condition class — kaggle ep 85467275
    lost to Carmine discarding Mega Lucario ex twice)."""
    hand_pokemon = _pokemon_in_hand(me)
    basis_names = _my_pokemon_names(me, hand_pokemon)
    for card in hand_pokemon:
        data = CARDS.get(card.id)
        if data is None:
            return True                      # unknown card: keep, conservatively
        if data.basic:
            return True
        if data.evolves_from is not None and data.evolves_from in basis_names:
            return True
        if attacker_quality(card.id) >= constants.HAND_DISCARD_PROTECT_QUALITY:
            return True
    return False


def fetch_value(card, me) -> float:
    """KEEP/fetch value with evolution-line awareness (M7.5): dead evolutions
    sink, basics rise while the bench is empty or their evolution waits in
    hand (kaggle ep 85469339: Poké Pad fetched a basis-less Hariyama twice
    over live basics; the benched-out loss followed)."""
    base = card_usefulness(card.id)
    data = CARDS.get(card.id)
    if card.id not in POKEMON_CARD_IDS or data is None:
        return base
    hand_pokemon = _pokemon_in_hand(me)
    if data.evolves_from is not None:                     # evolution card
        if data.evolves_from not in _my_pokemon_names(me, hand_pokemon):
            return constants.FETCH_DEAD_EVOLUTION
        return base
    bonus = 0                                             # basic Pokémon
    if not any(pokemon is not None for pokemon in me.bench):
        bonus += constants.FETCH_EMPTY_BENCH_BASIC_BONUS
    if data.name is not None and any(
            CARDS.get(c.id) is not None and CARDS[c.id].evolves_from == data.name
            for c in hand_pokemon):
        bonus += constants.FETCH_ENABLES_EVOLUTION_BONUS
    return base + bonus


def attach_recipient_value(pokemon, me, opponent_active) -> float:
    """ATTACH_FROM: which of MY (usually benched) Pokémon receives an energy.

    Marginal value — the opposite of the promote ladder: a Pokémon whose best
    attack is already paid gains nothing from another energy (kaggle ep
    85607769: 5 energies on a 1-cost Solrock). A basic whose evolution waits
    in hand is charged FOR the evolution — attached energy survives evolving,
    so Riolu carrying 2 is a pre-charged Mega Brave."""
    evolution = next(
        (c for c in _pokemon_in_hand(me)
         if CARDS.get(c.id) is not None and CARDS.get(pokemon.id) is not None
         and CARDS[c.id].evolves_from is not None
         and CARDS[c.id].evolves_from == CARDS[pokemon.id].name),
        None)
    profile = (SimpleNamespace(id=evolution.id,
                               energies=list(getattr(pokemon, "energies", ()) or ()))
               if evolution is not None else pokemon)
    # M13 0a twin: own board ids gate CONDITIONAL_ATTACKS (see rl/generic_pilot).
    board_ids = {p.id for p in (list(me.active or []) + list(me.bench or []))
                 if p is not None}
    energy_gap = turns_to_ready(profile, opponent_active, board_ids)
    if energy_gap == 0:
        return constants.ATTACH_RECIPIENT_CHARGED
    quality = min(attacker_quality(profile.id), constants.ATTACKER_QUALITY_CAP)
    return (constants.ATTACH_RECIPIENT_BASE + quality
            - constants.PROMOTE_TURN_PENALTY
            * min(energy_gap - 1, constants.PROMOTE_TURNS_CAP))


def score_card(option, observation) -> float:
    """Card-selection contexts (SETUP/SWITCH/TO_HAND/DISCARD/...) — no coin flips."""
    state = observation.current
    opponent = state.players[1 - state.yourIndex]
    opponent_active = opponent.active[0] if opponent.active else None
    context = observation.select.context
    player_index = (option.playerIndex if option.playerIndex is not None
                    else state.yourIndex)
    card = card_at(observation, option.area, option.index, player_index)
    if card is None:
        return 0

    if context == SelectContext.ATTACH_FROM:
        return attach_recipient_value(card, state.players[state.yourIndex],
                                      opponent_active)
    if context in constants.PROMOTE_CONTEXTS:
        # Start / promote / bench MY best attacker; bonus if it hits right now,
        # minus a race term for each attach it still needs (M7.2b) — a charged
        # attacker beats an equal-damage uncharged one, non-attackers sink.
        # (hasattr guard: only in-play Pokémon have energies; hand cards don't.)
        quality = attacker_quality(card.id)
        can_hit_now = (opponent_active is not None and hasattr(card, "energies")
                       and best_damage(card, opponent_active) > 0)
        turns_gap = min(turns_to_ready(card, opponent_active),
                        constants.PROMOTE_TURNS_CAP)
        return (quality + (constants.PROMOTE_READY_BONUS if can_hit_now else 0)
                - constants.PROMOTE_TURN_PENALTY * turns_gap)
    if context in constants.KEEP_CONTEXTS:
        return fetch_value(card, state.players[state.yourIndex])
    if context in constants.DISCARD_CONTEXTS:  # discard the LEAST useful
        return -card_usefulness(card.id)
    if context in constants.TARGET_CONTEXTS:   # damage the highest-prize Pokémon
        prize = CARDS.get(card.id, UNKNOWN_CARD).prize_count
        return constants.TARGET_PRIZE_WEIGHT * prize
    return constants.SCORE_CARD_NEUTRAL


def opponent_board_harmless(opponent_active, opponent_bench=()) -> bool:
    """CLOSE MODE predicate (M7.2b): every opponent board Pokémon has a KNOWN
    card id and zero printed attack damage — they can never take a prize by KO.
    Unknown ids count as threats (conservative: the floor-test case only)."""
    board = ([opponent_active] if opponent_active is not None else [])
    board += [pokemon for pokemon in opponent_bench if pokemon is not None]
    return all(pokemon.id in CARDS and attacker_quality(pokemon.id) == 0
               for pokemon in board)


def score_attack(option, my_active, opponent_active, opponent_bench=()) -> float:
    if opponent_active is None:
        return constants.SCORE_ATTACK_NO_TARGET

    damage = ATTACKS.get(option.attackId, UNKNOWN_ATTACK).damage
    attack_type = CARDS[my_active.id].energy_type
    defender = CARDS[opponent_active.id]

    if defender.weakness is not None and defender.weakness == attack_type:
        damage *= constants.WEAKNESS_MULTIPLIER
    elif defender.resistance is not None and defender.resistance == attack_type:
        # Deliberately NOT floored at 0 (unlike combat.best_damage): a weak
        # resisted attack scores slightly below SCORE_CHIP_BASE. Test-pinned.
        damage -= constants.RESISTANCE_REDUCTION

    if damage >= opponent_active.hp:
        # KO takes a prize NOW — see the ladder rationale in tcg.constants.
        return constants.SCORE_KO_BASE + constants.KO_PRIZE_BONUS * defender.prize_count
    if opponent_board_harmless(opponent_active, opponent_bench):
        # CLOSE MODE: the race is won — attack every turn instead of milling
        # (the M6 floor-test self-deck fix; chip jumps above trainers).
        return constants.SCORE_CHIP_CLOSE_BASE + damage / constants.CHIP_DAMAGE_DIVISOR
    return constants.SCORE_CHIP_BASE + damage / constants.CHIP_DAMAGE_DIVISOR


def score_retreat(observation) -> float:
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = me.active[0] if me.active else None
    opponent_active = opponent.active[0] if opponent.active else None
    bench = [pokemon for pokemon in me.bench if pokemon is not None]
    if my_active is None or not bench:
        return constants.SCORE_RETREAT_NEVER

    bench_best_damage = max(attacker_quality(pokemon.id) for pokemon in bench)

    # 1) Promote a lethal attacker from the bench if the active can't KO.
    if (opponent_active is not None
            and best_damage(my_active, opponent_active) < opponent_active.hp):
        if any(best_damage(pokemon, opponent_active) >= opponent_active.hp
               for pokemon in bench):
            return constants.SCORE_RETREAT_PROMOTE_LETHAL

    # 2) Escape a KO.
    in_danger = (opponent_active is not None
                 and best_damage(opponent_active, my_active) >= my_active.hp)
    if in_danger and bench_best_damage > 0:
        return constants.SCORE_RETREAT_ESCAPE_KO

    # 3) Otherwise low; ~0 when healthy, a bit higher if a strong bench
    #    attacker wants in.
    hp_fraction = my_active.hp / max(1, my_active.maxHp)
    if hp_fraction > constants.HEALTHY_HP_FRACTION:
        return constants.SCORE_RETREAT_NEVER
    return (constants.SCORE_RETREAT_HURT_BASE
            + bench_best_damage // constants.RETREAT_BENCH_DAMAGE_DIVISOR)


def score_play(option, observation) -> float:
    my_index = observation.current.yourIndex
    me = observation.current.players[my_index]
    # Engine quirk (found 2026-07-12, kaggle ep 85467275 + local repro): PLAY
    # options carry NO `area` field — the index is always a hand index. Without
    # this default every play resolved to None -> flat 2000, so the whole play
    # tier below (Pokémon vs trainer, taper, deck-out guard) never fired in a
    # real game — only in tests, whose synthetic options set area explicitly.
    area = option.area if option.area is not None else AreaType.HAND
    card = card_at(observation, area, option.index, my_index)

    if card is None:
        return constants.SCORE_PLAY_UNRESOLVED_CARD
    if card.id in POKEMON_CARD_IDS:
        if not any(pokemon is not None for pokemon in me.bench):
            return constants.SCORE_PLAY_POKEMON_EMPTY_BENCH
        return constants.SCORE_PLAY_POKEMON

    if card.id in HAND_DISCARD_TRAINER_IDS and hand_holds_keepers(me):
        return constants.SCORE_HAND_DISCARD_BLOCKED

    has_board = ((bool(me.active) and me.active[0] is not None)
                 or any(pokemon for pokemon in me.bench))
    if not has_board:
        return constants.SCORE_PLAY_TRAINER_NO_BOARD

    # Trainer: value on NEED, not flat — the anti-deck-out fix (see constants).
    hand_size = me.handCount or 0
    deck_size = me.deckCount or 0
    if deck_size <= constants.DECKOUT_RESERVE_CARDS:
        return constants.SCORE_PLAY_NEAR_DECKOUT
    tapered = (constants.SCORE_TRAINER_BASE - constants.TRAINER_HAND_TAPER
               * max(0, hand_size - constants.COMFORTABLE_HAND_SIZE))
    return max(constants.SCORE_TRAINER_FLOOR, tapered)


def score_attach(option, observation) -> float:
    my_index = observation.current.yourIndex
    opponent = observation.current.players[1 - my_index]
    opponent_active = opponent.active[0] if opponent.active else None

    target = card_at(observation, option.inPlayArea, option.inPlayIndex, my_index)
    if target is None:
        return constants.SCORE_ATTACH_NO_TARGET

    is_active = option.inPlayArea == AreaType.ACTIVE
    # M14 twin (see rl/generic_pilot.score_attach): CONDITIONAL_ATTACKS gate
    # this path too — a conditional attacker without its requirement is not
    # an attacker worth charging.
    my_state = observation.current.players[my_index]
    my_board_ids = {p.id for p in (list(my_state.active or [])
                                   + list(my_state.bench or []))
                    if p is not None}

    # 1) Lookahead: does this attach UNBLOCK a KO on the opponent's active?
    if opponent_active is not None:
        damage_now = best_damage(target, opponent_active, extra_energy=0,
                                 board_ids=my_board_ids)
        damage_after = best_damage(target, opponent_active, extra_energy=1,
                                   board_ids=my_board_ids)
        if damage_now < opponent_active.hp <= damage_after:
            return (constants.SCORE_ATTACH_UNBLOCKS_KO_ACTIVE if is_active
                    else constants.SCORE_ATTACH_UNBLOCKS_KO_BENCH)

    # 2) Otherwise: is the target a real attacker that still needs energy?
    # (iterate ids, not Attack records — the record carries no id, and
    # attack_available needs one)
    damaging_attacks = [(ATTACKS[a].damage, len(ATTACKS[a].cost))
                        for a in CARDS[target.id].attack_ids
                        if a in ATTACKS and ATTACKS[a].damage > 0
                        and attack_available(a, my_board_ids)]
    if not damaging_attacks:
        return constants.SCORE_ATTACH_NON_ATTACKER

    if turns_to_ready(target, opponent_active, board_ids=my_board_ids) == 0:
        # The BEST attack is charged (M7.2b — was the cheapest, which stopped
        # charging a 2-cost 270 attacker after its 1-cost 130 was paid).
        return constants.SCORE_ATTACH_ALREADY_LOADED

    best_dmg = max(damage for damage, _ in damaging_attacks)
    bonus = (min(best_dmg, constants.ATTACH_DAMAGE_BONUS_CAP)
             // constants.ATTACH_DAMAGE_BONUS_DIVISOR)

    # 3) Race math (M7.2b): charge THE ONE attacker that closes first. The
    #    target's own turns-to-first-KO must match the board minimum; ties
    #    bonus every tied target and self-commit after the first attach
    #    (the winner's energy gap drops, making it strictly unique).
    #    The BENCH tier additionally requires an attack-ready ACTIVE — banking
    #    on a benched closer while the active can neither attack nor pay its
    #    retreat starves the whole board and mills the deck (the measured
    #    floor-test failure, docs/M7.md 2026-07-09).
    if opponent_active is not None:
        me = observation.current.players[my_index]
        my_active = me.active[0] if me.active and me.active[0] is not None else None
        board = ([my_active] if my_active is not None else [])
        board += [pokemon for pokemon in me.bench if pokemon is not None]
        mine = turns_to_first_ko(target, opponent_active)
        if mine < constants.UNREACHABLE_TURNS and board and \
                mine <= min(turns_to_first_ko(p, opponent_active) for p in board):
            if is_active:
                return constants.SCORE_ATTACH_RACE_CLOSER_ACTIVE + bonus
            if my_active is not None and turns_to_ready(my_active, opponent_active) == 0:
                return constants.SCORE_ATTACH_RACE_CLOSER_BENCH + bonus

    base = (constants.SCORE_ATTACH_ACTIVE_BASE if is_active
            else constants.SCORE_ATTACH_BENCH_BASE)
    return base + bonus
