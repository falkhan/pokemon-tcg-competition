"""Old-vs-new behavioral parity.

Runs the old ``rl.combat`` / ``rl.generic_pilot`` and the new ``tcg`` package
against the same fake engine and asserts identical outputs — tables, damage
math, every scorer branch, and the end-to-end agent (including tie-breaking
and maxCount truncation). This is the proof that the refactor is behavior-
preserving; live-game validation needs the real engine (see docs/M6.md).
"""
import itertools

from rl import combat as old_combat
from rl import generic_pilot as old_pilot
from tcg import combat as new_combat
from tcg import library
from tcg import pilot as new_pilot
from tests import builders as b
from tests.fake_cg import AreaType, OptionType, SelectContext

FIGHTING, WATER, PSYCHIC, COLORLESS = 6, 3, 5, 0
POKEMON_IDS = sorted(library.POKEMON_CARD_IDS)


def test_card_table_parity():
    assert set(library.CARDS) == set(old_combat._CARD)
    for card_id, card in library.CARDS.items():
        weakness, resistance, energy_type, attacks, prize = old_combat._CARD[card_id]
        assert card.weakness == (None if weakness is None else int(weakness))
        assert card.resistance == (None if resistance is None else int(resistance))
        assert card.energy_type == energy_type
        assert list(card.attack_ids) == list(attacks)
        assert card.prize_count == prize


def test_attack_table_parity():
    assert set(library.ATTACKS) == set(old_combat._ATK)
    for attack_id, attack in library.ATTACKS.items():
        damage, cost = old_combat._ATK[attack_id]
        assert (attack.damage, attack.cost) == (damage, cost)


def test_card_id_set_parity():
    assert library.POKEMON_CARD_IDS == old_pilot._IS_POKEMON
    assert library.ENERGY_CARD_IDS == old_pilot._IS_ENERGY


def test_can_afford_parity():
    loadouts = [(), (COLORLESS,), (FIGHTING,), (WATER,), (FIGHTING, FIGHTING),
                (FIGHTING, WATER), (FIGHTING, COLORLESS), (WATER, WATER, PSYCHIC)]
    costs = [attack.cost for attack in library.ATTACKS.values()]
    costs += [(COLORLESS,), (COLORLESS, COLORLESS), (FIGHTING, COLORLESS, COLORLESS)]
    for energies, cost in itertools.product(loadouts, costs):
        assert old_combat._can_afford(list(energies), cost) \
            == new_combat.can_afford(list(energies), cost), (energies, cost)


def test_best_damage_parity():
    loadouts = [(), (FIGHTING,), (FIGHTING, FIGHTING), (WATER,),
                (FIGHTING, WATER), (PSYCHIC,), (COLORLESS, COLORLESS)]
    ids = POKEMON_IDS + [12345]  # incl. an id unknown to the card table
    for attacker_id, target_id, energies, extra in itertools.product(
            ids, ids, loadouts, (0, 1)):
        attacker = b.pokemon(attacker_id, energies=energies)
        target = b.pokemon(target_id)
        assert old_combat._best_damage(attacker, target, extra) \
            == new_combat.best_damage(attacker, target, extra), \
            (attacker_id, target_id, energies, extra)
    assert old_combat._best_damage(None, b.pokemon(1)) \
        == new_combat.best_damage(None, b.pokemon(1)) == 0
    assert old_combat._best_damage(b.pokemon(1), None) \
        == new_combat.best_damage(b.pokemon(1), None) == 0


def _attack_scenarios():
    """(option, observation) pairs covering score_attack's branches."""
    attackers = [b.pokemon(1, energies=[FIGHTING]),
                 b.pokemon(3, energies=[FIGHTING, FIGHTING]),
                 b.pokemon(10, energies=[FIGHTING]),
                 b.pokemon(4, energies=[PSYCHIC])]
    defenders = [b.pokemon(2, hp=30), b.pokemon(2, hp=200), b.pokemon(5, hp=50),
                 b.pokemon(3, hp=100), b.pokemon(1, hp=25), None]
    attack_ids = [101, 102, 103, 104, 105, 106, 107, 999]  # 999: unknown attack
    for attacker, defender, attack_id in itertools.product(
            attackers, defenders, attack_ids):
        obs = b.observation(me=b.player(active=attacker),
                            opponent=b.player(active=defender))
        yield b.option(OptionType.ATTACK, attack_id=attack_id), obs


def _attach_scenarios():
    targets = [
        b.pokemon(3, energies=[FIGHTING]),  # attach unblocks the 270 KO
        b.pokemon(1),                        # real attacker, needs energy
        b.pokemon(1, energies=[FIGHTING]),   # already loaded
        b.pokemon(5),                        # no damaging attack
        b.pokemon(9),                        # attacks missing from the table
    ]
    defenders = [b.pokemon(5, hp=100), b.pokemon(2, hp=300), None]
    for target, defender, area in itertools.product(
            targets, defenders, (AreaType.ACTIVE, AreaType.BENCH)):
        me = (b.player(active=target) if area == AreaType.ACTIVE
              else b.player(active=b.pokemon(5), bench=[target]))
        obs = b.observation(me=me, opponent=b.player(active=defender))
        yield b.option(OptionType.ATTACH, in_play_area=area, in_play_index=0), obs
    # unresolvable target
    obs = b.observation(me=b.player(active=b.pokemon(1)))
    yield b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE,
                   in_play_index=7), obs


def _play_scenarios():
    hands = [[b.hand_card(1)], [b.hand_card(6)], [b.hand_card(7)]]
    boards = [None, b.pokemon(1)]
    for hand, active, hand_count, deck_count in itertools.product(
            hands, boards, (0, 5, 7, 20, None), (3, 6, 7, 30)):
        me = b.player(active=active, hand=hand, hand_count=hand_count,
                      deck_count=deck_count)
        obs = b.observation(me=me)
        yield b.option(OptionType.PLAY, area=AreaType.HAND, index=0), obs
        yield b.option(OptionType.PLAY, area=AreaType.HAND, index=9), obs  # miss


def _retreat_scenarios():
    actives = [b.pokemon(5, hp=40, max_hp=100), b.pokemon(1, hp=100),
               b.pokemon(1, hp=60, max_hp=100, energies=[FIGHTING])]
    benches = [[], [b.pokemon(1)], [b.pokemon(1, energies=[FIGHTING, FIGHTING])],
               [b.pokemon(5)]]
    defenders = [None, b.pokemon(5, hp=100), b.pokemon(1, hp=999, energies=[FIGHTING])]
    for active, bench, defender in itertools.product(actives, benches, defenders):
        me = b.player(active=active, bench=bench)
        obs = b.observation(me=me, opponent=b.player(active=defender))
        yield b.option(OptionType.RETREAT), obs


def _card_scenarios():
    contexts = [
        SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.TO_ACTIVE, SelectContext.SWITCH, SelectContext.TO_FIELD,
        SelectContext.TO_BENCH, SelectContext.ATTACH_FROM,   # promote
        SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.NOT_MOVE,  # keep
        SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,         # discard
        SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY, SelectContext.EFFECT_TARGET,  # target
        SelectContext.MAIN,                                  # neutral
    ]
    me = b.player(hand=[b.hand_card(1), b.hand_card(6), b.hand_card(7)],
                  bench=[b.pokemon(1, energies=[FIGHTING])])
    opponent = b.player(active=b.pokemon(3, hp=100), bench=[b.pokemon(5)])
    picks = [
        b.option(OptionType.CARD, area=AreaType.HAND, index=0),
        b.option(OptionType.CARD, area=AreaType.HAND, index=1),
        b.option(OptionType.CARD, area=AreaType.HAND, index=2),
        b.option(OptionType.CARD, area=AreaType.HAND, index=9),      # unresolved
        b.option(OptionType.CARD, area=AreaType.BENCH, index=0),     # in-play, ready
        b.option(OptionType.CARD, area=AreaType.ACTIVE, index=0, player_index=1),
        b.option(OptionType.CARD, area=AreaType.BENCH, index=0, player_index=1),
        b.option(OptionType.CARD, area=None, index=0),
    ]
    for context, pick in itertools.product(contexts, picks):
        yield pick, b.observation(me=me, opponent=opponent, context=context)


def _misc_scenarios():
    obs = b.observation()
    for option_type in OptionType:
        if option_type in (OptionType.ATTACK, OptionType.ATTACH, OptionType.PLAY,
                           OptionType.RETREAT, OptionType.CARD):
            continue
        yield b.option(option_type), obs


def all_scenarios():
    yield from _attack_scenarios()
    yield from _attach_scenarios()
    yield from _play_scenarios()
    yield from _retreat_scenarios()
    yield from _card_scenarios()
    yield from _misc_scenarios()


def test_score_option_parity():
    count = 0
    for option, obs in all_scenarios():
        assert old_pilot.score_option(option, obs) \
            == new_pilot.score_option(option, obs), (option, obs.select.context)
        count += 1
    assert count > 400  # the sweep really covered the branch grid


def test_agent_parity_on_deck_return():
    deck = list(range(60))
    old_agent = old_pilot.make_generic_pilot(deck)
    new_agent = new_pilot.make_generic_pilot(deck)
    first_call = b.observation(select=False)
    assert old_agent(first_call) == new_agent(first_call) == deck


def test_agent_parity_on_selections():
    deck = list(range(60))
    old_agent = old_pilot.make_generic_pilot(deck)
    new_agent = new_pilot.make_generic_pilot(deck)

    me = b.player(active=b.pokemon(1, energies=[FIGHTING]),
                  bench=[b.pokemon(3, energies=[FIGHTING])],
                  hand=[b.hand_card(7), b.hand_card(1)])
    opponent = b.player(active=b.pokemon(2, hp=40))
    options = [
        b.option(OptionType.END),
        b.option(OptionType.ATTACK, attack_id=101),
        b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0),
        b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
        b.option(OptionType.PLAY, area=AreaType.HAND, index=1),
        b.option(OptionType.ABILITY),
        b.option(OptionType.RETREAT),
        b.option(OptionType.YES),  # ties with the YES below
        b.option(OptionType.YES),
    ]
    for max_count in range(1, len(options) + 1):
        obs = b.observation(me=me, opponent=opponent, options=options,
                            max_count=max_count)
        assert old_agent(obs) == new_agent(obs), max_count
