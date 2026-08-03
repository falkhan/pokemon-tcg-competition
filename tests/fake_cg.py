"""A fake ``cg`` engine, installed into sys.modules on import.

The real ``cg/`` bindings are gitignored (downloaded from Kaggle) and absent in
CI/dev environments, so tests run both the old ``rl`` pilot and the new ``tcg``
package against this stub. Enum values follow docs/pilot-cheatsheet.md; the
SelectContext members the cheatsheet doesn't document get synthetic distinct
values (harmless: old and new code compare against the same enum objects).

The synthetic card pool deliberately covers: typed / colorless / mixed attack
costs, weakness and resistance present and absent, all three prize tiers
(basic / ex / mega-ex), energy and trainer cards, a zero-damage attack, and a
card whose attack list contains an id missing from the attack table.
"""
import sys
import types
from enum import IntEnum
from types import SimpleNamespace


class EnergyType(IntEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10
    TEAM_ROCKET = 11


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    LOOKING = 12  # cards revealed by a search/look effect


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14


class CardType(IntEnum):
    POKEMON = 0
    TRAINER = 1
    BASIC_ENERGY = 2
    SPECIAL_ENERGY = 3
    # M42: STADIUM at its REAL value (4) so the stadium forensics run in CI.
    # The members above do NOT match the engine — the real enum is POKEMON 0,
    # ITEM 1, TOOL 2, SUPPORTER 3, STADIUM 4, BASIC_ENERGY 5, SPECIAL_ENERGY 6,
    # so this stub's BASIC_ENERGY/SPECIAL_ENERGY collide with the engine's
    # TOOL/SUPPORTER. Nothing compares these across the boundary today, and
    # renumbering would touch every test that names them, so only the
    # non-colliding member is added here. Recorded in docs/M42.md.
    STADIUM = 4


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_HAND = 7
    DISCARD = 8
    ATTACH_FROM = 21
    # Synthetic values (undocumented in the cheatsheet, distinct is all that matters):
    TO_FIELD = 30
    TO_BENCH = 31
    LOOK = 32
    NOT_MOVE = 33
    TO_DECK = 34
    TO_DECK_BOTTOM = 35
    DISCARD_CARD_OR_ATTACHED_CARD = 36
    DAMAGE = 37
    DAMAGE_COUNTER = 38
    DAMAGE_COUNTER_ANY = 39
    EFFECT_TARGET = 40


def _attack(attack_id, damage, energies):
    return SimpleNamespace(attackId=attack_id, damage=damage, energies=list(energies))


def _card(card_id, card_type, *, weakness=None, resistance=None,
          energy_type=EnergyType.COLORLESS, attacks=(), ex=False, mega_ex=False):
    return SimpleNamespace(cardId=card_id, cardType=card_type, weakness=weakness,
                           resistance=resistance, energyType=energy_type,
                           attacks=list(attacks), ex=ex, megaEx=mega_ex)


MISSING_ATTACK_ID = 999  # on cards' attack lists but absent from the attack table

_ATTACKS = [
    _attack(101, 50, [EnergyType.FIGHTING]),
    _attack(102, 120, [EnergyType.FIGHTING, EnergyType.COLORLESS]),
    _attack(103, 0, []),                                     # zero-damage effect attack
    _attack(104, 30, [EnergyType.COLORLESS]),
    _attack(105, 270, [EnergyType.FIGHTING, EnergyType.FIGHTING]),
    _attack(106, 20, [EnergyType.WATER]),
    _attack(107, 20, [EnergyType.COLORLESS]),  # weak hit: resisted, it goes below 0
]

_CARDS = [
    # 1: basic Fighting attacker; attack list includes an id missing from the table
    _card(1, CardType.POKEMON, weakness=EnergyType.PSYCHIC,
          energy_type=EnergyType.FIGHTING, attacks=[101, 102, MISSING_ATTACK_ID]),
    # 2: Water ex that RESISTS Fighting (exercises resistance math + 2-prize KO)
    _card(2, CardType.POKEMON, weakness=EnergyType.LIGHTNING,
          resistance=EnergyType.FIGHTING, energy_type=EnergyType.WATER,
          attacks=[106, 103], ex=True),
    # 3: mega-ex Fighting heavy hitter (3-prize KO)
    _card(3, CardType.POKEMON, weakness=EnergyType.PSYCHIC,
          energy_type=EnergyType.FIGHTING, attacks=[105], mega_ex=True),
    # 4: Psychic attacker with an all-colorless cost (hits 1 and 3 for weakness)
    _card(4, CardType.POKEMON, energy_type=EnergyType.PSYCHIC, attacks=[104]),
    # 5: Pokémon with no damaging attack
    _card(5, CardType.POKEMON, energy_type=EnergyType.GRASS, attacks=[103]),
    # 6/8: energy cards, 7: trainer
    _card(6, CardType.BASIC_ENERGY),
    _card(7, CardType.TRAINER),
    _card(8, CardType.SPECIAL_ENERGY),
    # 9: Pokémon whose only attack id is missing from the attack table
    _card(9, CardType.POKEMON, energy_type=EnergyType.FIGHTING,
          attacks=[MISSING_ATTACK_ID]),
    # 10: weak Fighting attacker — its 20 damage into card 2's resistance (-30)
    # goes negative, pinning the floored-vs-unfloored asymmetry
    _card(10, CardType.POKEMON, energy_type=EnergyType.FIGHTING, attacks=[107]),
]


def all_attack():
    return list(_ATTACKS)


def all_card_data():
    return list(_CARDS)


def to_observation_class(obs_dict):
    # Tests pass prebuilt SimpleNamespace observations straight through.
    return obs_dict


# Determinized-search API (rl/mcts.py). The stub only provides the names so
# the modules import; tests monkeypatch the module-bound names
# (e.g. ``rl.mcts.search_step``) with scripted engines.
def search_begin(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's search_begin")


def search_step(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's search_step")


def search_end(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's search_end")


# Direct battle loop (cg.game) — name-only stubs, same idea as the search_* API.
def battle_start(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's battle_start")


def battle_select(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's battle_select")


def battle_finish(*args, **kwargs):
    raise NotImplementedError("monkeypatch the consuming module's battle_finish")


def _install():
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")
    for name, value in (
        ("EnergyType", EnergyType), ("AreaType", AreaType),
        ("OptionType", OptionType), ("CardType", CardType),
        ("SelectContext", SelectContext),
        ("all_attack", all_attack), ("all_card_data", all_card_data),
        ("to_observation_class", to_observation_class),
        ("search_begin", search_begin), ("search_step", search_step),
        ("search_end", search_end),
    ):
        setattr(api, name, value)
    cg.api = api
    game = types.ModuleType("cg.game")
    for name, value in (("battle_start", battle_start),
                        ("battle_select", battle_select),
                        ("battle_finish", battle_finish)):
        setattr(game, name, value)
    cg.game = game
    sys.modules.setdefault("cg", cg)
    sys.modules.setdefault("cg.api", api)
    sys.modules.setdefault("cg.game", game)


_install()
