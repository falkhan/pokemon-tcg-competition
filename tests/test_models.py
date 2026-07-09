"""The Card / Attack dataclasses and the library loader."""
import dataclasses

import pytest

from tcg import library
from tcg.models import Attack, Card, UNKNOWN_ATTACK, UNKNOWN_CARD


def test_dataclasses_are_frozen():
    card = library.CARDS[1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.prize_count = 99
    attack = library.ATTACKS[101]
    with pytest.raises(dataclasses.FrozenInstanceError):
        attack.damage = 99


def test_dataclasses_use_slots():
    assert not hasattr(library.CARDS[1], "__dict__")
    assert not hasattr(library.ATTACKS[101], "__dict__")


def test_attack_fields():
    assert library.ATTACKS[102] == Attack(damage=120, cost=(6, 0))
    assert library.ATTACKS[103] == Attack(damage=0, cost=())


def test_card_fields_and_normalization():
    card = library.CARDS[1]
    assert card == Card(weakness=5, resistance=None, energy_type=6,
                        attack_ids=(101, 102, 999), prize_count=1)
    assert isinstance(card.weakness, int)  # normalized from the enum at load
    resistant = library.CARDS[2]
    assert (resistant.weakness, resistant.resistance) == (4, 6)


def test_prize_tiers():
    assert library.CARDS[1].prize_count == 1  # basic
    assert library.CARDS[2].prize_count == 2  # ex
    assert library.CARDS[3].prize_count == 3  # mega-ex


def test_card_id_sets():
    assert library.POKEMON_CARD_IDS == {1, 2, 3, 4, 5, 9, 10}
    assert library.ENERGY_CARD_IDS == {6, 8}


def test_known_attacks_filters_missing_table_entries():
    attacks = list(library.known_attacks(library.CARDS[1]))
    assert [a.damage for a in attacks] == [50, 120]  # id 999 dropped
    assert list(library.known_attacks(library.CARDS[9])) == []


def test_unknown_sentinels():
    assert UNKNOWN_CARD == Card(weakness=None, resistance=None, energy_type=0,
                                attack_ids=(), prize_count=1)
    assert UNKNOWN_ATTACK == Attack(damage=0, cost=())
