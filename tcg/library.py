"""The card database: every Card and Attack, keyed by id.

Built eagerly at import time from the bundled ``cg`` engine — exactly like the
old ``rl/combat.py`` tables — so a Kaggle submission can rely on plain module
import order. This is the only module that consumes ``cg.api.all_card_data()``
/ ``all_attack()``.
"""
from collections.abc import Iterator

from cg.api import CardType, all_attack, all_card_data

from tcg import constants
from tcg.models import Attack, Card

ATTACKS: dict[int, Attack] = {
    attack.attackId: Attack(damage=attack.damage,
                            cost=tuple(int(energy) for energy in attack.energies))
    for attack in all_attack()
}

# weakness/resistance are normalized to plain int | None here; every consumer
# only ever compares them with `x is not None and int(x) == attack_type`.
CARDS: dict[int, Card] = {
    card.cardId: Card(
        weakness=None if card.weakness is None else int(card.weakness),
        resistance=None if card.resistance is None else int(card.resistance),
        energy_type=int(card.energyType),
        attack_ids=tuple(card.attacks),
        prize_count=3 if card.megaEx else 2 if card.ex else 1,
        # getattr: the fake_cg test stub predates these fields (M7.5 guards)
        name=getattr(card, "name", None),
        basic=bool(getattr(card, "basic", True)),
        evolves_from=getattr(card, "evolvesFrom", None),
    )
    for card in all_card_data()
}

# M41: retreat cost, kept as its own map rather than a Card field so the frozen
# dataclass and every positional consumer of it stay untouched (the rl/combat.py
# _RETREAT twin — change BOTH). getattr for the fake_cg stub, which predates it.
RETREAT_COSTS: dict[int, int] = {
    card.cardId: int(getattr(card, "retreatCost", 0) or 0)
    for card in all_card_data()
}

HAND_DISCARD_TRAINER_IDS: frozenset[int] = frozenset(
    card_id for card_id, card in CARDS.items()
    if card.name in constants.HAND_DISCARD_TRAINER_NAMES)

POKEMON_CARD_IDS: frozenset[int] = frozenset(
    card.cardId for card in all_card_data() if card.cardType == CardType.POKEMON)
ENERGY_CARD_IDS: frozenset[int] = frozenset(
    card.cardId for card in all_card_data()
    if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY))


def known_attacks(card: Card) -> Iterator[Attack]:
    """The card's attacks that actually exist in the attack table.

    Some attack ids on cards have no table entry, so every consumer must
    filter on membership rather than index blindly.
    """
    return (ATTACKS[attack_id] for attack_id in card.attack_ids
            if attack_id in ATTACKS)
