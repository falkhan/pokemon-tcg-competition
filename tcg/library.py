"""The card database: every Card and Attack, keyed by id.

Built eagerly at import time from the bundled ``cg`` engine — exactly like the
old ``rl/combat.py`` tables — so a Kaggle submission can rely on plain module
import order. This is the only module that consumes ``cg.api.all_card_data()``
/ ``all_attack()``.
"""
from collections.abc import Iterator

from cg.api import CardType, all_attack, all_card_data

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
    )
    for card in all_card_data()
}

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
