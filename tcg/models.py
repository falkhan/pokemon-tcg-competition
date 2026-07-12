"""Dataclasses for the static card data we derive from the engine's database.

These replace the positional tuples of the old ``rl/combat.py``
(``_CARD[cid] = (weakness, resistance, energy_type, attacks, prize)`` and
``_ATK[aid] = (damage, cost)``), so consumers read ``card.prize_count`` instead
of ``_CARD[cid][4]``.

Only *static* card data lives here. An in-play Pokémon (hp, attached energies,
status, …) remains the engine's own ``cg.api.Pokemon`` object.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Attack:
    """One attack from the engine's attack table."""

    damage: int
    """Printed damage. 0 for variable / effect-only attacks (a known engine gap)."""

    cost: tuple[int, ...]
    """Energy cost as EnergyType ints; COLORLESS (0) slots are payable by any energy."""


@dataclass(frozen=True, slots=True)
class Card:
    """Static data for one card in the card database."""

    weakness: int | None
    """EnergyType this card takes double damage from, or None."""

    resistance: int | None
    """EnergyType this card takes reduced damage from, or None."""

    energy_type: int
    """The card's OWN attack type (EnergyType int) — what weakness math checks."""

    attack_ids: tuple[int, ...]
    """Ids into the attack table. Some ids have no attack-table entry, so
    consumers must filter on membership (see ``library.known_attacks``)."""

    prize_count: int
    """Prizes the opponent takes when this card is knocked out: 3 mega-ex / 2 ex / 1."""

    name: str | None = None
    """Printed card name — the key evolution lines are linked by. None in test
    stubs that predate the M7.5 fetch/hand-discard guards."""

    basic: bool = True
    """True for basic Pokémon (playable straight to the bench). Non-Pokémon
    cards keep the default; nothing consults it for them."""

    evolves_from: str | None = None
    """Name of the basic this card evolves from, or None for basics/non-Pokémon."""


# Fallbacks for ids missing from the tables. The old code used two default
# tuples — (None, None, 0, [], 1) in combat and (0, 0, 0, [], 1) in the pilot's
# prize read, where only the prize slot was consulted — so the None variant is
# observably identical at every call site.
UNKNOWN_CARD = Card(weakness=None, resistance=None, energy_type=0,
                    attack_ids=(), prize_count=1)
UNKNOWN_ATTACK = Attack(damage=0, cost=())
