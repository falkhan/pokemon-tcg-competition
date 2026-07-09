"""Damage math for the pilot: can an attack be paid for, and how hard can a
Pokémon hit?

Same algorithms as the old ``rl/combat.py`` ``_can_afford`` / ``_best_damage``,
now with public names and dataclass lookups.
"""
from collections.abc import Sequence

from tcg.constants import COLORLESS, RESISTANCE_REDUCTION, WEAKNESS_MULTIPLIER
from tcg.library import ATTACKS, CARDS
from tcg.models import UNKNOWN_CARD


def can_afford(attached_energies: Sequence[int], cost: Sequence[int]) -> bool:
    """Do a Pokémon's attached energies cover an attack's cost?

    Typed cost slots each consume one matching energy; COLORLESS slots are
    paid by whatever is left over (any type).
    """
    available: dict[int, int] = {}
    for energy in attached_energies:
        available[energy] = available.get(energy, 0) + 1

    colorless_slots = sum(1 for slot in cost if slot == COLORLESS)
    typed_slots_paid = 0
    for slot in cost:
        if slot == COLORLESS:
            continue
        if available.get(slot, 0) <= 0:
            return False
        available[slot] -= 1
        typed_slots_paid += 1
    return len(attached_energies) - typed_slots_paid >= colorless_slots


def best_damage(attacker, target, extra_energy: int = 0) -> int:
    """Max damage ``attacker`` can deal to ``target`` this turn.

    Best affordable attack, after weakness/resistance against the attacker's
    type. ``attacker`` and ``target`` are the engine's in-play Pokémon objects
    (or None). ``extra_energy`` simulates attaching that many of the attacker's
    own energy type first (the "+1 attach enables the attack" lookahead).
    """
    if attacker is None or target is None or attacker.id not in CARDS:
        return 0
    attacker_card = CARDS[attacker.id]
    attack_type = attacker_card.energy_type
    energies = list(attacker.energies) + [attack_type] * extra_energy
    target_card = CARDS.get(target.id, UNKNOWN_CARD)

    best = 0
    for attack_id in attacker_card.attack_ids:
        if attack_id not in ATTACKS:
            continue
        attack = ATTACKS[attack_id]
        damage = attack.damage
        if damage <= 0 or not can_afford(energies, attack.cost):
            continue
        if target_card.weakness is not None and target_card.weakness == attack_type:
            damage *= WEAKNESS_MULTIPLIER
        elif target_card.resistance is not None and target_card.resistance == attack_type:
            # Floored at 0 here — unlike pilot.score_attack, which deliberately
            # lets the resistance-adjusted damage go negative in its score.
            damage = max(0, damage - RESISTANCE_REDUCTION)
        best = max(best, damage)
    return best
