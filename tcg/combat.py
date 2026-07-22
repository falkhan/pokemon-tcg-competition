"""Damage math for the pilot: can an attack be paid for, and how hard can a
Pokémon hit?

Same algorithms as the old ``rl/combat.py`` ``_can_afford`` / ``_best_damage``,
now with public names and dataclass lookups.
"""
from collections.abc import Sequence

from tcg.constants import (COLORLESS, RESISTANCE_REDUCTION, UNREACHABLE_TURNS,
                           WEAKNESS_MULTIPLIER)
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


# --- Race math (M7.2b) — the deck-agnostic port of the experts' k-turn planning.
# Turn counts assume attach-1-of-own-type/turn (the extra_energy model);
# UNREACHABLE_TURNS marks "can never KO" as a large int, keeping scorer
# comparisons branch-free.

# Card FACTS the tables can't otherwise see (M13 Rung 0a — mirrored from
# rl/combat.py CONDITIONAL_ATTACKS, change BOTH): attack -> required own-board
# card id. Enforced only when callers pass board_ids; None = legacy behavior.
CONDITIONAL_ATTACKS = {980: 675}   # Solrock's attack needs Lunatone in play


def attack_available(attack_id, board_ids) -> bool:
    required = CONDITIONAL_ATTACKS.get(attack_id)
    return required is None or board_ids is None or required in board_ids


def charged_best(attacker, target=None, board_ids=None) -> tuple[int, int]:
    """Best attack by damage vs ``target`` assuming FULL charge: (damage, cost_total).

    Unlike ``best_damage`` this skips affordability — it answers "what is this
    Pokémon's endgame attack worth", which is what energy-attach planning needs
    (the affordable-only view is why the pilot stopped charging once the
    CHEAPEST attack was paid). Damage ties prefer the cheaper attack.
    ``target=None`` scores raw printed damage (promote with no opponent active).
    """
    if attacker is None or attacker.id not in CARDS:
        return (0, 0)
    attacker_card = CARDS[attacker.id]
    attack_type = attacker_card.energy_type
    target_card = CARDS.get(target.id, UNKNOWN_CARD) if target is not None else UNKNOWN_CARD

    best = (0, 0)
    for attack_id in attacker_card.attack_ids:
        if attack_id not in ATTACKS or not attack_available(attack_id, board_ids):
            continue
        attack = ATTACKS[attack_id]
        damage = attack.damage
        if damage <= 0:
            continue
        if target_card.weakness is not None and target_card.weakness == attack_type:
            damage *= WEAKNESS_MULTIPLIER
        elif target_card.resistance is not None and target_card.resistance == attack_type:
            damage = max(0, damage - RESISTANCE_REDUCTION)
        if damage > best[0] or (damage == best[0] and len(attack.cost) < best[1]):
            best = (damage, len(attack.cost))
    return best


def turns_to_ready(pokemon, target=None, board_ids=None) -> int:
    """Attaches still needed before ``pokemon`` can fire its charged-best attack
    (attach 1/turn). Total cost, not typed: own-type energy pays typed AND
    colorless slots, so the gap is cost_total - attached (off-type costs are
    undercounted — accepted approximation; ``can_afford`` stays the exact check).
    Works on hand cards (no ``.energies`` -> 0 attached). UNREACHABLE_TURNS if
    it can never deal damage."""
    damage, cost_total = charged_best(pokemon, target, board_ids)
    if damage <= 0:
        return UNREACHABLE_TURNS
    return max(0, cost_total - len(getattr(pokemon, "energies", ())))


def hits_to_ko(attacker, target) -> int:
    """Charged-best hits to KO ``target`` (UNREACHABLE_TURNS if damage is 0)."""
    damage = charged_best(attacker, target)[0]
    if damage <= 0 or target is None:
        return UNREACHABLE_TURNS
    return -(-target.hp // damage)       # ceil without math


def turns_to_first_ko(attacker, target) -> int:
    """My turns until ``attacker`` KOs ``target``: max(gap,1) + hits - 1 — an
    attacker one energy short still fires THIS turn (attach happens before the
    attack, the same semantics as the pilot's +1-attach unblock tier)."""
    gap = turns_to_ready(attacker, target)
    hits = hits_to_ko(attacker, target)
    if gap >= UNREACHABLE_TURNS or hits >= UNREACHABLE_TURNS:
        return UNREACHABLE_TURNS
    return min(UNREACHABLE_TURNS, max(gap, 1) + hits - 1)


def best_damage(attacker, target, extra_energy: int = 0, board_ids=None) -> int:
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
        if attack_id not in ATTACKS or not attack_available(attack_id, board_ids):
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


# Spread / multi-target attacks (M22c C2). `cg.api.Attack` exposes only
# (damage, energies) plus free-form English `text`, so a multi-target effect is
# invisible to ATTACKS — which is why our threat model could only ever ask "can
# their ACTIVE KO my ACTIVE" while dragapult put a third of its damage on our
# bench (M22 diagnostic, p<0.0001). Curated like CONDITIONAL_ATTACKS above:
# across EVERY deck in decks/ + data/kaggle/meta_v*, only these are real bench
# threats. (bench_pool_damage, mode); "any" = distributable across the bench,
# "one" = a single chosen target. Mirrored in rl/combat.py — change BOTH.
SPREAD_ATTACKS = {
    154: (60, "any"),   # Phantom Dive  — 200 active + 6 counters placed anywhere on the bench
    183: (100, "one"),  # Cruel Arrow   — 100 to ONE of the opponent's Pokemon (active damage is 0)
    412: (30, "one"),   # Insta-Strike  — 30 active + 30 to one benched Pokemon
}


def threatened(attacker, defenders, board_ids=None) -> list:
    """Which of `defenders` (index 0 = the active) `attacker` could KO THIS turn
    with a single attack, counting spread damage.

    Generalises the active-only check the solver leaf and rl/plan.py both use.
    With no spread attack available it returns [active] or [] — byte-identical
    behaviour to the old test — so this is a strict superset.

    APPROXIMATE by design (C2.0 is a falsification test): the pool is taken from
    the best affordable spread attack and allocated greedily to the cheapest
    bench KOs, and weakness is NOT applied to bench damage — the printed cards
    say "Don't apply Weakness and Resistance for Benched Pokemon".
    """
    if attacker is None or not defenders or attacker.id not in CARDS:
        return []
    out = []
    active = defenders[0] if defenders else None
    if active is not None and best_damage(attacker, active,
                                           board_ids=board_ids) >= (active.hp or 0):
        out.append(active)
    energies = list(getattr(attacker, "energies", ()) or [])
    pool, mode = 0, "one"
    for aid in (CARDS[attacker.id].attack_ids or ()):
        if aid not in ATTACKS or not attack_available(aid, board_ids):
            continue
        if not can_afford(energies, ATTACKS[aid].cost):
            continue
        p, m = SPREAD_ATTACKS.get(aid, (0, "one"))
        if p > pool:
            pool, mode = p, m
    bench = [d for d in defenders[1:] if d is not None]
    if pool > 0 and bench:
        if mode == "one":
            hit = min((d for d in bench if pool >= (d.hp or 0)),
                      key=lambda d: d.hp or 0, default=None)
            if hit is not None:
                out.append(hit)
        else:
            left = pool
            for d in sorted(bench, key=lambda x: x.hp or 0):
                if left >= (d.hp or 0):
                    left -= (d.hp or 0)
                    out.append(d)
    return out
