"""Deck-agnostic combat core — damage / affordability / KO lookahead.

Pure Python, built ONLY from the bundled `cg` engine (no numpy / polars / torch),
so it ships inside the rule-based Kaggle submission unchanged (see rl/generic_pilot.py
and submission_rules/). `rl.encoders` re-exports these so training code keeps its
`from rl.encoders import _CARD, _best_damage` imports working.

Tables (positional tuples — dataclass refactor is tracked tech-debt in docs/M6.md):
  _ATK[attack_id]  = (damage, cost)           cost = tuple[EnergyType-int]; 0 = colorless/any
  _CARD[card_id]   = (weakness, resistance, energy_type, attacks, prize)
"""
from cg.api import all_card_data as _all_card_data, all_attack as _all_attack

COLORLESS = 0

_ATK = {a.attackId: (a.damage, tuple(int(e) for e in a.energies)) for a in _all_attack()}
_CARD = {c.cardId: (c.weakness, c.resistance, int(c.energyType), c.attacks,
                    3 if c.megaEx else 2 if c.ex else 1)
         for c in _all_card_data()}


def _can_afford(energies, cost) -> bool:
    """Do a Pokémon's attached energies cover an attack's cost (colorless = any)?"""
    have = {}
    for e in energies:
        have[e] = have.get(e, 0) + 1
    total = len(energies)
    n_colorless = sum(1 for c in cost if c == COLORLESS)
    used = 0
    for c in cost:
        if c == COLORLESS:
            continue
        if have.get(c, 0) <= 0:
            return False
        have[c] -= 1
        used += 1
    return total - used >= n_colorless


# --- Race math (M7.2b) — the deck-agnostic port of the experts' k-turn planning.
# All turn counts assume attach-1-of-own-type/turn (the same model the extra_energy
# lookahead uses); UNREACHABLE marks "can never KO" as a large int so scorer
# comparisons stay branch-free.
UNREACHABLE = 99


def _charged_best(attacker, target=None) -> tuple[int, int]:
    """Best attack by damage vs `target` assuming FULL charge: (damage, cost_total).

    Unlike _best_damage this skips affordability — it answers "what is this
    Pokémon's endgame attack worth", which is what energy-attach planning needs
    (_best_damage's affordable-only view is why the pilot stopped charging once
    the CHEAPEST attack was paid). Damage ties prefer the cheaper attack.
    target=None scores raw printed damage (promote contexts with no opponent).
    """
    if attacker is None or attacker.id not in _CARD:
        return (0, 0)
    _, _, atk_type, attacks, _ = _CARD[attacker.id]
    t_weak, t_res, _, _, _ = _CARD.get(target.id, (None, None, 0, [], 1)) \
        if target is not None else (None, None, 0, [], 1)
    best = (0, 0)
    for aid in attacks:
        if aid not in _ATK:
            continue
        dmg, cost = _ATK[aid]
        if dmg <= 0:
            continue
        if t_weak is not None and int(t_weak) == atk_type:
            dmg *= 2
        elif t_res is not None and int(t_res) == atk_type:
            dmg = max(0, dmg - 30)
        if dmg > best[0] or (dmg == best[0] and len(cost) < best[1]):
            best = (dmg, len(cost))
    return best


def _turns_to_ready(pokemon, target=None) -> int:
    """Attaches still needed before `pokemon` can fire its charged-best attack
    (attach 1/turn). Total cost, not typed: own-type energy pays typed AND
    colorless slots, so the gap is cost_total - attached (off-type costs are
    undercounted — accepted approximation; _can_afford stays the exact check).
    Works on hand cards (no .energies -> 0 attached). UNREACHABLE if it can
    never deal damage."""
    dmg, cost_total = _charged_best(pokemon, target)
    if dmg <= 0:
        return UNREACHABLE
    return max(0, cost_total - len(getattr(pokemon, "energies", ())))


def _hits_to_ko(attacker, target) -> int:
    """Charged-best hits needed to KO `target` (UNREACHABLE if damage is 0)."""
    dmg = _charged_best(attacker, target)[0]
    if dmg <= 0 or target is None:
        return UNREACHABLE
    return -(-target.hp // dmg)          # ceil without math


def _turns_to_first_ko(attacker, target) -> int:
    """My turns until `attacker` KOs `target`: max(gap,1) + hits - 1 — an attacker
    one energy short still fires THIS turn (attach happens before the attack,
    the same semantics as the pilot's +1-attach unblock tier)."""
    gap = _turns_to_ready(attacker, target)
    hits = _hits_to_ko(attacker, target)
    if gap >= UNREACHABLE or hits >= UNREACHABLE:
        return UNREACHABLE
    return min(UNREACHABLE, max(gap, 1) + hits - 1)


def _best_damage(attacker, target, extra_energy: int = 0) -> int:
    """Max damage `attacker` can deal to `target` this turn (best affordable attack,
    after weakness/resistance vs the attacker's type). extra_energy simulates attaching
    that many of the attacker's own energy (the '+1 attach enables the attack' case)."""
    if attacker is None or target is None or attacker.id not in _CARD:
        return 0
    _, _, atk_type, attacks, _ = _CARD[attacker.id]
    energies = list(attacker.energies) + [atk_type] * extra_energy
    t_weak, t_res, _, _, _ = _CARD.get(target.id, (None, None, 0, [], 1))
    best = 0
    for aid in attacks:
        if aid not in _ATK:
            continue
        dmg, cost = _ATK[aid]
        if dmg <= 0 or not _can_afford(energies, cost):
            continue
        if t_weak is not None and int(t_weak) == atk_type:
            dmg *= 2
        elif t_res is not None and int(t_res) == atk_type:
            dmg = max(0, dmg - 30)
        best = max(best, dmg)
    return best
