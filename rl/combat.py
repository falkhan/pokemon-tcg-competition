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
