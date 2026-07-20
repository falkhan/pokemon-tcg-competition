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

# Card FACTS the tables can't otherwise see (M13 Rung 0a — mirrored in
# tcg/combat.py, change BOTH): attacks whose printed damage requires a
# specific card on the attacker's OWN board. Enforced only when callers pass
# `board_ids` (the ids in play on the attacker's side); board_ids=None keeps
# every legacy call site byte-identical.
CONDITIONAL_ATTACKS = {980: 675}   # Solrock's attack needs Lunatone in play


def _attack_available(attack_id, board_ids) -> bool:
    req = CONDITIONAL_ATTACKS.get(attack_id)
    return req is None or board_ids is None or req in board_ids


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


def _charged_best(attacker, target=None, board_ids=None) -> tuple[int, int]:
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
        if aid not in _ATK or not _attack_available(aid, board_ids):
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


def _turns_to_ready(pokemon, target=None, board_ids=None) -> int:
    """Attaches still needed before `pokemon` can fire its charged-best attack
    (attach 1/turn). Total cost, not typed: own-type energy pays typed AND
    colorless slots, so the gap is cost_total - attached (off-type costs are
    undercounted — accepted approximation; _can_afford stays the exact check).
    Works on hand cards (no .energies -> 0 attached). UNREACHABLE if it can
    never deal damage."""
    dmg, cost_total = _charged_best(pokemon, target, board_ids)
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


def _best_damage(attacker, target, extra_energy: int = 0, board_ids=None) -> int:
    """Max damage `attacker` can deal to `target` this turn (best affordable attack,
    after weakness/resistance vs the attacker's type). extra_energy simulates attaching
    that many of the attacker's own energy (the '+1 attach enables the attack' case).
    board_ids (optional): own in-play card ids — gates CONDITIONAL_ATTACKS."""
    if attacker is None or target is None or attacker.id not in _CARD:
        return 0
    _, _, atk_type, attacks, _ = _CARD[attacker.id]
    energies = list(attacker.energies) + [atk_type] * extra_energy
    t_weak, t_res, _, _, _ = _CARD.get(target.id, (None, None, 0, [], 1))
    best = 0
    for aid in attacks:
        if aid not in _ATK or not _attack_available(aid, board_ids):
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


# Spread / multi-target attacks (M22c C2). `cg.api.Attack` exposes only
# (damage, energies) plus free-form English `text`, so a multi-target effect is
# invisible to _ATK — which is why our threat model could only ever ask "can
# their ACTIVE KO my ACTIVE" while dragapult put a third of its damage on our
# bench (M22 diagnostic, p<0.0001). Curated like CONDITIONAL_ATTACKS above:
# across EVERY deck in decks/ + data/kaggle/meta_v*, only these are real bench
# threats. (bench_pool_damage, mode); "any" = distributable across the bench,
# "one" = a single chosen target. Mirrored in tcg/combat.py — change BOTH.
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
    if attacker is None or not defenders or attacker.id not in _CARD:
        return []
    out = []
    active = defenders[0] if defenders else None
    if active is not None and _best_damage(attacker, active,
                                           board_ids=board_ids) >= (active.hp or 0):
        out.append(active)
    energies = list(getattr(attacker, "energies", ()) or [])
    pool, mode = 0, "one"
    for aid in (_CARD[attacker.id][3] or ()):
        if aid not in _ATK or not _attack_available(aid, board_ids):
            continue
        if not _can_afford(energies, _ATK[aid][1]):
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
