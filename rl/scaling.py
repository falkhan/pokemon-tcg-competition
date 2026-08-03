"""Effective damage for attacks whose printed damage is not their real damage.

`rl/combat.py::_ATK` stores an attack's PRINTED damage, which for a scaling
attack is its base — the number before "…for each Energy attached" or "…for
each card in your hand". 174 of the pool's 1556 attacks scale and 142 of those
print <=30, so the damage model reads them as harmless. The starkest case is our
own list: Alakazam #743 `Powerful Hand` prints **0** and really places 2 damage
counters per card in hand — 340 at the 17-card hands that deck holds.

This module is a curated table, in the spirit of `combat.SPREAD_ATTACKS` and
`combat.CONDITIONAL_ATTACKS`: free-form English rules text cannot be parsed
reliably, so the handful of attacks that decide real matchups are encoded by
hand against the printed card, and everything else falls through to the printed
number.

OPT-IN ONLY. Nothing imports this unless a caller asks for it — the shipped
agent's features stay byte-identical, which matters because the live net was
BC-trained against the current (wrong) features and demonstrably learned around
them: over 617 live turns with Alakazam active it attacked 617 times while
`_best_damage` said "cannot attack" every single time
(`scripts/m41_scaling_probe.py`). Changing what the net is fed without
retraining would be a silent distribution shift for no measured gain.

The consumer today is the deck probe, whose rule pilot has no such learned
compensation and therefore ranks our own win condition below a 90-damage
Dudunsparce.

Each entry is `attackId -> (mode, per_unit, base)`; `effective_damage` resolves
the mode against the board:

    "hand"       per_unit x cards in our hand
    "opp_nrg"    per_unit x energy attached to the DEFENDER
    "my_nrg"     per_unit x energy attached to the ATTACKER
    "both_nrg"   per_unit x energy on BOTH actives
    "my_bench"   per_unit x our benched Pokemon

Every id below was read off the engine's own rules text, not guessed — a first
draft of this table guessed two of three ids and pointed at Trapinch and
Magcargo ex instead of the cards it named. `tests/test_scaling.py` re-derives
each id from `cg.api` so a card-pool update cannot silently re-point an entry.
"""
from rl.combat import _ATK

# attackId -> (mode, per_unit, base), covering the scaling attacks played in
# multiples by the decks the leaderboard census says matter.
SCALING_ATTACKS: dict[int, tuple[str, int, int]] = {
    # Alakazam #743 "Powerful Hand" — 2 damage counters per card in hand. OUR deck's plan.
    1072: ("hand", 20, 0),
    # Teal Mask Ogerpon ex "Myriad Leaf Shower" — +30 per Energy on BOTH actives.
    120: ("both_nrg", 30, 30),
    # Team Rocket's Spidops "Rocket Rush" — 30 per Team Rocket's Pokemon in play.
    # Approximated by our bench+active count: these lists are ~all Team Rocket's.
    560: ("my_bench", 30, 0),
    # Team Rocket's Mewtwo ex "Erasure Ball" — +60 per Energy discarded from our
    # bench (optional, up to 2). Modelled at the 1-discard middle case, +60.
    608: ("flat", 0, 220),
}

# Damage assumed for a scaling attack we have NOT curated. Enough to beat the
# 10-30 chip attacks of an evolution line's lower stages, low enough not to
# outrank a genuine printed heavy hitter. Only used when `assume_unknown`.
UNKNOWN_SCALING_DAMAGE = 100


def _count_energy(pokemon) -> int:
    return len(getattr(pokemon, "energies", None) or [])


def effective_damage(attack_id: int, attacker, defender, hand_size: int = 0,
                     bench_size: int = 0) -> int:
    """Printed damage, or the curated scaling estimate when we have one.

    Returns the PRE-weakness number, exactly like `_ATK[id][0]`, so callers keep
    applying weakness and resistance themselves.
    """
    printed = _ATK.get(attack_id, (0, ()))[0]
    entry = SCALING_ATTACKS.get(attack_id)
    if entry is None:
        return printed
    mode, per_unit, base = entry
    if mode == "flat":
        units = 0
    elif mode == "hand":
        units = hand_size
    elif mode == "opp_nrg":
        units = _count_energy(defender)
    elif mode == "my_nrg":
        units = _count_energy(attacker)
    elif mode == "both_nrg":
        units = _count_energy(attacker) + _count_energy(defender)
    elif mode == "my_bench":
        units = bench_size + 1                # the active counts too
    else:
        raise ValueError(f"unknown scaling mode {mode!r} for attack {attack_id}")
    return max(printed, base + per_unit * units)


def is_scaling(attack_id: int) -> bool:
    return attack_id in SCALING_ATTACKS
