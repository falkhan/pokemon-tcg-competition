"""M42: the scaling table DERIVED from the engine's own attack text.

Two layers, because `tests/conftest.py` replaces `cg` with a 10-card stub whose
attacks carry no rules text at all — so under pytest `derive_scaling_table()`
sees nothing and `SCALING_ATTACKS` collapses to the curated six.

  1. `parse_attack_text` against LITERAL real card text. The strings are facts
     printed on cards, pinned the same way `tests/test_scaling.py` pins owner
     NAMES. This is where the accept and reject grammar lives.
  2. One SUBPROCESS test against the REAL engine (the `tcg/decklab.py` smoke
     runner precedent), pinning the derived table's shape and the curated
     entries' survival.

Honest limit of (2): it pins the table against the pool *as installed*. A pool
update that re-points an attack id will fail it loudly, which is the trap
`tests/test_scaling.py` exists for — but the expected numbers here are a
snapshot, so a deliberate pool change means re-reading this table, not deleting
the assertion.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl.scaling import (CURATED_ATTACKS, _HUMAN_OVERRIDES, NOMINAL_UNITS,
                        derive_scaling_table, effective_damage,
                        parse_attack_text)

ROOT = Path(__file__).resolve().parent.parent


class Mon:
    def __init__(self, cid=1, energies=(), hp=100, max_hp=None):
        self.id, self.energies, self.hp = cid, list(energies), hp
        self.maxHp = hp if max_hp is None else max_hp


# --- (1a) the grammar ACCEPTS what the cards really say --------------------
# id, text, printed damage, expected (mode, per_unit, base)
ACCEPT = [
    (1072, "Place 2 damage counters on your opponent’s Active Pokémon for "
           "each card in your hand.", 0, ("hand", 20, 0)),
    (120, "This attack does 30 more damage for each Energy attached to both "
          "Active Pokémon.", 30, ("both_nrg", 30, 30)),
    (115, "This attack does 20 damage for each of your Benched Pokémon.", 0,
     ("bench_only", 20, 0)),
    (195, "This attack does 30 more damage for each {G} Energy attached to "
          "all of your Pokémon.", 30, ("team_nrg", 30, 30)),
    (123, "This attack does 30 damage for each card in your opponent’s hand.",
     0, ("opp_hand", 30, 0)),
    (333, "This attack does 10 damage for each damage counter on this Pokémon.",
     0, ("dmg_counters_self", 10, 0)),
    (119, "This attack does 20 more damage for each of your opponent’s "
          "Benched Pokémon.", 20, ("opp_bench", 20, 20)),
    (274, "This attack does 20 more damage for each Benched Pokémon (both "
          "yours and your opponent’s).", 60, ("all_bench", 20, 60)),
    (184, "This attack does 60 damage for each Prize card your opponent has "
          "taken.", 0, ("prizes_taken_opp", 60, 0)),
    (1006, "This attack does 80 damage for each Prize card you have taken.", 0,
     ("prizes_taken_us", 80, 0)),
    (362, "This attack does 50 more damage for each {W} Energy attached to "
          "this Pokémon.", 10, ("my_nrg", 50, 10)),
    # Two sentences, only the FIRST of which is this attack's damage.
    (732, "This attack does 50 more damage for each Prize card your opponent "
          "has taken. This Pokémon also does 30 damage to itself.", 130,
     ("prizes_taken_opp", 50, 130)),
    # A leading condition sentence must not swallow the real clause.
    (232, "If you go second, you can’t use this attack during your first "
          "turn. This attack does 30 damage for each of your Benched Pokémon.",
     0, ("bench_only", 30, 0)),
]


@pytest.mark.parametrize("aid,text,printed,want", ACCEPT,
                         ids=[str(a[0]) for a in ACCEPT])
def test_grammar_accepts_real_card_text(aid, text, printed, want):
    assert parse_attack_text(text, printed) == want


def test_more_means_on_top_of_printed_and_bare_means_replace():
    """The single distinction the whole table turns on."""
    assert parse_attack_text(
        "This attack does 20 more damage for each card in your hand.", 90) == \
        ("hand", 20, 90)
    assert parse_attack_text(
        "This attack does 20 damage for each card in your hand.", 90) == \
        ("hand", 20, 0)


# --- (1b) the grammar REJECTS everything it cannot evaluate ----------------
REJECT = [
    # a coin flip is not a countable board fact
    ("Flip a coin until you get tails. For each heads, discard an Energy from "
     "your opponent’s Active Pokémon.", 140),
    # a cost paid INSIDE the attack: the unit is a choice not yet made
    ("Discard the top card of each player’s deck. This attack does 140 more "
     "damage for each Energy card discarded in this way.", 140),
    ("Discard up to 5 {R} Energy from this Pokémon. This attack does 70 "
     "damage for each card you discarded in this way.", 0),
    # self-referential: the unit is created by the attack itself
    ("Put up to 9 damage counters on this Pokémon. This attack does 20 damage "
     "for each damage counter you placed in this way.", 0),
    ("This attack does 20 damage for each Pokémon in your discard pile that "
     "has the United Wings attack.", 0),
    # damage that does not land on the defender we are scoring against
    ("This Pokémon also does 10 damage to itself for each damage counter on "
     "it.", 130),
    ("This attack does 30 damage to 1 of your opponent’s Pokémon for each "
     "Energy attached to this Pokémon.", 0),
    ("This attack also does 10 damage to each of your opponent’s Benched "
     "Pokémon for each Prize card your opponent has taken.", 130),
    # NAMED or qualified subsets we cannot count from the observation
    ("This attack does 30 damage for each of your Team Rocket’s Pokémon in "
     "play.", 0),
    ("This attack does 30 damage for each of your Ancient Pokémon in play.", 0),
    ("This attack does 40 more damage for each Stage 2 Pokémon on your Bench.",
     180),
    ("This attack does 60 damage for each of your opponent’s Pokémon {ex} in "
     "play.", 0),
    ("This attack does 20 more damage for each {L} Energy attached to all of "
     "your Iono’s Pokémon.", 20),
    # zones and objects with no evaluator
    ("This attack does 30 damage for each Basic Energy card in your "
     "opponent’s discard pile.", 0),
    ("This attack does 30 damage for each Pokémon Tool attached to all of "
     "your Pokémon.", 0),
    ("This attack does 100 damage for each Special Condition affecting your "
     "opponent’s Active Pokémon.", 0),
]


@pytest.mark.parametrize("text,printed", REJECT,
                         ids=[t[:34] for t, _ in REJECT])
def test_grammar_rejects_what_it_cannot_evaluate(text, printed):
    """Unparsed must mean 'fall through to printed damage', never a guess.
    That direction is safe: a missing entry under-rates a scaling attacker,
    which is the status quo, while a wrong entry invents damage — and every
    invented damage number in this lane has been falsified (docs/M41.md)."""
    assert parse_attack_text(text, printed) is None


def test_no_text_at_all_is_not_an_entry():
    assert parse_attack_text("", 90) is None
    assert parse_attack_text(None, 90) is None


# --- the curated entries the parser must NOT reproduce ---------------------
def test_human_overrides_are_exactly_the_ones_the_parser_cannot_derive():
    """Both are judgement calls, and a parser that reproduced them would be
    guessing. Pinned so neither can be quietly promoted to 'derived'."""
    assert set(_HUMAN_OVERRIDES) == {560, 608}
    assert parse_attack_text("This attack does 30 damage for each of your "
                             "Team Rocket’s Pokémon in play.", 0) is None
    assert parse_attack_text("You may discard up to 2 Energy from your Benched "
                             "Pokémon. This attack does 60 more damage for "
                             "each card you discarded in this way.", 160) is None


def test_curated_always_wins_the_merge():
    table = derive_scaling_table()
    for aid, entry in CURATED_ATTACKS.items():
        assert table[aid] == entry


# --- every mode has an evaluator and a nominal ------------------------------
def test_every_mode_is_evaluable_and_has_a_nominal():
    modes = {m for m, _, _ in CURATED_ATTACKS.values()} | {
        m for _p, m in
        __import__("rl.scaling", fromlist=["_UNIT_MODES"])._UNIT_MODES}
    for mode in modes:
        assert mode in NOMINAL_UNITS, f"{mode} has no NOMINAL_UNITS entry"


def test_damage_counter_modes_read_the_board():
    """dmg_counters_* is (maxHp - hp) / 10, and must floor at 0 on a fresh
    body rather than going negative on a healed one."""
    import rl.scaling as sc
    hurt = Mon(hp=60, max_hp=200)          # 14 counters
    fresh = Mon(hp=200, max_hp=200)
    healed = Mon(hp=220, max_hp=200)
    assert sc._damage_counters(hurt) == 14
    assert sc._damage_counters(fresh) == 0
    assert sc._damage_counters(healed) == 0


def test_unsupplied_units_fall_back_to_printed_not_to_zero():
    """Every counting argument defaults to 0, and `max(printed, ...)` means a
    caller that cannot supply a unit gets the printed floor. rl/encoders.py's
    energy block relies on this: it passes only energy-derived units, and a
    hand-mode attack must not invent damage there."""
    import rl.scaling as sc
    aid = next(iter(a for a, (m, _p, _b) in sc.SCALING_ATTACKS.items()
                    if m == "hand"), None)
    assert aid is not None
    printed = sc._ATK.get(aid, (0, ()))[0]
    assert effective_damage(aid, Mon(), Mon()) == printed


# --- (2) the real pool, in a subprocess -------------------------------------
_SNAPSHOT_PROBE = r"""
import json
from collections import Counter
import rl.scaling as sc
t = sc.derive_scaling_table()
print(json.dumps({
    "n": len(t),
    "modes": dict(Counter(m for m, _, _ in t.values())),
    "curated": {str(k): list(t[k]) for k in sc.CURATED_ATTACKS},
    "spot": {str(k): list(t[k]) for k in (123, 333, 119, 732, 1072)
             if k in t},
}))
"""

# Pinned 2026-08-04 against the installed pool. A pool refresh that moves these
# is a REAL event to read, not a number to bump silently.
EXPECTED_MODES = {
    "my_nrg": 13, "opp_nrg": 8, "dmg_counters_self": 8, "dmg_counters_opp": 7,
    "bench_only": 4, "prizes_taken_opp": 4, "my_bench": 4, "opp_bench": 4,
    "opp_hand": 2, "team_nrg": 2, "all_bench": 2, "prizes_taken_us": 2,
    "both_nrg": 1, "hand": 1, "flat": 1,
}


def test_real_pool_snapshot():
    """The derived table against the REAL engine — pytest stubs `cg`, so this
    is the only place the parser meets all 1,556 attacks."""
    proc = subprocess.run([sys.executable, "-c", _SNAPSHOT_PROBE], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.skip(f"real engine unavailable: {proc.stderr.strip()[:200]}")
    got = json.loads(proc.stdout)

    assert got["modes"] == EXPECTED_MODES
    assert got["n"] == sum(EXPECTED_MODES.values()) == 63

    # the six curated survive the merge byte-identical
    for aid, entry in CURATED_ATTACKS.items():
        assert got["curated"][str(aid)] == list(entry)

    # and four of the six are re-derived by the parser INDEPENDENTLY, which is
    # what makes this extraction rather than a hand table with extra steps
    assert got["spot"]["1072"] == ["hand", 20, 0]          # Powerful Hand
    assert got["spot"]["123"] == ["opp_hand", 30, 0]       # Mind Ruler
    assert got["spot"]["333"] == ["dmg_counters_self", 10, 0]   # Flail
    assert got["spot"]["119"] == ["opp_bench", 20, 20]     # Ogre Comeback
    assert got["spot"]["732"] == ["prizes_taken_opp", 50, 130]  # Voltage Burst
