"""The curated scaling-damage table (rl/scaling.py).

A first draft of that table guessed two of its three attack ids and pointed at
Trapinch and Magcargo ex instead of the cards named in its own comments. The
id checks here re-derive every entry from the card PARQUETS — which carry the
whole 1267-card pool and are readable without the engine, unlike `rl.combat`'s
`_CARD`/`_ATK`, which `tests/conftest.py` replaces with a 10-card stub.
"""
import pytest

from rl.scaling import (CURATED_ATTACKS, NOMINAL_UNITS, SCALING_ATTACKS,
                        effective_damage, is_scaling, nominal_damage)
from tcg.cardpool import attacks_by_card, cards


class Mon:
    """Minimal stand-in for an engine Pokemon (energies is all we read)."""

    def __init__(self, cid=1, energies=(), hp=100):
        self.id, self.energies, self.hp = cid, list(energies), hp


# Which card each curated attack must belong to, by NAME — names, not ids, are
# the whole point of the check.
EXPECTED_OWNER = {
    1072: "Alakazam",
    120: "Teal Mask Ogerpon ex",
    560: "Team Rocket's Spidops",
    608: "Team Rocket's Mewtwo ex",
    115: "Dipplin",
    195: "Hydrapple ex",
}

_OWNER_BY_ATTACK = {a["attackId"]: cid
                    for cid, rows in attacks_by_card().items() for a in rows}
_PRINTED = {a["attackId"]: a["damage"]
            for rows in attacks_by_card().values() for a in rows}


# --- the table points where it says it does --------------------------------

def test_the_curated_entries_are_exactly_the_documented_ones():
    """M42: this used to assert on SCALING_ATTACKS, which is now the MERGED
    table (curated + text-derived, 63 entries against the real pool). Under
    conftest's stub the derived half is empty, so the old assertion still
    passed while being false in production — it was pinning the stub, not the
    table. It now pins what it always meant: the hand-curated set.
    `tests/test_scaling_derive.py` covers the merged table against the real
    pool, in a subprocess."""
    assert set(CURATED_ATTACKS) == set(EXPECTED_OWNER)
    for aid, entry in CURATED_ATTACKS.items():
        assert SCALING_ATTACKS[aid] == entry      # curated always wins


@pytest.mark.parametrize("aid,name", sorted(EXPECTED_OWNER.items()))
def test_curated_ids_point_at_the_cards_they_claim(aid, name):
    cid = _OWNER_BY_ATTACK.get(aid)
    assert cid is not None, f"attack {aid} belongs to no card in the pool"
    assert cards()[cid]["name_norm"] == name, (
        f"attack {aid} belongs to {cards()[cid]['name_norm']!r}, "
        f"the table says {name!r}")


def test_curated_attacks_really_do_scale():
    """Every entry must be an attack whose text scales — otherwise the table is
    inventing damage rather than recovering it."""
    from tcg.cardpool import attack_texts
    texts = attack_texts()
    if not all(texts.get(aid, {}).get("text") for aid in SCALING_ATTACKS):
        pytest.skip("needs the real engine's rules text (conftest installs a stub)")
    for aid in SCALING_ATTACKS:
        assert "for each" in texts[aid]["text"].lower(), (
            f"attack {aid} ({texts[aid]['name']}) does not scale")


def test_modes_are_all_known():
    # M42 added seven derivable modes. The list stays explicit rather than
    # reading NOMINAL_UNITS, so adding a mode to one place and forgetting the
    # other still fails here.
    valid = {"flat", "hand", "opp_hand", "opp_nrg", "my_nrg", "both_nrg",
             "my_bench", "bench_only", "opp_bench", "all_bench", "team_nrg",
             "dmg_counters_self", "dmg_counters_opp", "prizes_taken_us",
             "prizes_taken_opp"}
    for aid, (mode, _per, _base) in SCALING_ATTACKS.items():
        assert mode in valid, f"attack {aid} has unknown mode {mode!r}"


# --- the estimates ---------------------------------------------------------

def test_powerful_hand_scales_with_our_hand():
    """Alakazam #743 prints 0 and really places 2 damage counters per card in
    hand — the reason the rule pilot played our own list as a Dudunsparce deck."""
    assert _PRINTED[1072] == 0
    me, them = Mon(743), Mon(648, hp=320)
    assert effective_damage(1072, me, them, hand_size=0) == 0
    assert effective_damage(1072, me, them, hand_size=7) == 140
    assert effective_damage(1072, me, them, hand_size=17) == 340


def test_ogerpon_counts_energy_on_both_actives():
    assert _PRINTED[120] == 30
    me, them = Mon(energies=[1, 1]), Mon(energies=[7, 7, 7])
    assert effective_damage(120, me, them) == 30 + 30 * 5


def test_spidops_scales_with_our_board():
    assert _PRINTED[560] == 0
    me, them = Mon(), Mon()
    assert effective_damage(560, me, them, bench_size=4) == 150   # 5 in play


def test_dipplin_counts_the_bench_only_not_the_active():
    """`Do the Wave` reads "for each of your BENCHED Pokemon", unlike Spidops'
    "in play" — off by one is a whole extra 20 damage, doubled by Festival
    Grounds."""
    assert _PRINTED[115] == 0
    me, them = Mon(), Mon()
    assert effective_damage(115, me, them, bench_size=5) == 100
    assert effective_damage(115, me, them, bench_size=0) == 0
    # Spidops counts the active too, so the same board gives it one more unit.
    assert effective_damage(560, me, them, bench_size=5) == 180


def test_hydrapple_counts_energy_across_the_whole_team():
    assert _PRINTED[195] == 30
    assert effective_damage(195, Mon(), Mon(), team_energy=8) == 30 + 30 * 8


def test_uncurated_attacks_return_printed_damage_untouched():
    """The table is a strict addition — anything not in it is unchanged."""
    from rl.combat import _ATK
    me, them = Mon(energies=[1, 1, 1]), Mon(energies=[2, 2])
    for aid, (printed, _cost) in _ATK.items():
        if aid in SCALING_ATTACKS:
            continue
        assert effective_damage(aid, me, them, hand_size=9) == printed


def test_estimate_never_undercuts_the_printed_number():
    me, them = Mon(), Mon()
    for aid in SCALING_ATTACKS:
        assert effective_damage(aid, me, them) >= 0


def test_is_scaling():
    assert is_scaling(1072)
    assert not is_scaling(983)                 # Mega Brave, a flat 270


def test_unknown_mode_raises_rather_than_silently_returning_printed(monkeypatch):
    monkeypatch.setitem(SCALING_ATTACKS, 1072, ("nonsense", 10, 0))
    with pytest.raises(ValueError, match="unknown scaling mode"):
        effective_damage(1072, Mon(), Mon())


# --- the pilot wiring: OFF unless asked for --------------------------------

def test_matchrunner_exposes_the_scaling_kinds():
    from rl.matchrunner import _FIXED_KINDS, parse_spec
    assert _FIXED_KINDS["generic-scale"] == ("generic", frozenset({"scaling"}))
    assert _FIXED_KINDS["solver-scale"] == ("solver", frozenset({"scaling"}))
    assert parse_spec("generic-scale:lucario") == ("generic-scale", "lucario")


def test_score_attack_defaults_to_printed_damage():
    """The shipped pilot must be untouched: scaling is opt-in, so the default
    argument has to stay False."""
    import inspect

    from rl.generic_pilot import score_attack
    assert inspect.signature(score_attack).parameters["scaling"].default is False


def test_scaling_token_changes_the_score_on_the_real_pool():
    from rl.combat import _CARD
    if 743 not in _CARD:
        pytest.skip("needs the real card pool (conftest installs a 10-card stub)")
    from rl.generic_pilot import score_attack

    class Opt:
        type, attackId, index = 13, 1072, None

    me, them = Mon(743, hp=140), Mon(648, energies=[7, 7], hp=320)
    assert score_attack(Opt(), me, them, scaling=True, hand_size=17) > \
        score_attack(Opt(), me, them)


# --- the card ladders: fetch / promote / discard / CLOSE MODE ---------------

def test_every_mode_has_a_nominal_unit_count():
    for _aid, (mode, _per, _base) in SCALING_ATTACKS.items():
        assert mode in NOMINAL_UNITS, mode


def test_nominal_damage_leaves_flat_attacks_alone():
    from rl.combat import _ATK
    for aid, (printed, _cost) in _ATK.items():
        if aid not in SCALING_ATTACKS:
            assert nominal_damage(aid) == printed


def _q(cid, scaling):
    from rl.generic_pilot import _attacker_quality
    return _attacker_quality(cid, scaling)


def test_fetch_ladder_ranked_the_win_condition_below_its_own_bench_filler():
    """The measured misplay: menu ['Applin','Thwackey','Dipplin'] -> Thwackey.
    Dipplin's Do the Wave prints 0, so the deck's win condition scored under
    Grookey (30) and Thwackey (50)."""
    from rl.combat import _CARD
    if 93 not in _CARD:
        pytest.skip("needs the real card pool")
    assert _q(93, False) < _q(90, False)      # Dipplin  < Thwackey  (printed)
    assert _q(93, True) > _q(90, True)        # Dipplin  > Thwackey  (scaling)


def test_our_own_win_condition_outranks_its_pre_evolution_only_with_scaling():
    from rl.combat import _CARD
    if 743 not in _CARD:
        pytest.skip("needs the real card pool")
    kadabra = next(cid for cid, r in cards().items() if r["name_norm"] == "Kadabra")
    assert _q(743, False) < _q(kadabra, False)
    assert _q(743, True) > _q(kadabra, True)


def test_close_mode_no_longer_calls_a_scaling_board_harmless():
    """`_op_board_harmless` decides whether to stop developing and race. On
    printed damage a lone Alakazam or Dipplin reads as harmless."""
    from rl.combat import _CARD
    if 743 not in _CARD:
        pytest.skip("needs the real card pool")
    from rl.generic_pilot import _op_board_harmless

    class P:
        def __init__(self, i):
            self.id = i

    for cid in (743, 93):
        assert _op_board_harmless(P(cid), []) is True
        assert _op_board_harmless(P(cid), [], True) is False


def test_ladders_are_untouched_when_scaling_is_off():
    """Default-off is what keeps the shipped pilot byte-identical."""
    from rl.combat import _ATK, _CARD
    from rl.generic_pilot import _attacker_quality
    for cid in list(_CARD)[:120]:
        printed = max((_ATK[a][0] for a in _CARD[cid][3] if a in _ATK), default=0)
        assert _attacker_quality(cid) == printed


def test_over_attach_flag_skips_self_scaling_attackers():
    """Piotr, reviewing the ogerpon QC: "over-attach works in its favour since
    it scales the attack based on the attached energies". Myriad Leaf Shower is
    paid at 3 energy and gains +30 per further attachment (180 -> 300 between 3
    and 7), so `already charged` is not `saturated`. The M19 flag assumed
    flat-damage attackers and reported correct play as a defect."""
    from rl.combat import _CARD
    if 96 not in _CARD:
        pytest.skip("needs the real card pool")
    from rl.postmortem import _attach_saturated

    ogerpon = {"id": 96, "energies": [1, 1, 1, 1, 1], "hp": 210}
    cur = {"players": [{"active": [ogerpon], "bench": []},
                       {"active": [{"id": 648, "energies": [7, 7], "hp": 320}],
                        "bench": []}]}
    opt = {"inPlayArea": 4, "inPlayIndex": 0}      # AreaType.ACTIVE
    assert _attach_saturated(opt, cur, 0, 1, 3) is None


def test_over_attach_flag_still_fires_on_flat_attackers():
    """The M19 defect it was written for must still be caught."""
    from rl.combat import _CARD
    if 678 not in _CARD:
        pytest.skip("needs the real card pool")
    from rl.postmortem import _attach_saturated

    solrock = next((cid for cid, r in cards().items()
                    if r["name_norm"] == "Solrock"), None)
    if solrock is None:
        pytest.skip("Solrock not in this pool")
    loaded = {"id": solrock, "energies": [6, 6, 6, 6], "hp": 90}
    cur = {"players": [{"active": [loaded], "bench": []},
                       {"active": [{"id": 648, "energies": [], "hp": 320}],
                        "bench": []}]}
    got = _attach_saturated({"inPlayArea": 4, "inPlayIndex": 0}, cur, 0, 1, 3)
    assert got is None or "over-attach" in got   # fires iff its attack is paid
