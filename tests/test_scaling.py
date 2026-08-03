"""The curated scaling-damage table (rl/scaling.py).

A first draft of that table guessed two of its three attack ids and pointed at
Trapinch and Magcargo ex instead of the cards named in its own comments. The
id checks here re-derive every entry from the card PARQUETS — which carry the
whole 1267-card pool and are readable without the engine, unlike `rl.combat`'s
`_CARD`/`_ATK`, which `tests/conftest.py` replaces with a 10-card stub.
"""
import pytest

from rl.scaling import SCALING_ATTACKS, effective_damage, is_scaling
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

def test_the_table_covers_exactly_the_documented_entries():
    assert set(SCALING_ATTACKS) == set(EXPECTED_OWNER)


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
    valid = {"flat", "hand", "opp_nrg", "my_nrg", "both_nrg", "my_bench",
             "bench_only", "team_nrg"}
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
