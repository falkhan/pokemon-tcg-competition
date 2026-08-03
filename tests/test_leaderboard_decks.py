"""Deck identification from a decklist (scripts/leaderboard_decks.py).

Every case here is a real mislabelling caught while building the census, kept
as a regression: the naive readings of cards_features all look reasonable and
all name the wrong card. Pure and offline — no network, no engine.
"""
from pathlib import Path

import pytest

from scripts.leaderboard_decks import (ENERGY_TYPES, cards, describe_deck,
                                       preevolutions)
from tcg.decks import load_deck_file

ROOT = Path(__file__).resolve().parent.parent


def deck(rel: str) -> list[int]:
    return load_deck_file(ROOT / rel)


# --- main attacker ----------------------------------------------------------

@pytest.mark.parametrize("path,attacker,tier", [
    # The plan is the top of the evolution line, never its 4-of basic: a
    # copies-led key returns Marnie's Impidimp / Cynthia's Gabite here.
    ("decks/grim_live.csv", "Marnie's Grimmsnarl ex", "ex"),
    ("data/kaggle/garchomp_c7b3253f_deck.csv", "Cynthia's Garchomp ex", "ex"),
    # Scaling attackers print 10 (Alakazam) and 0 (Spidops) damage. Without the
    # variable-attack proxy the first loses to Kadabra and the second is not
    # even eligible, handing the deck to Team Rocket's Tarountula.
    ("decks/alakazam_v2_h4.csv", "Alakazam", "regular"),
    ("data/kaggle/rocket_3394cd30_deck.csv", "Team Rocket's Spidops", "regular"),
    # Basic Mega ex attackers must still win over evolved chaff.
    ("decks/lucario.csv", "Mega Lucario ex", "Mega ex"),
    ("decks/archaludon.csv", "Archaludon ex", "ex"),
    ("decks/iono.csv", "Iono's Bellibolt ex", "ex"),
])
def test_main_attacker(path, attacker, tier):
    d = describe_deck(deck(path))
    assert d["attacker"] == attacker
    assert d["attacker_tier"] == tier


def test_rocket_grass_deck_ignores_the_psychic_tech():
    """The regression that forced energy-consistency to rank first.

    These lists run 7-8 Grass energy behind a Grass line, plus a single Psychic
    Team Rocket's Mewtwo ex — which has the highest printed damage in the deck.
    A damage-led key names the 1-of tech as the whole deck's plan.
    """
    d = describe_deck(deck("data/kaggle/rocket_3394cd30_deck.csv"))
    assert d["primary_energy"] == "Grass"
    assert d["attacker_energy"] == "Grass"
    assert "Mewtwo" not in d["attacker"]
    assert d["family"] == "rocket"


def test_zero_damage_support_ex_is_never_the_attacker():
    """Fezandipiti ex (0 damage, ability-only) headlined 52 harvested decks."""
    ft = cards()
    fez = next(cid for cid, r in ft.items() if r["name"] == "Fezandipiti ex")
    assert ft[fez]["max_damage"] == 0 and not ft[fez]["has_variable_attack"]
    grim = deck("decks/grim_live.csv")
    assert fez not in grim or describe_deck(grim)["attacker"] != "Fezandipiti ex"


# --- pre-evolution discount (replay evidence) -------------------------------

def test_preevolutions_finds_the_lines_present_in_the_deck():
    """Pilots chip with the basic before evolving, so raw attack counts name
    Impidimp as the Grimmsnarl deck's attacker. Only lines in THIS deck count."""
    ft = cards()
    ids = deck("decks/grim_live.csv")
    skip = {ft[i]["name"] for i in preevolutions(ids)}
    assert "Marnie's Impidimp" in skip
    assert "Marnie's Grimmsnarl ex" not in skip  # nothing evolves from the top


def test_preevolutions_returns_only_cards_the_deck_evolves_past():
    """Every id returned is in the deck AND something in the deck evolves from
    it — so a lone Basic attacker (Mega Kangaskhan ex) is never discounted."""
    ft = cards()
    ids = deck("decks/lucario.csv")
    present = set(ids)
    skip = preevolutions(ids)
    assert skip <= present
    for i in skip:
        assert any(ft[j]["evolves_from_id"] == i for j in present if j in ft)


def test_preevolutions_of_empty_deck_is_empty():
    assert preevolutions([]) == set()


# --- energy identity --------------------------------------------------------

@pytest.mark.parametrize("path,energy", [
    ("decks/grim_live.csv", "Darkness"),
    ("decks/kyogre.csv", "Water"),
    ("decks/lucario.csv", "Fighting"),
    ("decks/iono.csv", "Lightning"),
    ("decks/archaludon.csv", "Metal"),
])
def test_primary_energy(path, energy):
    assert describe_deck(deck(path))["primary_energy"] == energy


def test_special_energy_only_deck_reports_no_primary_energy():
    """A real shape (some Mega lists run 0 basic energy) — must not crash."""
    d = describe_deck(deck("decks/greattusk_wall.csv"))
    assert d["primary_energy"] is None
    assert d["n_special_energy"] > 0
    assert d["attacker"] is not None  # still identifiable without an energy anchor


# --- owner theme ------------------------------------------------------------

@pytest.mark.parametrize("path,theme", [
    ("decks/grim_live.csv", "Marnie"),
    ("data/kaggle/rocket_3394cd30_deck.csv", "Team Rocket"),
    ("data/kaggle/garchomp_c7b3253f_deck.csv", "Cynthia"),
    ("decks/iono.csv", "Iono"),
    # No owner prefix anywhere in the line — must stay None, not borrow one
    # from a splashed engine card (the plurality fallback called this "Lillie").
    ("decks/lucario.csv", None),
])
def test_owner_theme(path, theme):
    assert describe_deck(deck(path))["theme"] == theme


# --- family label stays the canonical one -----------------------------------

@pytest.mark.parametrize("path,family", [
    ("decks/alakazam_v2_h4.csv", "mirror"),
    ("decks/grim_live.csv", "grim"),
    ("decks/lucario.csv", "lucario"),
    ("decks/archaludon.csv", "archaludon"),
    ("decks/hops_stall.csv", "stall"),
    ("decks/greattusk_wall.csv", "wall"),
])
def test_family_matches_the_gate_taxonomy(path, family):
    assert describe_deck(deck(path))["family"] == family


# --- shape ------------------------------------------------------------------

def test_describe_deck_is_total_over_every_reference_deck():
    """No deck in the repo may crash or come back malformed."""
    ft = cards()
    for path in sorted((ROOT / "decks").glob("*.csv")):
        ids = load_deck_file(path)
        d = describe_deck(ids)
        assert d["family"], f"{path.name} has no family"
        if d["primary_energy"] is not None:
            assert d["primary_energy"] in ENERGY_TYPES.values()
        # An attacker is required exactly when the deck can deal damage at all.
        can_attack = any((ft[i]["max_damage"] or 0) > 0 or ft[i]["has_variable_attack"]
                         for i in set(ids) if i in ft and ft[i]["is_pokemon"])
        assert bool(d["attacker"]) == can_attack, f"{path.name}: {d['attacker']!r}"
        if can_attack:
            assert d["attacker_tier"] in ("Mega ex", "ex", "Tera", "regular")


def test_deck_that_cannot_deal_damage_has_no_attacker():
    """decks/floor_zero_damage.csv is a purpose-built control: every Pokemon
    prints 0 damage. Naming an 'attacker' there would be an invention."""
    d = describe_deck(deck("decks/floor_zero_damage.csv"))
    assert d["attacker"] is None
    assert d["label"] == "unknown"


def test_unknown_card_ids_are_skipped_not_fatal():
    ids = deck("decks/grim_live.csv") + [999_999]
    assert describe_deck(ids)["attacker"] == "Marnie's Grimmsnarl ex"


def test_empty_deck_degrades_to_unknown():
    d = describe_deck([])
    assert d["label"] == "unknown"
    assert d["attacker"] is None and d["primary_energy"] is None
