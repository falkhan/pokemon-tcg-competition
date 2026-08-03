"""Card pool facts the deck lab rests on (tcg/cardpool.py).

Most of these pin a data trap rather than a behaviour: the naive reading of
cards_features is plausible and wrong in four separate places, and each one
silently produces a deck builder that lies about the card pool.
"""
import polars as pl
import pytest

from tcg import cardpool as cp


# --- the attack join -------------------------------------------------------

def test_attack_join_covers_every_card_and_every_attack():
    """The cumulative-n_attacks walk is the only link between the two parquets;
    a re-export that shifts it must fail loudly, not silently mis-assign."""
    atk = cp.attacks_by_card()
    assert len(atk) == len(cp.cards())
    assigned = [a["attackId"] for rows in atk.values() for a in rows]
    assert len(assigned) == 1556
    assert sorted(assigned) == list(range(1, 1557))       # contiguous, no reuse


def test_attack_join_matches_known_cards():
    atk = cp.attacks_by_card()
    lucario = next(cid for cid, r in cp.cards().items()
                   if r["name_norm"] == "Mega Lucario ex")
    assert [a["attackId"] for a in atk[lucario]] == [982, 983]
    assert max(a["damage"] for a in atk[lucario]) == 270
    # A non-Pokemon has no attacks at all.
    boomerang = next(cid for cid, r in cp.cards().items()
                     if r["name_norm"] == "Boomerang Energy")
    assert atk[boomerang] == []


def test_energy_cost_string_puts_colorless_last():
    assert cp.energy_cost_string({"cost_Fighting": 2, "cost_Colorless": 1}) == "FFC"
    assert cp.energy_cost_string({"cost_Psychic": 1}) == "P"
    assert cp.energy_cost_string({}) == ""


# --- name normalisation and owner themes -----------------------------------

def test_norm_name_handles_curly_and_mojibake():
    assert cp.norm_name("Hop’s Snorlax") == "Hop's Snorlax"
    assert cp.norm_name("Hopâ€™s Snorlax") == "Hop's Snorlax"
    assert cp.norm_name("Hop's Snorlax") == "Hop's Snorlax"


def test_owners_are_pokemon_themes_not_trainer_possessives():
    """The owner regex over the WHOLE pool also matches 'Boss's Orders' and
    'Hero's Cape', which are cards, not archetype themes."""
    owners = cp.owners()
    assert "Team Rocket" in owners and "Marnie" in owners and "Cynthia" in owners
    for not_a_theme in ("Boss", "Hero", "Emcee"):
        assert not_a_theme not in owners
    assert len(owners) == 13


def test_owner_is_none_for_unowned_pokemon():
    ft = cp.cards()
    lucario = next(r for r in ft.values() if r["name_norm"] == "Mega Lucario ex")
    assert lucario["owner"] is None
    grim = next(r for r in ft.values() if r["name_norm"] == "Marnie's Grimmsnarl ex")
    assert grim["owner"] == "Marnie"


# --- evolution, by name ----------------------------------------------------

def test_pre_evo_names_is_total_and_single_valued():
    pe = cp.pre_evo_names()
    assert len(pe) == 399
    names = {r["name_norm"] for r in cp.cards().values()}
    assert set(pe).issubset(names) and set(pe.values()).issubset(names)


def test_pre_evo_follows_this_pools_lines_not_the_real_game():
    """Mega Lucario ex evolves from Riolu HERE (printing #974) — not from
    Lucario as the physical card game would have it. The pool is the truth."""
    assert cp.pre_evo_names()["Mega Lucario ex"] == "Riolu"
    assert cp.pre_evo_names()["Marnie's Grimmsnarl ex"] == "Marnie's Morgrem"


def test_multi_printing_names_exist():
    """154 names have >1 printing and the 4-copy rule is per NAME, so the deck
    editor must total across printings."""
    multi = {n: ids for n, ids in cp.printings_by_name().items() if len(ids) > 1}
    assert len(multi) == 154
    assert len(cp.printings_by_name()["Riolu"]) > 1


# --- tiers: the disjointness trap ------------------------------------------

def test_ex_and_mega_ex_are_disjoint():
    df = cp.cards_frame()
    assert df.filter(pl.col("tier") == "ex").height == 121
    assert df.filter(pl.col("tier") == "Mega ex").height == 30
    ft = cp.cards()
    assert not any(r["is_ex"] and r["is_mega_ex"] for r in ft.values())


# --- filters ---------------------------------------------------------------

def test_filter_by_owner():
    got = cp.filter_cards(cp.CardFilter(owner="Marnie"))
    assert got.height == 8
    assert all(n.startswith("Marnie's") for n in got["name"])


def test_filter_by_energy_type_excludes_trainers():
    """Trainers all carry energy_type_id 0; an unguarded Colorless filter
    returns 303 rows of which 199 are Trainers."""
    got = cp.filter_cards(cp.CardFilter(energy_types=(0,)))
    assert got.filter(pl.col("kind") == "Trainer").height == 0
    assert got.filter(pl.col("is_pokemon")).height == 104


def test_damage_floor_keeps_variable_attackers_by_default():
    """Alakazam prints 10 and Team Rocket's Spidops prints 0 — both are the plan
    of a real ladder deck. A damage floor must not hide them by default."""
    hi = cp.CardFilter(damage=(150, 350))
    names = set(cp.filter_cards(hi)["name"])
    assert "Alakazam" in names and "Team Rocket's Spidops" in names
    strict = set(cp.filter_cards(cp.CardFilter(damage=(150, 350),
                                               keep_variable=False))["name"])
    assert "Alakazam" not in strict and "Team Rocket's Spidops" not in strict


def test_attack_facets_do_not_exclude_cards_without_attacks():
    """A Supporter is not filtered out for failing a damage floor it can never
    meet — otherwise every trainer vanishes the moment a slider moves."""
    got = cp.filter_cards(cp.CardFilter(damage=(200, 350)))
    assert got.filter(pl.col("card_type") == 3).height > 0


def test_ace_spec_filter():
    assert cp.filter_cards(cp.CardFilter(ace_spec_only=True)).height == 29


def test_text_search_is_case_and_apostrophe_insensitive():
    assert cp.filter_cards(cp.CardFilter(text="grimmsnarl")).height > 0
    assert cp.filter_cards(cp.CardFilter(text="Marnie’s")).height == \
        cp.filter_cards(cp.CardFilter(text="Marnie's")).height


def test_empty_filter_returns_the_whole_pool():
    assert cp.filter_cards(cp.CardFilter()).height == 1267


def test_stage_filter():
    got = cp.filter_cards(cp.CardFilter(stages=("Stage 2",)))
    assert got.height == 116
    assert set(got["stage"]) == {"Stage 2"}


# --- engine text degrades, never raises ------------------------------------

def test_engine_text_accessors_survive_the_fake_engine():
    """tests/fake_cg.py builds attacks with no name/text and cards with no
    skills, and conftest installs it unconditionally — so these accessors must
    getattr-default rather than raise."""
    texts = cp.attack_texts()
    abilities = cp.card_abilities()
    assert isinstance(texts, dict) and isinstance(abilities, dict)
    for v in texts.values():
        assert isinstance(v["name"], str) and isinstance(v["text"], str)


def test_cards_does_not_mutate_the_ship_gate_table():
    """cards() derives columns; CARD_ROWS is tcg.deck_search's validator table."""
    from tcg.deck_search import CARD_ROWS
    cp.cards()
    assert "name_norm" not in CARD_ROWS[1]
    assert "owner" not in CARD_ROWS[1]


def test_card_label_is_stable_and_handles_unknown():
    assert "#678" in cp.card_label(678)
    assert "unknown" in cp.card_label(999_999)


# --- display vocabulary ----------------------------------------------------

def test_every_energy_type_has_a_colour_and_a_symbol():
    """Colour is never the only cue (the palette carries two CVD WARNs), so a
    type without a symbol would leave some readers with nothing."""
    for name in cp.ENERGY_TYPES.values():
        assert name in cp.ENERGY_COLORS, name
        assert cp.ENERGY_EMOJI.get(name), name


def test_card_icon_covers_the_whole_pool():
    """No card may render as a blank glyph in the browser."""
    for cid, r in cp.cards().items():
        assert cp.card_icon(r), cid


def test_card_icon_distinguishes_pokemon_from_trainers():
    ft = cp.cards()
    grim = next(r for r in ft.values() if r["name_norm"] == "Marnie's Grimmsnarl ex")
    assert cp.card_icon(grim) == cp.ENERGY_EMOJI["Darkness"]
    boss = next(r for r in ft.values() if r["name_norm"].startswith("Boss's Orders"))
    assert cp.card_icon(boss) == cp.TRAINER_EMOJI[boss["card_type"]]


def test_browse_frame_carries_icon_and_badge_columns():
    df = cp.cards_frame()
    assert "icon" in df.columns and "badge" in df.columns
    row = df.filter(pl.col("name") == "Mega Lucario ex").to_dicts()[0]
    assert row["icon"] == cp.ENERGY_EMOJI["Fighting"]
    assert cp.TIER_EMOJI["Mega ex"] in row["badge"]


def test_ace_spec_cards_are_badged():
    df = cp.cards_frame().filter(pl.col("card_id").is_in(
        [cid for cid, r in cp.cards().items() if r["is_ace_spec"]]))
    assert df.height == 29
    assert all("🅰" in b for b in df["badge"])


def test_energy_color_falls_back_for_untyped_cards():
    assert cp.energy_color(None) == "#8a8a8a"
    assert cp.energy_emoji(None) == ""
