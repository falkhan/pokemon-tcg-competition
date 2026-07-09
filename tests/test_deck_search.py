"""Old-vs-new parity: rl/deck_search.py vs tcg/deck_search.py.

matchup() needs the real engine — import-smoke only. Everything else (the
legality checker, both mutation primitives, and the tournament/search loops
with a monkeypatched deterministic matchup) is pinned, exploiting that seeded
``random`` call sequences are identical in both copies.
"""
import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.deck_search as old
import tcg.deck_search as new
from tcg.decks import load_deck

LUCARIO = load_deck("lucario")
IONO = load_deck("iono")
KYOGRE = load_deck("kyogre")


def test_card_tables_parity():
    assert new.ALL_CARD_IDS == old.ALL_IDS
    assert new.CARD_ROWS == old._ft
    assert (new.DECK_SIZE, new.MAX_COPIES) == (old.DECK_SIZE, old.MAX_COPIES)


@pytest.mark.parametrize("deck", [LUCARIO, IONO, KYOGRE])
def test_validate_deck_parity_repo_decks(deck):
    assert old.validate_deck(deck) == new.validate_deck(deck)
    assert new.validate_deck(deck)[0]


def test_validate_deck_parity_illegal_cases():
    non_energy = next(i for i in LUCARIO if not new.CARD_ROWS[i]["is_basic_energy"])
    cases = [
        LUCARIO[:59],                       # wrong size
        [non_energy] * 5 + LUCARIO[5:],     # 5+ copies of one non-energy name
        LUCARIO[:59] + [999999],            # unknown id
        [next(i for i in new.ALL_CARD_IDS
              if not new.CARD_ROWS[i]["is_pokemon"])] * 60,  # no Basic Pokémon
    ]
    ace_specs = [i for i in new.ALL_CARD_IDS if new.CARD_ROWS[i]["is_ace_spec"]]
    if len(ace_specs) >= 2:
        cases.append(LUCARIO[:58] + ace_specs[:2])          # two ACE SPECs
    for deck in cases:
        old_verdict = old.validate_deck(deck)
        new_verdict = new.validate_deck(deck)
        assert old_verdict == new_verdict
        assert not new_verdict[0]


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_swaps", [None, 2])
def test_mutate_parity(seed, n_swaps):
    random.seed(seed)
    old_deck = old.mutate(LUCARIO, n_swaps=n_swaps)
    random.seed(seed)
    new_deck = new.mutate(LUCARIO, n_swaps=n_swaps)
    assert old_deck == new_deck
    assert new.validate_deck(new_deck)[0]


def test_mutate_parity_with_candidate_weights():
    weights = {card_id: 1.0 for card_id in LUCARIO}
    random.seed(5)
    old_deck = old.mutate(LUCARIO, candidate_weights=weights)
    random.seed(5)
    new_deck = new.mutate(LUCARIO, candidate_weights=weights)
    assert old_deck == new_deck


@pytest.mark.parametrize("seed", [0, 3])
def test_mutate_flex_parity(seed):
    random.seed(seed)
    old_deck = old.mutate_flex(LUCARIO)
    random.seed(seed)
    new_deck = new.mutate_flex(LUCARIO)
    assert old_deck == new_deck
    assert new.validate_deck(new_deck)[0]
    # The Pokémon core is untouched — only flex (non-Pokémon) slots changed.
    for original, mutated in zip(LUCARIO, new_deck):
        if new.CARD_ROWS[original]["is_pokemon"]:
            assert mutated == original


def fake_matchup(deck_a, deck_b, n_games=12, agent="lucario"):
    """Deterministic: the deck with the larger id-sum wins every game."""
    winner = 0 if sum(deck_a) >= sum(deck_b) else 1
    return [winner] * n_games


def test_rate_population_parity(monkeypatch):
    pytest.importorskip("openskill")
    monkeypatch.setattr(old, "matchup", fake_matchup)
    monkeypatch.setattr(new, "matchup", fake_matchup)
    decks = [LUCARIO, IONO, KYOGRE, list(reversed(LUCARIO))]
    old_ordinals = old.rate_population(decks, n_rounds=3, seed=11)
    new_ordinals = new.rate_population(decks, n_rounds=3, seed=11)
    assert old_ordinals == new_ordinals


def test_hill_climb_parity(monkeypatch):
    monkeypatch.setattr(old, "matchup", fake_matchup)
    monkeypatch.setattr(new, "matchup", fake_matchup)
    random.seed(21)
    old_result = old.hill_climb(LUCARIO, proposals=6, games=10, seed=13)
    random.seed(21)
    new_result = new.hill_climb(LUCARIO, proposals=6, games=10, seed=13)
    assert old_result == new_result


def test_evolve_parity(monkeypatch):
    pytest.importorskip("openskill")
    monkeypatch.setattr(old, "matchup", fake_matchup)
    monkeypatch.setattr(new, "matchup", fake_matchup)
    random.seed(31)
    old_result = old.evolve(LUCARIO, pop_size=4, generations=2, seed=17)
    random.seed(31)
    new_result = new.evolve(LUCARIO, pop_size=4, generations=2, seed=17)
    assert old_result == new_result
