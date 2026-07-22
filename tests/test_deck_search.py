"""Unit pins for rl/deck_search.py (legality checker, mutations, search loops).

Ported from the retired old-vs-new parity suite when tcg/deck_search.py was
removed — parity assertions became invariant + seeded-determinism pins.
matchup() needs the real engine — import-smoke only; the tournament/search
loops run against a monkeypatched deterministic matchup.
"""
import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.deck_search as ds
from tcg.decks import load_deck

LUCARIO = load_deck("lucario")
IONO = load_deck("iono")
KYOGRE = load_deck("kyogre")


def test_card_tables():
    assert (ds.DECK_SIZE, ds.MAX_COPIES) == (60, 4)
    assert len(ds.ALL_IDS) > 0
    row = ds._ft[ds.ALL_IDS[0]]
    for key in ("is_basic_energy", "is_pokemon", "is_ace_spec"):
        assert key in row


@pytest.mark.parametrize("deck", [LUCARIO, IONO, KYOGRE])
def test_validate_deck_repo_decks(deck):
    ok, reasons = ds.validate_deck(deck)
    assert ok
    assert reasons == []


def test_validate_deck_illegal_cases():
    non_energy = next(i for i in LUCARIO if not ds._ft[i]["is_basic_energy"])
    cases = [
        LUCARIO[:59],                       # wrong size
        [non_energy] * 5 + LUCARIO[5:],     # 5+ copies of one non-energy name
        LUCARIO[:59] + [999999],            # unknown id
        [next(i for i in ds.ALL_IDS
              if not ds._ft[i]["is_pokemon"])] * 60,  # no Basic Pokémon
    ]
    ace_specs = [i for i in ds.ALL_IDS if ds._ft[i]["is_ace_spec"]]
    if len(ace_specs) >= 2:
        cases.append(LUCARIO[:58] + ace_specs[:2])    # two ACE SPECs
    for deck in cases:
        ok, reasons = ds.validate_deck(deck)
        assert not ok
        assert reasons


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_swaps", [None, 2])
def test_mutate(seed, n_swaps):
    random.seed(seed)
    deck = ds.mutate(LUCARIO, n_swaps=n_swaps)
    assert ds.validate_deck(deck)[0]
    assert len(deck) == ds.DECK_SIZE
    random.seed(seed)
    assert ds.mutate(LUCARIO, n_swaps=n_swaps) == deck   # seeded determinism


def test_mutate_with_candidate_weights():
    weights = {card_id: 1.0 for card_id in LUCARIO}
    random.seed(5)
    deck = ds.mutate(LUCARIO, candidate_weights=weights)
    assert ds.validate_deck(deck)[0]
    random.seed(5)
    assert ds.mutate(LUCARIO, candidate_weights=weights) == deck


@pytest.mark.parametrize("seed", [0, 3])
def test_mutate_flex(seed):
    random.seed(seed)
    deck = ds.mutate_flex(LUCARIO)
    assert ds.validate_deck(deck)[0]
    # The Pokémon core is untouched — only flex (non-Pokémon) slots changed.
    for original, mutated in zip(LUCARIO, deck):
        if ds._ft[original]["is_pokemon"]:
            assert mutated == original
    random.seed(seed)
    assert ds.mutate_flex(LUCARIO) == deck


def fake_matchup(deck_a, deck_b, n_games=12, agent="lucario"):
    """Deterministic: the deck with the larger id-sum wins every game."""
    winner = 0 if sum(deck_a) >= sum(deck_b) else 1
    return [winner] * n_games


def test_rate_population(monkeypatch):
    pytest.importorskip("openskill")
    monkeypatch.setattr(ds, "matchup", fake_matchup)
    decks = [LUCARIO, IONO, KYOGRE, list(reversed(LUCARIO))]
    ordinals = ds.rate_population(decks, n_rounds=3, seed=11)
    assert len(ordinals) == len(decks)
    assert all(isinstance(o, float) for o in ordinals)
    # Self-seeded: a rerun reproduces the exact ratings.
    assert ds.rate_population(decks, n_rounds=3, seed=11) == ordinals


def test_hill_climb(monkeypatch):
    monkeypatch.setattr(ds, "matchup", fake_matchup)
    random.seed(21)
    champ, accepted, log = ds.hill_climb(LUCARIO, proposals=6, games=10,
                                         seed=13)
    assert ds.validate_deck(champ)[0]
    random.seed(21)
    assert ds.hill_climb(LUCARIO, proposals=6, games=10, seed=13) == \
           (champ, accepted, log)


def test_evolve(monkeypatch):
    pytest.importorskip("openskill")
    monkeypatch.setattr(ds, "matchup", fake_matchup)
    random.seed(31)
    best_deck, best_ordinal, history = ds.evolve(LUCARIO, pop_size=4,
                                                 generations=2, seed=17)
    assert ds.validate_deck(best_deck)[0]
    random.seed(31)
    assert ds.evolve(LUCARIO, pop_size=4, generations=2, seed=17) == \
           (best_deck, best_ordinal, history)
