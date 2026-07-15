"""rl/determinize.py — the L3 archetype determinizer (M8.4a), offline half:
inference/pool math on real deck lists + the parquet card table; the
ground-truth harness is [ENGINE] (docs/M8-plan.md M8.4a gate)."""
import random
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.determinize as dz
from rl.kaggle_ingest import _core_vec, _ft
from tests import builders as b

ROOT = Path(__file__).resolve().parent.parent


def _deck(name):
    return [int(x) for x in (ROOT / "decks" / f"{name}.csv").read_text().split()
            if x.strip()]


LUCARIO = _deck("lucario")
IONO = _deck("iono")


def _meta():
    # iono heavier on purpose: the no-reveal prior must pick it
    return [dz.MetaDeck("iono", IONO, _core_vec(IONO), 5.0),
            dz.MetaDeck("lucario", LUCARIO, _core_vec(LUCARIO), 1.0)]


def _distinctive_pokemon(deck, other):
    other_set = set(other)
    return next(i for i in deck if _ft[i]["is_pokemon"] and i not in other_set)


def test_infer_uses_field_prior_before_any_reveal():
    obs = b.observation(me=b.player(), opponent=b.player())
    assert dz.infer_deck(obs, _meta()).archetype == "iono"


def test_infer_matches_revealed_core():
    poke = _distinctive_pokemon(LUCARIO, IONO)
    obs = b.observation(opponent=b.player(active=b.pokemon(poke)))
    assert dz.infer_deck(obs, _meta()).archetype == "lucario"
    # ... and via the DISCARD alone (revealed_ids covers it)
    obs = b.observation(opponent=b.player(discard=[b.hand_card(poke)]))
    assert dz.infer_deck(obs, _meta()).archetype == "lucario"


def test_remaining_pool_subtracts_multiset_and_clamps():
    deck = [7, 7, 7, 9]
    assert Counter(dz.remaining_pool(deck, [7])) == Counter({7: 2, 9: 1})
    # revealing MORE copies than the variant runs must clamp, not go negative
    assert Counter(dz.remaining_pool(deck, [9, 9, 9])) == Counter({7: 3})


def test_determinize_kwargs_sizes_and_basic_active():
    poke = _distinctive_pokemon(LUCARIO, IONO)
    opp = b.player(active=None, hand_count=5, deck_count=30, prizes_remaining=6,
                   discard=[b.hand_card(poke)])
    opp.active = [None]                       # hidden active slot
    obs = b.observation(opponent=opp)
    kw = dz.determinize_kwargs(obs, _meta(), random.Random(0))
    assert len(kw["opponent_hand"]) == 5
    assert len(kw["opponent_prize"]) == 6
    assert len(kw["opponent_deck"]) == 30
    assert len(kw["opponent_active"]) == 1
    active = kw["opponent_active"][0]
    assert _ft[active]["is_pokemon"] and _ft[active]["is_basic"]
    # everything sampled comes from the inferred archetype's remaining pool
    pool = Counter(dz.remaining_pool(dz.infer_deck(obs, _meta()).ids, [poke]))
    sampled = Counter(kw["opponent_hand"] + kw["opponent_prize"]
                      + kw["opponent_deck"] + kw["opponent_active"])
    assert not (sampled - pool - Counter({dz.FILLER_ENERGY: 60}))


def test_determinize_pads_when_variant_runs_short():
    opp = b.player(active=b.pokemon(_distinctive_pokemon(LUCARIO, IONO)),
                   hand_count=15, deck_count=55, prizes_remaining=6)
    obs = b.observation(opponent=opp)
    kw = dz.determinize_kwargs(obs, _meta(), random.Random(1))
    total = len(kw["opponent_hand"]) + len(kw["opponent_prize"]) + len(kw["opponent_deck"])
    assert total == 15 + 55 + 6               # padded with FILLER_ENERGY
