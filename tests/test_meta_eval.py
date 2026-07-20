"""M22b: meta-eval retargeted to the off-mirror tail.

The old form played every snapshot deck at equal n and reported a
manifest-weighted mean whose dominant cell (weight 0.899) is byte-identical to
our own deck — i.e. it re-measured the mirror gate at n=60 and the disagreement
between the two replicates was read as a mirror-vs-meta trade for two
milestones. These tests pin the selection so that cannot come back.
"""
import json

import pytest

import rl.matchrunner as mr
import rl.replay_bc as rb

MIRROR = [1, 2, 3]          # the deck our candidate pilots
OTHER = [4, 5, 6]
TINY = [7, 8, 9]


@pytest.fixture
def snap(tmp_path, monkeypatch):
    """A snapshot whose dominant deck IS the mirror deck — the meta_v2 shape."""
    (tmp_path / "a.csv").write_text(" ".join(map(str, MIRROR)))
    (tmp_path / "b.csv").write_text(" ".join(map(str, OTHER)))
    (tmp_path / "c.csv").write_text(" ".join(map(str, TINY)))
    (tmp_path / "manifest.json").write_text(json.dumps({"decks": [
        {"archetype": "mirror_archetype", "csv": "a.csv", "weight": 900.0, "n_games": 3521},
        {"archetype": "real_tail", "csv": "b.csv", "weight": 90.0, "n_games": 347},
        {"archetype": "singleton", "csv": "c.csv", "weight": 1.0, "n_games": 1},
    ]}))

    def fake_resolve(deck):
        if isinstance(deck, (list, tuple)):
            return list(deck)
        name = str(deck)
        if name.endswith("a.csv"):
            return MIRROR
        if name.endswith("b.csv"):
            return OTHER
        if name.endswith("c.csv"):
            return TINY
        return MIRROR                      # "lucario" -> the mirror deck
    monkeypatch.setattr(mr, "resolve_deck", fake_resolve)
    return tmp_path


@pytest.fixture
def captured(monkeypatch):
    """Capture the pairs meta_eval would play, without playing them."""
    box = {}

    def fake_run_pairs(pairs, workers=4, game_fn=None, seed=0, checkpoint=None):
        box["pairs"] = pairs
        return [[0] * n for *_, n in pairs]     # 0 == side-a WIN
    monkeypatch.setattr(mr, "run_pairs", fake_run_pairs)
    return box


def _played(box):
    return {str(p[1][1]).rsplit("/", 1)[-1]: p[2] for p in box["pairs"]}


class TestSelection:
    def test_drops_the_deck_identical_to_our_own(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert "mirror_archetype" in res["dropped"]
        assert "a.csv" not in _played(captured)

    def test_drops_archetypes_seeded_on_too_few_games(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert "singleton" in res["dropped"], "n_games=1 is a sampling artifact"
        assert "c.csv" not in _played(captured)

    def test_keeps_the_genuine_tail(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert "real_tail" in res["per_deck"]
        assert "b.csv" in _played(captured)

    def test_include_mirror_restores_the_old_behaviour(self, snap, captured):
        rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1,
                     exclude_mirror=False)
        assert "a.csv" in _played(captured), "opt-in escape hatch must still work"


class TestBudget:
    def test_whole_budget_goes_to_the_survivors(self, snap, captured):
        """Same compute as the old form, concentrated where mirror cannot see."""
        rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        played = _played(captured)
        assert sum(played.values()) == pytest.approx(60 * 3, abs=3), \
            "budget is n * len(all decks), not n * len(survivors)"
        assert played["b.csv"] > 60, "the surviving cell must gain precision"

    def test_no_survivors_returns_no_score_rather_than_a_fake_one(self, snap, captured,
                                                                 monkeypatch):
        (snap / "manifest.json").write_text(json.dumps({"decks": [
            {"archetype": "mirror_archetype", "csv": "a.csv", "weight": 1.0,
             "n_games": 999}]}))
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert res["tail_score"] is None and res["coverage"] == 0.0


class TestReporting:
    def test_coverage_is_reported_and_is_not_the_whole_field(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert 0.0 < res["coverage"] < 0.5, \
            "tail must never look like a whole-field score"

    def test_per_deck_carries_a_confidence_interval(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        cell = res["per_deck"]["real_tail"]
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["wr"] <= hi <= 1.0 and cell["n"] > 0

    def test_no_weighted_scalar_masquerading_as_the_old_meta_score(self, snap, captured):
        res = rb.meta_eval("model:ck.pt:lucario", snapshot=snap, n=60, workers=1)
        assert "meta_score" not in res, \
            "the laundered whole-field scalar must not come back under its old name"
