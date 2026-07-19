"""M13 rl/setup_value.py: matched-pair construction invariants."""
import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

import rl.setup_value as sv


def _rows():
    # 8 states across 4 games: two buckets; games 0/2 won, 1/3 lost
    return {
        "states": np.zeros((8, 6), np.float32),
        "state_ids": np.zeros((8, 3), np.int32),
        "turn": np.array([4, 4, 4, 4, 9, 9, 1, 4], np.int32),
        "my_prizes": np.array([6, 6, 6, 6, 4, 4, 6, 6], np.int32),
        "opp_prizes": np.array([6, 6, 6, 6, 5, 5, 6, 6], np.int32),
        "game_ids": np.array([0, 1, 2, 3, 0, 1, 2, 0], np.int32),
        "results": np.array([1, -1, 1, -1, 1, -1, 1, 0], np.float32),
        "deck_idx": np.zeros(8, np.int32),
    }


def test_pairs_match_context_and_direction():
    pairs, buckets = sv.build_pairs(_rows(), random.Random(0))
    rows = _rows()
    assert pairs, "expected at least one pair"
    for w, l in pairs:
        assert rows["results"][w] > 0 > rows["results"][l]
        assert rows["game_ids"][w] != rows["game_ids"][l]
        assert rows["turn"][w] // sv.TURN_BUCKET == \
            rows["turn"][l] // sv.TURN_BUCKET
        assert rows["my_prizes"][w] == rows["my_prizes"][l]
        assert rows["opp_prizes"][w] == rows["opp_prizes"][l]


def test_min_turn_draws_and_exclusions_filtered():
    rows = _rows()
    pairs, _ = sv.build_pairs(rows, random.Random(0))
    used = {i for p in pairs for i in p}
    assert 6 not in used                      # turn 1 < MIN_TURN
    assert 7 not in used                      # draw (result 0)
    # excluding the winning games removes every pair source
    pairs2, _ = sv.build_pairs(rows, random.Random(0),
                               exclude_games=np.array([0, 2]))
    assert not pairs2
