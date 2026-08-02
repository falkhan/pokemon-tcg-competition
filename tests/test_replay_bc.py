"""M25: rl/replay_bc.py encode_decisions hand_aware dispatch.

encode_decisions is [ENGINE] (imports cg.api.to_observation_class); this test
stubs both to_observation_class and iter_replay_decisions so the real
encode_state_v2 / encode_state_v3 run on a builder-shaped observation, no
engine or raw replay JSON needed.
"""
from collections import Counter

import pytest

pytest.importorskip("numpy")

import rl.replay_bc as rb  # noqa: E402
from rl.encoders import N_STATE_IDS, N_STATE_IDS_V3  # noqa: E402


def _stub_decisions(monkeypatch):
    from tests import builders as b

    me = b.player(active=b.pokemon(1), hand=[b.hand_card(7), b.hand_card(3)])
    obs = b.observation(me=me, options=[b.option(0)])
    monkeypatch.setattr(rb, "to_observation_class", lambda d: obs)
    monkeypatch.setattr(rb, "iter_replay_decisions",
                        lambda steps, seat, drops: iter([(0, obs, [0])]))


def test_encode_decisions_default_is_board_only(monkeypatch):
    _stub_decisions(monkeypatch)
    rows = rb.encode_decisions([], seat=0, deck_ids=[1] * 60, drops=Counter())
    assert rows[0][1].shape == (N_STATE_IDS,)


def test_encode_decisions_hand_aware_widens_state_ids(monkeypatch):
    _stub_decisions(monkeypatch)
    rows = rb.encode_decisions([], seat=0, deck_ids=[1] * 60, drops=Counter(),
                               hand_aware=True)
    assert rows[0][1].shape == (N_STATE_IDS_V3,)
    assert sorted(rows[0][1][12:14].tolist()) == [3, 7]


def test_iter_replay_decisions_truncated_next_step_is_a_drop():
    """A malformed replay whose next step has fewer per-agent entries must
    count as no_next_action, not crash the whole build with IndexError."""
    obs = {"select": {"option": [{}, {}], "context": 0},
           "current": {"players": [{}, {}]}}
    steps = [
        [{"status": "ACTIVE", "observation": obs},
         {"status": "ACTIVE", "observation": obs}],
        [{"status": "ACTIVE", "observation": obs, "action": [0]}],  # seat 1 gone
    ]
    drops = Counter()
    assert list(rb.iter_replay_decisions(steps, seat=1, drops=drops)) == []
    assert drops["no_next_action"] == 1
    # The intact seat still yields its decision.
    drops0 = Counter()
    rows = list(rb.iter_replay_decisions(steps, seat=0, drops=drops0))
    assert [(i, a) for i, _, a in rows] == [(0, [0])]


# ---------------------------------------------------------------------------
# M39: --opp-deck-hash, the matchup-specific corpus filter
# ---------------------------------------------------------------------------
def _stub_build_world(monkeypatch, tmp_path, decks_by_episode):
    """Drive build() over fake episodes without engine or replay JSON.

    decks_by_episode: {episode_id: (seat0_deck, seat1_deck)}; a deck of None
    models a replay whose decklist could not be extracted.
    """
    import types

    meta = {eid: {"status_0": "DONE", "status_1": "DONE", "n_steps": 50,
                  "submission_id_0": 111, "submission_id_1": 222,
                  "updated_score_0": 900.0, "updated_score_1": 900.0,
                  "our_seat": None}
            for eid in decks_by_episode}
    monkeypatch.setattr(rb, "_episode_meta", lambda: meta)
    monkeypatch.setattr(rb, "_cached_episodes",
                        lambda: [(eid, tmp_path / f"e{eid}.gz")
                                 for eid in decks_by_episode])
    monkeypatch.setattr(rb, "_load_raw", lambda p: {"steps": []})
    monkeypatch.setattr(
        rb, "parse_episode",
        lambda raw, episode_id: types.SimpleNamespace(
            decks=list(decks_by_episode[episode_id]), rewards=[1.0, -1.0]))
    # one row per seat, so "kept" is visible as a nonzero shard write
    monkeypatch.setattr(rb, "encode_decisions",
                        lambda *a, **k: [(_np.zeros(3), _np.zeros(2),
                                          _np.zeros((1, 2)), _np.zeros((1, 2)), 0)])
    seats = []
    real_hash = rb.deck_hash
    monkeypatch.setattr(rb, "deck_hash", lambda d: real_hash(d))
    return seats


import numpy as _np  # noqa: E402


def test_opp_deck_hash_keeps_only_the_targeted_matchup(monkeypatch, tmp_path):
    """A bed built for one matchup must drop seats whose OPPONENT played a
    different deck — that filter is the whole point of the flag."""
    ours, theirs = [1] * 60, [2] * 60
    ours_h, theirs_h = rb.deck_hash(ours), rb.deck_hash(theirs)
    _stub_build_world(monkeypatch, tmp_path,
                      {1: (theirs, ours), 2: (theirs, theirs)})

    rb.build(out_dir=tmp_path / "matched", min_score=0.0,
             only_deck_hash=theirs_h[:8], opp_deck_hash=(ours_h[:8],))
    rb.build(out_dir=tmp_path / "unfiltered", min_score=0.0,
             only_deck_hash=theirs_h[:8])

    matched = list((tmp_path / "matched").glob("shard_*.npz"))
    unfiltered = list((tmp_path / "unfiltered").glob("shard_*.npz"))
    assert matched and unfiltered, "both builds must produce shards"
    n_matched = sum(len(_np.load(f, allow_pickle=True)["labels"]) for f in matched)
    n_unfiltered = sum(len(_np.load(f, allow_pickle=True)["labels"]) for f in unfiltered)
    # episode 1 has the target matchup; episode 2 is theirs-vs-theirs and both
    # of its seats must be excluded by the opponent filter.
    assert n_matched < n_unfiltered
    assert n_matched == 1


def test_opp_deck_hash_drops_seats_with_an_unextractable_opponent_deck(
        monkeypatch, tmp_path):
    """An opponent whose decklist is missing cannot be verified as the target
    matchup, so it must be DROPPED, never silently kept."""
    ours, theirs = [1] * 60, [2] * 60
    ours_h, theirs_h = rb.deck_hash(ours), rb.deck_hash(theirs)
    _stub_build_world(monkeypatch, tmp_path, {1: (theirs, None)})

    rb.build(out_dir=tmp_path / "out", min_score=0.0,
             only_deck_hash=theirs_h[:8], opp_deck_hash=(ours_h[:8],))
    shards = list((tmp_path / "out").glob("shard_*.npz"))
    kept = sum(len(_np.load(f, allow_pickle=True)["labels"]) for f in shards)
    assert kept == 0
