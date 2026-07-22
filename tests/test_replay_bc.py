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
