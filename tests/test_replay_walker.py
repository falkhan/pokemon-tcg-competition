"""M40 S5: the full-prompt log walker and the v4 encode path.

`rl.memory.OppMemory` requires observe() to see EVERY own prompt's log window
exactly once — its prefix-dedupe cancels RE-DELIVERED events (the sub-prompt
chain case) but a window that is never delivered is lost permanently, and every
later feature for that seat is wrong. `iter_replay_decisions` skips six prompt
classes, so it cannot be the iterator a v4 corpus is built on.

The refactor risk is the other direction: thirteen call sites depend on
iter_replay_decisions' exact behaviour, including its drop taxonomy and
counters. The first three tests pin that it did not move.
"""
from collections import Counter

import numpy as np
import pytest

import rl.replay_bc as rb
from rl.deck_search import DECK_SIZE
from rl.replay_bc import (encode_decisions, encode_decisions_v4,
                          iter_replay_decisions, iter_replay_prompts)


def _obs(*, select=True, options=2, context=0, turn=1, logs=()):
    """A raw replay observation — a DICT, which is what the walker must handle
    and what OppMemory.observe consumes natively (rl/memory.py's _g supports
    both raw dicts and cg.api.Log)."""
    o = {"current": {"players": [{}, {}], "yourIndex": 0, "turn": turn},
         "logs": list(logs), "step": 0}
    if select:
        o["select"] = {"option": [{}] * options, "maxCount": 1,
                       "context": context}
    return o


def _steps(specs):
    """specs: list of (obs, answer). The answer to a prompt lives in the NEXT
    step's `action` field — that offset IS the alignment rule the iterator
    implements, so the fixture has to reproduce it rather than flatten it."""
    steps = [[{"status": "ACTIVE", "observation": obs}, {"status": "INACTIVE"}]
             for obs, _ in specs]
    # One trailing INACTIVE step so the LAST prompt has somewhere to be
    # answered; without it every fixture's final prompt is a no_next_action
    # drop and the assertions quietly measure that instead.
    steps.append([{"status": "INACTIVE"}, {"status": "INACTIVE"}])
    for i, (_obs_i, answer) in enumerate(specs):
        if answer is not None:
            steps[i + 1][0]["action"] = answer
    return steps


@pytest.fixture(autouse=True)
def _stub_converter(monkeypatch):
    """The fake engine's to_observation_class is a passthrough, so a raw dict
    stays a dict and the encoders (which want attribute access) fault. Map it
    to a builder-shaped observation, preserving the fields the encode path
    reads. Memory still consumes the RAW dict, which is the real contract."""
    from tests import builders as b

    def convert(d):
        sel = d.get("select") or {}
        return b.observation(
            me=b.player(active=b.pokemon(1), hand=[b.hand_card(7)]),
            opponent=b.player(active=b.pokemon(4)),
            turn=d.get("current", {}).get("turn", 1),
            options=[b.option(0)] * max(1, len(sel.get("option", [{}]))),
            context=sel.get("context", 0))

    monkeypatch.setattr(rb, "to_observation_class", convert)


# --- the refactor did not move iter_replay_decisions ------------------------

def test_decisions_is_exactly_the_unfiltered_walker():
    steps = _steps([
        (_obs(), [0]),                          # usable
        (_obs(select=False), [1]),              # blank_obs
        (_obs(options=0), [0]),                 # empty_menu
        (_obs(), list(range(DECK_SIZE))),       # deck_return
        (_obs(), [99]),                         # LABEL_OUT_OF_RANGE
        (_obs(), None),                         # no_next_action (nothing after)
    ])
    d_drops, w_drops = Counter(), Counter()
    dec = list(iter_replay_decisions(steps, 0, d_drops))
    walk = [(i, o, a) for i, o, a, r in iter_replay_prompts(steps, 0, w_drops)
            if not r]
    assert dec == walk
    assert d_drops == w_drops, "the drop taxonomy moved"


def test_walker_is_a_strict_superset_and_labels_every_drop():
    steps = _steps([(_obs(), [0]), (_obs(select=False), [0]), (_obs(), [99]),
                    (_obs(), [0])])
    drops = Counter()
    rows = list(iter_replay_prompts(steps, 0, drops))
    assert len(rows) == 4
    assert [r[3] for r in rows] == ["", "blank_obs", "LABEL_OUT_OF_RANGE", ""]
    assert sum(drops.values()) == 2


def test_walker_keeps_the_ACTIVE_boundary():
    # An INACTIVE seat repeats a stale select AND stale logs (the kaggle
    # interpreter refreshes observation.logs only for the ACTIVE seat), so it
    # is not a prompt and must never be observed — double-counting its window
    # would corrupt memory exactly as skipping one does.
    steps = _steps([(_obs(), [0])])
    steps.append([{"status": "INACTIVE", "observation": _obs()}, {}])
    assert len(list(iter_replay_prompts(steps, 0, Counter()))) == 1


# --- the v4 consumer --------------------------------------------------------

def test_v4_rows_match_v3_rows_one_for_one():
    """Row count and ordering must be identical to encode_decisions, or a v4
    rebuild is not comparable with the v3 corpus it replaces."""
    steps = _steps([(_obs(turn=1), [0]), (_obs(select=False), [0]),
                    (_obs(turn=2), [1]), (_obs(turn=2), [0])])
    v3 = encode_decisions(steps, 0, list(range(60)), Counter())
    v4 = encode_decisions_v4(steps, 0, list(range(60)), Counter())
    assert len(v3) == len(v4)
    assert [r[4] for r in v3] == [r[4] for r in v4], "labels diverged"


def test_v4_widths_are_the_v4_encoder_widths():
    from rl.encoders import N_CONTEXTS, N_STATE_IDS_V4, STATE_V2_DIM, V4_EXTRA_DIM
    steps = _steps([(_obs(), [0])])
    rows = encode_decisions_v4(steps, 0, list(range(60)), Counter())
    assert rows
    state_ctx, state_ids, _opts, _oids, _label = rows[0]
    assert state_ctx.shape[0] == STATE_V2_DIM + N_CONTEXTS + V4_EXTRA_DIM
    assert state_ids.shape[0] == N_STATE_IDS_V4


def test_memory_is_observed_on_dropped_prompts_too():
    """The whole point of the lane. A prompt the label iterator skips still
    delivers a log window, and that window must be consumed."""
    steps = _steps([
        (_obs(turn=1, logs=[{"type": 15, "playerIndex": 1, "cardId": 7}]), [0]),
        (_obs(turn=1, options=0, logs=[{"type": 15, "playerIndex": 1,
                                        "cardId": 8}]), [0]),   # empty_menu
        (_obs(turn=2, logs=[{"type": 15, "playerIndex": 1, "cardId": 9}]), [0]),
    ])
    stats = Counter()
    encode_decisions_v4(steps, 0, list(range(60)), Counter(), mem_stats=stats)
    assert stats["observed"] == 3, "a dropped prompt's window was not consumed"
    assert stats["observed_but_dropped"] == 1


def test_deck_prompt_resets_and_is_not_observed():
    """Transcribed from submission/main.py: on select-None the live agent
    resets memory and RETURNS before observe(). Observing here would diverge
    from live at prompt zero of every game."""
    steps = _steps([
        (_obs(turn=1, logs=[{"type": 15, "playerIndex": 1, "cardId": 7}]), [0]),
        (_obs(select=False, logs=[{"type": 15, "playerIndex": 1}]), [0]),
        (_obs(turn=1, logs=[]), [0]),
    ])
    stats = Counter()
    encode_decisions_v4(steps, 0, list(range(60)), Counter(), mem_stats=stats)
    assert stats["reset"] == 1
    assert stats["observed"] == 2, "the deck prompt was observed"


def test_skipping_a_window_actually_changes_the_features():
    """Non-vacuity for the whole lane: if consuming a dropped prompt's window
    made no difference to the encoding, the walker would be pointless. A
    fresh window (not a prefix re-delivery) must move the memory state."""
    from rl.memory import OppMemory
    attack = {"type": 15, "playerIndex": 1, "cardId": 7, "attackId": 3}
    with_it, without_it = OppMemory(), OppMemory()
    with_it.observe(_obs(logs=[attack]))
    with_it.observe(_obs(logs=[{"type": 2, "playerIndex": 1}]))
    without_it.observe(_obs(logs=[{"type": 2, "playerIndex": 1}]))
    assert not np.array_equal(with_it.ids(), without_it.ids()) or \
        with_it._attacks_total != without_it._attacks_total, \
        "dropping a window is undetectable - the lane has no mechanism"
