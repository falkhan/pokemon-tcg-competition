"""scripts/build_rule_facts.py (M46 C3) — miner unit tests + seed facts.

Two layers: (1) `mine_episode` on synthetic step lists, one per table, so
the mining laws (same-turn state-diff attribution, ALL-ZERO veto, log-window
attack pairing, sub-prompt prefix dedupe) are pinned without the corpus;
(2) the COMMITTED data/rule_facts.json must contain the seed facts the plan
names — Dudunsparce self-removal, Boss's Orders gust, a Crustle zero-damage
defender — and the hand-frozen registries in rl/plan must stay subsets of
the mined tables (the constants are the servable copy of these facts).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tests.fake_cg import LogType, OptionType, SelectContext  # noqa: E402

import scripts.build_rule_facts as rf  # noqa: E402

DUDU, ABRA, KADABRA, OPP_A, OPP_B = 66, 741, 742, 500, 501
BOSS = 1182


def _player(active=None, bench=(), hand=(), deck=30):
    return {"active": [{"id": active}] if active else [],
            "bench": [{"id": b} if b else None for b in bench],
            "hand": [{"id": h} for h in hand],
            "deckCount": deck}


def _prompt(me, op, options, turn=1, logs=()):
    return {"select": {"context": int(SelectContext.MAIN), "option": options},
            "current": {"players": [me, op], "yourIndex": 0, "turn": turn},
            "logs": list(logs)}


def _steps(prompts_and_actions):
    """steps[i][0] = ACTIVE prompt; the answer lands at steps[i+1][0]
    (the replay off-by-one law). Seat 1 stays INACTIVE throughout."""
    steps = []
    for i, (obs, _) in enumerate(prompts_and_actions):
        entry = {"status": "ACTIVE", "observation": obs}
        if i > 0:
            entry["action"] = prompts_and_actions[i - 1][1]
        steps.append([entry, {"status": "INACTIVE"}])
    last = {"status": "ACTIVE", "action": prompts_and_actions[-1][1],
            "observation": {"select": None, "current": None}}
    steps.append([last, {"status": "INACTIVE"}])
    return steps


ABILITY_OPT = {"type": int(OptionType.ABILITY), "area": 4, "index": 0}
END_OPT = {"type": int(OptionType.END)}


def _play_opt(idx):
    return {"type": int(OptionType.PLAY), "index": idx, "area": 2}


def test_self_removal_mined_from_board_diff():
    before = _prompt(_player(active=DUDU, deck=30), _player(active=OPP_A),
                     [ABILITY_OPT, END_OPT])
    after = _prompt(_player(active=None, deck=28), _player(active=OPP_A),
                    [END_OPT])
    t = rf.mine_episode(_steps([(before, [0]), (after, [0])]), 7)
    assert t["self_removal"][DUDU] == 1
    assert t["self_removal_ep"][DUDU] == 7


def test_self_removal_needs_a_shrinking_board():
    # ability by the active while the board stays the same size (e.g. the
    # id vanishes because the ACTIVE evolved via a different mechanism
    # is not mined here) -> no fact
    before = _prompt(_player(active=DUDU, bench=(ABRA,)),
                     _player(active=OPP_A), [ABILITY_OPT, END_OPT])
    after = _prompt(_player(active=KADABRA, bench=(ABRA,)),
                    _player(active=OPP_A), [END_OPT])
    t = rf.mine_episode(_steps([(before, [0]), (after, [0])]))
    assert not t["self_removal"]


def test_gust_mined_from_opp_active_flip_in_our_turn():
    before = _prompt(_player(active=ABRA, hand=(BOSS,)),
                     _player(active=OPP_A), [_play_opt(0), END_OPT])
    after = _prompt(_player(active=ABRA), _player(active=OPP_B), [END_OPT])
    t = rf.mine_episode(_steps([(before, [0]), (after, [0])]))
    assert t["gust"][BOSS] == 1
    # across turns the flip is their retreat, not our gust
    after_t2 = _prompt(_player(active=ABRA), _player(active=OPP_B),
                       [END_OPT], turn=2)
    t = rf.mine_episode(_steps([(before, [0]), (after_t2, [0])]))
    assert not t["gust"]


def test_deck_cost_is_worst_case_same_turn():
    p1 = _prompt(_player(active=ABRA, hand=(BOSS, BOSS), deck=30),
                 _player(active=OPP_A), [_play_opt(0), END_OPT])
    p2 = _prompt(_player(active=ABRA, hand=(BOSS,), deck=27),
                 _player(active=OPP_A), [_play_opt(0), END_OPT])
    p3 = _prompt(_player(active=ABRA, deck=26), _player(active=OPP_A),
                 [END_OPT])
    t = rf.mine_episode(_steps([(p1, [0]), (p2, [0]), (p3, [0])]))
    assert t["deck_cost"][BOSS] == [3, 2]          # worst 3 over {3, 1}


def test_zero_damage_pairs_and_the_veto():
    atk = {"type": int(LogType.ATTACK), "cardId": ABRA}
    hp = {"type": int(LogType.HP_CHANGE), "cardId": OPP_A, "value": -30}
    end = {"type": int(LogType.TURN_END)}
    board = (_player(active=ABRA), _player(active=OPP_A))
    p1 = _prompt(*board, options=[END_OPT])                  # anchors actives
    p2 = _prompt(*board, options=[END_OPT], turn=2,
                 logs=[atk, end])                            # 0-damage attack
    p3 = _prompt(*board, options=[END_OPT], turn=3,
                 logs=[atk, hp, end])                        # damaging attack
    t = rf.mine_episode(_steps([(p1, [0]), (p2, [0]), (p3, [0])]))
    assert dict(t["dmg_obs"])[(ABRA, OPP_A)] == [1, 1]
    merged = rf._merge([{"dmg_obs": {(ABRA, OPP_A): [5, 0],
                                     (ABRA, OPP_B): [5, 1]},
                         "self_removal": rf.Counter(), "self_removal_ep": {},
                         "deck_cost": {}, "gust": rf.Counter()}])
    facts = rf.finalize(merged, 1)
    pairs = {(p["attacker"], p["defender"]) for p in facts["zero_damage_pairs"]}
    assert (ABRA, OPP_A) in pairs                  # all-zero at n>=3: in
    assert (ABRA, OPP_B) not in pairs              # one damaging obs: VETOED


def test_prefix_redelivery_counts_once():
    atk = {"type": int(LogType.ATTACK), "cardId": ABRA}
    end = {"type": int(LogType.TURN_END)}
    board = (_player(active=ABRA), _player(active=OPP_A))
    p1 = _prompt(*board, options=[END_OPT])
    p2 = _prompt(*board, options=[END_OPT], turn=2, logs=[atk, end])
    p3 = _prompt(*board, options=[END_OPT], turn=2,
                 logs=[atk, end])                  # re-delivered prefix
    t = rf.mine_episode(_steps([(p1, [0]), (p2, [0]), (p3, [0])]))
    assert dict(t["dmg_obs"])[(ABRA, OPP_A)] == [1, 0]


def test_unknown_attacker_is_skipped_not_guessed():
    atk = {"type": int(LogType.ATTACK), "cardId": 999}
    p1 = _prompt(_player(active=ABRA), _player(active=OPP_A),
                 [END_OPT])
    p2 = _prompt(_player(active=ABRA), _player(active=OPP_A),
                 [END_OPT], turn=2, logs=[atk])
    t = rf.mine_episode(_steps([(p1, [0]), (p2, [0])]))
    assert not t["dmg_obs"]


# --- the committed artifact + the servable constants -------------------------

FACTS = Path(rf.OUT)


@pytest.mark.skipif(not FACTS.exists(),
                    reason="data/rule_facts.json not yet mined on this box "
                           "(run scripts/build_rule_facts.py)")
def test_committed_seed_facts():
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    assert str(DUDU) in facts["self_removal_abilities"]      # Run Away Draw
    assert str(BOSS) in facts["gust_cards"]                  # Boss's Orders
    defenders = {p["defender_name"] for p in facts["zero_damage_pairs"]}
    assert any("Crustle" in d for d in defenders)            # the O-vs-wall 0


@pytest.mark.skipif(not FACTS.exists(),
                    reason="data/rule_facts.json not yet mined on this box")
def test_frozen_registries_match_the_mined_tables():
    """rl/plan's hand-frozen id sets are the SERVABLE copy of these facts
    (the bundle cannot read data/ at Kaggle serve time) — they must stay
    subsets of what the corpus actually shows."""
    import rl.plan as rp
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    mined_removal = {int(k) for k in facts["self_removal_abilities"]}
    assert set(rp.DUDUNSPARCE_IDS) <= mined_removal
    mined_gust = {int(k) for k in facts["gust_cards"]}
    assert set(rp.GUST_IDS) <= mined_gust
