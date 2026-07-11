"""M7.4a within-turn combo solver — the offline combo-position suite.

The real curated suite (from Kaggle forensics) is [NET]-deferred, so these are
hand-authored scripted-tree positions, the tests/test_search.py idiom: a
dict-keyed ``{(search_id, action_tuple): SearchState}`` game tree monkeypatched
onto rl.turn_solver's module-bound search_* names. The headline fixture is the
milestone's target behavior: a multi-prize lethal needing item -> submenu ->
attack that the greedy pilot provably misses. Win-rate / latency gates are
[ENGINE] (docs/M7-manual-tests.md).
"""
from time import perf_counter

import pytest

import rl.turn_solver as ts
from rl.generic_pilot import make_generic_pilot
from tests.builders import (hand_card, observation, option, player, pokemon,
                            search_state)
from tests.fake_cg import AreaType, EnergyType, OptionType, SelectContext

F = EnergyType.FIGHTING
W = EnergyType.WATER
DECK = list(range(60))


def flipped(my_p, op_p, **kw):
    """A turn-passed observation: players[0] stays ME (seat 0), opponent to
    move. builders.observation puts its `me` arg at players[your_index]."""
    return observation(op_p, my_p, your_index=1, **kw)


def me_board(energies=(F, F), hand=(hand_card(7),), prizes=4):
    return player(active=pokemon(1, hp=120, energies=energies),
                  hand=list(hand), prizes_remaining=prizes)


def op_board(hp=130, prizes=4):
    return player(active=pokemon(2, hp=hp), prizes_remaining=prizes)


def main_menu(my_p, op_p, options):
    obs = observation(my_p, op_p, context=SelectContext.MAIN, options=options)
    obs.search_begin_input = object()
    return obs


EVOLVE = option(OptionType.EVOLVE)
PLAY_HAND0 = option(OptionType.PLAY, area=AreaType.HAND, index=0)
ATTACK_102 = option(OptionType.ATTACK, attack_id=102)
END = option(OptionType.END)


def combo_tree():
    """The headline position. Card 1 (Fighting, 120dmg best attack) vs card 2
    (Water ex, 130hp, resists Fighting -> 90 best): no KO exists greedily.
    The scripted trainer (card 7) is a damage boost: PLAY -> target submenu ->
    ATTACK takes BOTH prizes. Greedy prefers the EVOLVE develop move.

    Returns (root_obs, step_fn)."""
    my_p, op_p = me_board(), op_board()
    root = main_menu(my_p, op_p, [EVOLVE, PLAY_HAND0, ATTACK_102, END])

    chip = flipped(my_p, player(active=pokemon(2, hp=40), prizes_remaining=4))
    ended = flipped(my_p, op_p)
    ko2 = flipped(my_p, player(active=pokemon(5, hp=60), prizes_remaining=2))

    def main2(sid):  # a my-turn ATTACK/END menu deeper in the line
        return search_state(
            observation(my_p, op_p, context=SelectContext.MAIN,
                        options=[ATTACK_102, END]), search_id=sid)

    submenu = search_state(observation(
        my_p, op_p, context=SelectContext.EFFECT_TARGET,
        options=[option(OptionType.CARD, area=AreaType.ACTIVE, index=0, player_index=1),
                 option(OptionType.CARD, area=AreaType.ACTIVE, index=0, player_index=0)]),
        search_id=20)

    tree = {
        (0, (2,)): search_state(chip, 10),      # greedy attack: 90 < 130, no prize
        (0, (0,)): main2(11),                   # EVOLVE develops, then...
        (11, (0,)): search_state(chip, 12),
        (11, (1,)): search_state(ended, 13),
        (0, (1,)): submenu,                     # PLAY the boost trainer
        (20, (0,)): main2(21),                  # right target -> boosted MAIN
        (21, (0,)): search_state(ko2, 22),      # ATTACK: KO the ex, 2 prizes
        (21, (1,)): search_state(ended, 23),
        (20, (1,)): main2(24),                  # wrong target: no boost
        (24, (0,)): search_state(chip, 25),
        (24, (1,)): search_state(ended, 26),
        (0, (3,)): search_state(ended, 27),
    }
    return root, lambda sid, action: tree[(sid, tuple(action))]


def patch_engine(monkeypatch, root_obs, step, end_calls=None):
    monkeypatch.setattr(ts, "search_begin",
                        lambda obs, **kw: search_state(root_obs, 0))
    monkeypatch.setattr(ts, "search_step", step)
    monkeypatch.setattr(ts, "search_end",
                        lambda: None if end_calls is None else end_calls.append(1))


def test_headline_finds_the_multi_prize_lethal_greedy_misses(monkeypatch):
    root, step = combo_tree()
    end_calls: list = []
    patch_engine(monkeypatch, root, step, end_calls)

    assert make_generic_pilot(DECK)(root)[0] == 0        # greedy: EVOLVE (2800)
    assert ts.should_solve(root)                         # T2: 90 >= 130-70 + trainer
    assert ts.solve_turn(root, DECK) == [1]              # solver: PLAY the boost
    assert end_calls                                     # search closed

    # The winning line is worth both prizes, found through the submenu.
    snap = ts._root_snapshot(root)
    score, line = ts._dfs(search_state(root, 0), snap, 0,
                          perf_counter() + 1.0, {"nodes": 0})
    assert score >= 2 * ts.W_PRIZE
    assert line == [[1], [0], [0]]                       # PLAY -> target -> ATTACK

    pilot = ts.make_solver_pilot(DECK)
    assert pilot(root) == [1]


def test_win_short_circuit_stops_expanding_siblings(monkeypatch):
    my_p, op_p = me_board(), op_board()
    root = main_menu(my_p, op_p, [ATTACK_102, EVOLVE, END])
    won = flipped(my_p, op_p)
    won.current.result = 0                               # seat 0 (me) wins
    calls = []

    def step(sid, action):
        calls.append((sid, tuple(action)))
        assert (sid, tuple(action)) == (0, (0,))         # siblings never expanded
        return search_state(won, 1)

    patch_engine(monkeypatch, root, step)
    assert ts.solve_turn(root, DECK) == [0]
    assert len(calls) == 1


def test_no_lethal_declines_and_pilot_matches_greedy(monkeypatch):
    my_p, op_p = me_board(), op_board()                  # 90 max dmg vs 130 hp
    root = main_menu(my_p, op_p, [ATTACK_102, END])
    chip = flipped(my_p, player(active=pokemon(2, hp=40), prizes_remaining=4))
    ended = flipped(my_p, op_p)
    tree = {(0, (0,)): search_state(chip, 1), (0, (1,)): search_state(ended, 2)}
    patch_engine(monkeypatch, root, lambda s, a: tree[(s, tuple(a))])

    assert ts.should_solve(root)                         # T2 fires...
    assert ts.solve_turn(root, DECK) is None             # ...but no prize line
    assert ts.make_solver_pilot(DECK)(root) == make_generic_pilot(DECK)(root)


def test_multi_select_prompt_capped_and_lethal_pair_found(monkeypatch):
    my_p, op_p = me_board(), op_board()
    root = main_menu(my_p, op_p, [PLAY_HAND0, END])
    ended = flipped(my_p, op_p)
    ko2 = flipped(my_p, player(active=pokemon(5, hp=60), prizes_remaining=2))
    picks = observation(my_p, op_p, context=SelectContext.DISCARD, max_count=2,
                        options=[option(OptionType.CARD, area=AreaType.HAND, index=0)] * 4)

    cands = ts._candidate_actions(picks)
    assert len(cands) <= ts.MAX_MULTI_COMBOS
    assert all(len(c) == 2 for c in cands)

    kill = observation(my_p, op_p, context=SelectContext.MAIN,
                       options=[ATTACK_102, END])
    tree = {(0, (0,)): search_state(picks, 1), (0, (1,)): search_state(ended, 2),
            (1, (1, 3)): search_state(kill, 3),
            (3, (0,)): search_state(ko2, 4), (3, (1,)): search_state(ended, 5)}
    for pair in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]:  # every other pair: dead
        tree[(1, pair)] = search_state(ended, 10 + pair[0] * 4 + pair[1])
    patch_engine(monkeypatch, root, lambda s, a: tree[(s, tuple(a))])

    snap = ts._root_snapshot(root)
    score, line = ts._dfs(search_state(root, 0), snap, 0,
                          perf_counter() + 1.0, {"nodes": 0})
    assert score >= 2 * ts.W_PRIZE
    assert line == [[0], [1, 3], [0]]                    # the lethal discard pair


def test_deadline_aborts_before_any_step(monkeypatch):
    root, _ = combo_tree()
    end_calls: list = []

    def step(sid, action):
        raise AssertionError("deadline 0 must not expand anything")

    patch_engine(monkeypatch, root, step, end_calls)
    assert ts.solve_turn(root, DECK, deadline_s=0.0) is None
    assert end_calls


def test_node_cap_bounds_the_search(monkeypatch):
    my_p, op_p = me_board(), op_board()
    root = main_menu(my_p, op_p, [EVOLVE, PLAY_HAND0, END])
    calls = []
    next_id = [1]

    def step(sid, action):                               # endless my-turn tree
        calls.append(1)
        obs = observation(my_p, op_p, context=SelectContext.MAIN,
                          options=[EVOLVE, PLAY_HAND0, END])
        sid2, next_id[0] = next_id[0], next_id[0] + 1
        return search_state(obs, sid2)

    patch_engine(monkeypatch, root, step)
    monkeypatch.setattr(ts, "MAX_NODES", 5)
    assert ts.solve_turn(root, DECK) is None
    assert len(calls) == 5


def test_engine_error_still_closes_search_and_pilot_falls_back(monkeypatch):
    root, _ = combo_tree()
    end_calls: list = []

    def step(sid, action):
        raise RuntimeError("engine hiccup")

    patch_engine(monkeypatch, root, step, end_calls)
    with pytest.raises(RuntimeError):
        ts.solve_turn(root, DECK)
    assert end_calls

    assert ts.make_solver_pilot(DECK)(root) == make_generic_pilot(DECK)(root)


def _trigger_obs(my_p, op_p, n_options=2, with_input=True):
    obs = observation(my_p, op_p, context=SelectContext.MAIN,
                      options=[ATTACK_102, END][:n_options])
    if with_input:
        obs.search_begin_input = object()
    return obs


@pytest.mark.parametrize("case,my_p,op_p,expected", [
    # T1: KO one attach away (120dmg-30resist=90 vs 90hp, second F affordable)
    ("T1", me_board(energies=(F,), hand=()), op_board(hp=90), True),
    # T2: 90 affordable now, gap 40 <= LETHAL_MARGIN, boost trainer in hand
    ("T2", me_board(), op_board(hp=130), True),
    # T3: card 3 charged-best 240 one-shots the ex, 1 attach short, but its
    # attached WATER can't pay [F,F] so best_damage(+1) stays 0 (no T1)
    ("T3", player(active=pokemon(3, hp=200, energies=(W,)), prizes_remaining=4),
     op_board(hp=130), True),
    # T4: 2 prizes from winning, any damage in reach
    ("T4", me_board(energies=(), hand=()), op_board(hp=200, prizes=2), True),
    # negatives
    ("far", me_board(energies=(), hand=()), op_board(hp=200), False),
    ("no-trainer", me_board(hand=()), op_board(hp=130), False),
])
def test_trigger_table(case, my_p, op_p, expected):
    assert ts.should_solve(_trigger_obs(my_p, op_p)) is expected


def test_trigger_needs_forward_model_and_a_real_choice():
    my_p, op_p = me_board(), op_board()
    assert not ts.should_solve(_trigger_obs(my_p, op_p, with_input=False))
    assert not ts.should_solve(_trigger_obs(my_p, op_p, n_options=1))


def test_score_leaf_ordering():
    my_p = me_board()
    snap = ts._Snap(me=0, my_prizes=4, op_prizes=4, op_active_hp=130,
                    my_deck_count=30)

    def leaf(op_hp=130, op_prizes=4, my_prizes=4, my_deck=30, result=-1):
        mine = player(active=pokemon(1, hp=120, energies=(F, F)),
                      prizes_remaining=my_prizes, deck_count=my_deck)
        opp = player(active=pokemon(2, hp=op_hp), prizes_remaining=op_prizes)
        obs = flipped(mine, opp)
        obs.current.result = result
        return ts.score_leaf(snap, obs)

    win = leaf(result=0)
    two_prizes = leaf(op_prizes=2, op_hp=60)
    one_prize_threat = leaf(op_prizes=3, op_hp=60)       # 90 dmg >= 60: lethal next
    one_prize = leaf(op_prizes=3)
    threat_only = leaf(op_hp=60)
    dev = leaf()
    assert win > two_prizes > one_prize_threat > one_prize > threat_only > dev
    assert leaf(op_prizes=3, my_prizes=3) < one_prize    # conceding a prize costs
    assert leaf(my_deck=4) < dev                         # deck-out draws penalized
    assert leaf(result=1) < leaf(result=2) < dev         # loss < draw < any live line


def test_open_search_fills_opponent_zones_and_samples_mine(monkeypatch):
    captured = {}

    def fake_begin(obs, **kwargs):
        captured.update(kwargs)
        return search_state(obs, 0)

    monkeypatch.setattr(ts, "search_begin", fake_begin)
    my_p = me_board()
    op_p = op_board()
    obs = observation(my_p, op_p)
    ts._open_search(obs, DECK)

    assert len(captured["your_deck"]) == my_p.deckCount
    assert set(captured["your_deck"]) <= set(DECK)
    assert len(captured["your_prize"]) == len(my_p.prize)
    assert captured["opponent_deck"] == [ts.FILLER_POKEMON] * op_p.deckCount
    assert captured["opponent_prize"] == [ts.FILLER_ENERGY] * len(op_p.prize)
    assert captured["opponent_hand"] == [ts.FILLER_ENERGY] * op_p.handCount
    assert captured["opponent_active"] == []             # active is known

    op_p.active = [None]                                 # hidden active
    ts._open_search(observation(my_p, op_p), DECK)
    assert captured["opponent_active"] == [ts.FILLER_POKEMON]


def test_candidate_actions_keeps_attacks_and_end():
    my_p, op_p = me_board(), op_board()
    weak = [option(OptionType.CARD, area=AreaType.HAND, index=0)] * 6
    obs = observation(my_p, op_p, context=SelectContext.MAIN,
                      options=weak + [ATTACK_102, END])
    cands = ts._candidate_actions(obs)
    assert [6] in cands                                  # the attack survives
    assert [7] in cands                                  # END always kept
    assert len(cands) <= 1 + ts.TOP_K + 1


def test_source_stays_bundle_pure():
    from pathlib import Path
    src = Path(ts.__file__).read_text()
    imports = [line for line in src.splitlines()
               if line.startswith(("import ", "from "))]
    for heavy in ("numpy", "polars", "torch", "rl.mcts", "rl.encoders"):
        assert not any(heavy in line for line in imports), \
            f"turn_solver must not import {heavy} (ships in the pure bundle)"
