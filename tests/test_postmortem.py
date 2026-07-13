"""Unit tests for rl/postmortem.py — the pure logic (classification, option
decoding, audit flags) on synthetic observations and a synthetic card table;
no engine, no network. Card tables are patched module-bound, the fake_cg way."""
from types import SimpleNamespace

import pytest

import rl.postmortem as pm

# synthetic pool: a basic, its evolution, and a hand-discard trainer
BASIC = SimpleNamespace(cardId=1, name="Riolu", basic=True, evolvesFrom=None)
EVO = SimpleNamespace(cardId=2, name="Mega Lucario ex", basic=False,
                      evolvesFrom="Riolu")
CARMINE = SimpleNamespace(cardId=3, name="Carmine", basic=False, evolvesFrom=None)
POOL = (BASIC, EVO, CARMINE)


@pytest.fixture(autouse=True)
def _patch_cards(monkeypatch):
    monkeypatch.setattr(pm, "_CARDS", {c.cardId: c for c in POOL})
    monkeypatch.setattr(pm, "_CARDS_BY_NAME", {c.name: c for c in POOL})


def _player(prizes=6, deck=30, active=None, bench=(), hand_ids=()):
    return {
        "prize": [{}] * prizes,
        "deckCount": deck,
        "active": [active] if active else [{}],
        "bench": [{"id": i} for i in bench],
        "hand": [{"id": i} for i in hand_ids],
        "handCount": len(hand_ids),
        "discard": [],
    }


def test_classify_win():
    cur = {"players": [_player(), _player()]}
    assert pm.classify_end(cur, 0, 1.0) == "WE WON"


def test_classify_benched_out_beats_prize_read():
    # Empty bench at the final decision = the KO that follows ends the game,
    # regardless of how many prizes the opponent still had.
    us = _player(active={"id": BASIC.cardId, "hp": 80}, bench=())
    opp = _player(prizes=2)
    assert "BENCHED-OUT" in pm.classify_end({"players": [opp, us]}, 1, -1.0)


def test_classify_deck_out():
    us = _player(deck=0, active={"id": BASIC.cardId, "hp": 80},
                 bench=(BASIC.cardId,))
    opp = _player(prizes=4)
    assert "DECK-OUT" in pm.classify_end({"players": [opp, us]}, 1, -1.0)


def test_classify_prizes():
    us = _player(active={"id": BASIC.cardId, "hp": 80}, bench=(BASIC.cardId,))
    opp = _player(prizes=1)
    assert "PRIZES" in pm.classify_end({"players": [opp, us]}, 1, -1.0)


def test_is_basic_pokemon():
    assert pm._is_basic_pokemon("PLAY:hand[2]=Riolu")
    assert not pm._is_basic_pokemon("PLAY:hand[0]=Mega Lucario ex")
    assert not pm._is_basic_pokemon("PLAY:hand[1]=Carmine")


def test_describe_option_resolves_hand_card():
    cur = {"players": [_player(hand_ids=(BASIC.cardId,)), _player()]}
    assert pm.describe_option({"type": 7, "index": 0}, cur, us=0) == \
        "PLAY:hand[0]=Riolu"


def test_describe_option_resolves_area_target():
    cur = {"players": [_player(), _player(bench=(BASIC.cardId,))]}
    out = pm.describe_option({"type": 3, "area": 5, "playerIndex": 1, "index": 0},
                             cur, us=1)
    assert out == "CARD:us.BENCH[0]=Riolu"


def _main_step(hand_ids, bench_ids, options, action, turn=2):
    """One decision in the real replay shape (verified 2026-07-12): the select
    lives at step i, the answering action is recorded at step i+1 — so a
    decision is TWO steps. Returns them as a 2-item list; concatenate multiple
    decisions with [*a, *b]."""
    cur = {"turn": turn, "players": [
        _player(),
        _player(hand_ids=hand_ids, bench=bench_ids),
    ]}
    select_step = [{}, {"observation": {"current": cur, "select":
                                        {"context": 0, "option": options}}}]
    answer_step = [{}, {"action": action}]
    return [select_step, answer_step]


def test_flag_carmine_discarding_evolutions():
    steps = _main_step(hand_ids=(CARMINE.cardId, EVO.cardId), bench_ids=(BASIC.cardId,),
                       options=[{"type": 7, "index": 0}, {"type": 14}],
                       action=[0])
    flags = pm.audit_flags(steps, us=1)
    assert len(flags) == 1 and "Mega Lucario ex" in flags[0]


def test_flag_turn_ending_with_empty_bench():
    steps = _main_step(hand_ids=(BASIC.cardId,), bench_ids=(),
                       options=[{"type": 7, "index": 0}, {"type": 14}],
                       action=[1])
    flags = pm.audit_flags(steps, us=1)
    assert len(flags) == 1 and "empty bench" in flags[0]


def test_no_flag_when_benched_later_same_turn():
    declined = _main_step(hand_ids=(BASIC.cardId,), bench_ids=(),
                          options=[{"type": 7, "index": 0}, {"type": 14}],
                          action=[1])
    benched = _main_step(hand_ids=(), bench_ids=(BASIC.cardId,),
                         options=[{"type": 14}], action=[0])
    assert pm.audit_flags([*declined, *benched], us=1) == []


# --- M8.0 setup-taxonomy checks (3)-(6) ---------------------------------------

def test_flag_evolution_left_in_hand_at_turn_end():
    # EVOLVE (type 9) offered at the turn's last MAIN prompt, END chosen.
    steps = _main_step(hand_ids=(EVO.cardId,), bench_ids=(BASIC.cardId,),
                       options=[{"type": 9, "index": 0}, {"type": 14}],
                       action=[1])
    flags = pm.audit_flags(steps, us=1)
    assert len(flags) == 1
    assert flags[0].startswith("[evolve-left]") and "Mega Lucario ex" in flags[0]


def test_no_evolve_flag_when_evolved_later_same_turn():
    declined = _main_step(hand_ids=(EVO.cardId,), bench_ids=(BASIC.cardId,),
                          options=[{"type": 9, "index": 0}, {"type": 14}],
                          action=[1])
    evolved = _main_step(hand_ids=(), bench_ids=(EVO.cardId,),
                         options=[{"type": 14}], action=[0])
    assert pm.audit_flags([*declined, *evolved], us=1) == []


def test_flag_fetch_dead_evolution():
    # KEEP context (TO_HAND): fetched the evolution while its basis is nowhere.
    from tests.fake_cg import SelectContext
    cur = {"turn": 3, "players": [
        _player(),
        _player(hand_ids=()),
    ]}
    steps = [[{}, {"observation": {"current": cur, "select": {
        "context": int(SelectContext.TO_HAND),
        "deck": [{"id": EVO.cardId}, {"id": BASIC.cardId}],
        "option": [{"type": 6, "area": 1, "index": 0},
                   {"type": 6, "area": 1, "index": 1}]}}}],
             [{}, {"action": [0]}]]
    flags = pm.audit_flags(steps, us=1)
    assert len(flags) == 1
    assert flags[0].startswith("[fetch-dead-evolution]")
    assert "Mega Lucario ex" in flags[0] and "Riolu" in flags[0]


def test_no_fetch_flag_when_basis_in_hand():
    from tests.fake_cg import SelectContext
    cur = {"turn": 3, "players": [
        _player(),
        _player(hand_ids=(BASIC.cardId,)),
    ]}
    steps = [[{}, {"observation": {"current": cur, "select": {
        "context": int(SelectContext.TO_HAND),
        "deck": [{"id": EVO.cardId}],
        "option": [{"type": 6, "area": 1, "index": 0}]}}}],
             [{}, {"action": [0]}]]
    assert pm.audit_flags(steps, us=1) == []


def test_flag_trainer_hoarded_across_turns():
    # Carmine PLAY-able on 4 distinct turns, never chosen (END every time).
    steps = [s for t in (2, 3, 4, 5)
             for s in _main_step(hand_ids=(CARMINE.cardId,), bench_ids=(BASIC.cardId,),
                                 options=[{"type": 7, "index": 0}, {"type": 14}],
                                 action=[1], turn=t)]
    flags = pm.audit_flags(steps, us=1)
    assert len(flags) == 1
    assert flags[0].startswith("[trainer-hoarded]") and "Carmine" in flags[0]


def test_no_hoard_flag_below_turn_threshold_or_when_played():
    below = [s for t in (2, 3, 4)
             for s in _main_step(hand_ids=(CARMINE.cardId,), bench_ids=(BASIC.cardId,),
                                 options=[{"type": 7, "index": 0}, {"type": 14}],
                                 action=[1], turn=t)]
    assert pm.audit_flags(below, us=1) == []
    played_last = below + _main_step(
        hand_ids=(CARMINE.cardId,), bench_ids=(BASIC.cardId,),
        options=[{"type": 7, "index": 0}, {"type": 14}], action=[0], turn=5)
    flags = pm.audit_flags(played_last, us=1)
    assert all(not f.startswith("[trainer-hoarded]") for f in flags)
