"""OppMemory (rl/memory.py) — log ingestion, prefix-dedupe, feature output."""
from types import SimpleNamespace

import numpy as np
import pytest

from rl.memory import N_MEM, N_MEM_IDS, OppMemory
from tests.builders import observation, player, pokemon

MEGA_LUCARIO = 678
SOLROCK = 980
BOSS = 1182
FIGHTING_ENERGY = 6


def poke(card_id, serial, **kw):
    p = pokemon(card_id, **kw)
    p.serial = serial
    return p


def obs_with_logs(logs, me=None, opponent=None, your_index=0):
    obs = observation(me=me, opponent=opponent, your_index=your_index)
    obs.logs = logs
    return obs


def ev(etype, player_index, **kw):
    return {"type": etype, "playerIndex": player_index, **kw}


def test_dims():
    m = OppMemory()
    state = observation().current
    assert m.features(state).shape == (N_MEM,)
    assert m.ids().shape == (N_MEM_IDS,)


def test_prefix_redelivery_not_double_counted():
    m = OppMemory()
    first = [ev(15, 1, cardId=SOLROCK, serial=70, attackId=675)]
    m.observe(obs_with_logs(first))
    # sub-prompt chain: previous window re-delivered verbatim + one new event
    m.observe(obs_with_logs(first + [ev(15, 1, cardId=SOLROCK, serial=70, attackId=675)]))
    assert m._attacks_total == 2  # prefix consumed once, the appended copy counts
    m.observe(obs_with_logs(first))  # identical re-delivery -> nothing new...
    # (a fresh delta identical to an OLD window is indistinguishable; the probe
    # showed 'identical' happens only within sub-prompt chains, where prev==new)


def test_identical_redelivery_ignored():
    m = OppMemory()
    logs = [ev(15, 1, cardId=SOLROCK, serial=70, attackId=675)]
    m.observe(obs_with_logs(logs))
    m.observe(obs_with_logs(list(logs)))  # same window again
    assert m._attacks_total == 1


def test_disjoint_windows_both_counted():
    m = OppMemory()
    m.observe(obs_with_logs([ev(15, 1, cardId=SOLROCK, serial=70, attackId=675)]))
    m.observe(obs_with_logs([ev(15, 1, cardId=MEGA_LUCARIO, serial=71, attackId=1)]))
    assert m._attacks_total == 2
    assert m._last_attacker_id == MEGA_LUCARIO


def test_turn_flags():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(2, 1), ev(15, 1, cardId=SOLROCK, serial=70, attackId=675), ev(3, 1)]))
    assert m._attacked_last_turn and not m._passed_last_turn
    m.observe(obs_with_logs([ev(2, 1), ev(3, 1)]))
    assert m._passed_last_turn and not m._attacked_last_turn


def test_own_events_ignored():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(15, 0, cardId=MEGA_LUCARIO, serial=10, attackId=1),
        ev(10, 0, cardId=BOSS, serial=11),
        ev(5, 0)]))
    assert m._attacks_total == 0 and not m._played and m._draw_reverse == 0


def test_known_hand_add_and_remove():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(6, 1, cardId=BOSS, serial=90, fromArea=1, toArea=2)]))  # revealed to hand
    state = observation().current
    f = m.features(state)
    assert f[13] == pytest.approx(0.1)          # known-hand count 1/10
    assert f[15:].sum() > 0                      # FEAT pool nonzero
    m.observe(obs_with_logs([ev(10, 1, cardId=BOSS, serial=90)]))  # played it
    f = m.features(state)
    assert f[13] == 0.0 and f[15:].sum() == 0.0


def test_attach_target_one_hot_resolves_bench():
    m = OppMemory()
    target = poke(SOLROCK, serial=55, energies=[3])
    opp = player(active=poke(MEGA_LUCARIO, serial=50), bench=[target])
    state = observation(opponent=opp).current
    m.observe(obs_with_logs([
        ev(11, 1, cardId=FIGHTING_ENERGY, serial=91, serialTarget=55)]))
    f = m.features(state)
    hot = f[3:10]
    assert hot[2] == 1.0 and hot.sum() == 1.0    # bench slot 0
    # attacker promoted to active -> one-hot follows the serial
    opp2 = player(active=target, bench=[])
    state2 = observation(opponent=opp2).current
    hot2 = m.features(state2)[3:10]
    assert hot2[1] == 1.0 and hot2.sum() == 1.0  # active


def test_attach_target_unresolved_is_none_slot():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(11, 1, cardId=FIGHTING_ENERGY, serial=91, serialTarget=999)]))
    hot = m.features(observation().current)[3:10]
    assert hot[0] == 1.0 and hot.sum() == 1.0


def test_last_attack_features_vs_my_active():
    m = OppMemory()
    # fake_cg card 4 (Psychic): attack 104, 30 printed dmg, 1 colorless cost;
    # my active card 1 is WEAK to Psychic -> effective doubles to 60.
    m.observe(obs_with_logs([ev(15, 1, cardId=4, serial=70, attackId=104)]))
    me = player(active=poke(1, serial=10, hp=100))
    opp = player(active=poke(4, serial=70))
    state = observation(me=me, opponent=opp).current
    f = m.features(state)
    assert f[0] == pytest.approx(30 / 300.0)
    assert f[1] == pytest.approx(1 / 5.0)
    assert f[2] == pytest.approx(60 / 300.0)     # weakness-doubled vs my active


def test_ids_order_and_padding():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(10, 1, cardId=BOSS, serial=1),
        ev(11, 1, cardId=FIGHTING_ENERGY, serial=2, serialTarget=55),
        ev(15, 1, cardId=SOLROCK, serial=70, attackId=675)]))
    ids = m.ids()
    assert list(ids) == [0, 0, BOSS, FIGHTING_ENERGY, SOLROCK]


def test_draw_reverse_intensity():
    m = OppMemory()
    m.observe(obs_with_logs([ev(2, 1)] + [ev(5, 1)] * 4 + [ev(3, 1)]))
    f = m.features(observation().current)
    assert f[14] == pytest.approx(0.3)           # (4 draws - 1 turn)/10


def test_reset_clears_everything():
    m = OppMemory()
    m.observe(obs_with_logs([
        ev(15, 1, cardId=SOLROCK, serial=70, attackId=675),
        ev(6, 1, cardId=BOSS, serial=90, fromArea=1, toArea=2)]))
    m.reset()
    assert m._attacks_total == 0 and not m._known_hand
    assert not m._prev_fps
    assert list(m.ids()) == [0, 0, 0, 0, 0]


def test_seat1_perspective():
    m = OppMemory()
    # I'm seat 1; opponent is player 0
    m.observe(obs_with_logs([ev(15, 0, cardId=MEGA_LUCARIO, serial=10, attackId=1)],
                            your_index=1))
    assert m._attacks_total == 1


def test_dataclass_like_logs_supported():
    m = OppMemory()
    log = SimpleNamespace(type=15, playerIndex=1, cardId=SOLROCK, serial=70,
                          attackId=675, serialTarget=None, value=None, head=None)
    m.observe(obs_with_logs([log]))
    assert m._attacks_total == 1
