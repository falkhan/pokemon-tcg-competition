"""Encoder v4 (M21): layout, v3-prefix invariance, and the appended block."""
from types import SimpleNamespace

import numpy as np
import pytest

import rl.encoders as enc
from rl.memory import OppMemory
from tests.builders import hand_card, observation, player, pokemon


def poke(card_id, serial=0, **kw):
    p = pokemon(card_id, **kw)
    p.serial = serial
    return p


def obs_with_logs(logs=(), **kw):
    obs = observation(**kw)
    obs.logs = list(logs)
    return obs


DECK = [1] * 30 + [6] * 30


def encode_both(obs, memory=None):
    m = memory or OppMemory()
    m.observe(obs)
    return enc.encode_ctx_v4(obs, DECK, m)


def test_dims():
    ctx, ids = encode_both(obs_with_logs())
    assert ctx.shape == (enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.V4_EXTRA_DIM,)
    assert ids.shape == (enc.N_STATE_IDS_V4,)
    assert ctx.dtype == np.float32 and ids.dtype == np.int32


def test_v3_prefix_byte_identical():
    me = player(active=poke(1, 10, energies=[6]), hand=[hand_card(7)])
    opp = player(active=poke(3, 20))
    obs = obs_with_logs(me=me, opponent=opp)
    ctx, ids = encode_both(obs)
    numeric, ids3 = enc.encode_state_v3(obs.current, DECK)
    v3_ctx = np.concatenate([numeric, enc.encode_context(obs.select.context)])
    assert np.array_equal(ctx[:v3_ctx.shape[0]], v3_ctx)
    assert np.array_equal(ids[:ids3.shape[0]], ids3)


def test_memory_ids_appended_at_tail():
    m = OppMemory()
    obs = obs_with_logs([{"type": 10, "playerIndex": 1, "cardId": 7, "serial": 5}])
    _, ids = encode_both(obs, m)
    assert list(ids[-enc.N_MEM_IDS:]) == [0, 0, 0, 7, 0]


def test_slot_extras_flags():
    active = poke(1, 10, energies=[6])          # card 1 attack 101 affordable?
    bench_mon = poke(3, 11)
    bench_mon.appearThisTurn = True
    bench_mon.preEvolution = [SimpleNamespace(id=1)]
    me = player(active=active, bench=[bench_mon])
    obs = obs_with_logs(me=me, opponent=player(active=poke(2, 20)))
    ctx, _ = encode_both(obs)
    block = ctx[enc.STATE_V2_DIM + enc.N_CONTEXTS:]
    slots = block[enc.N_MEM:enc.N_MEM + 12 * enc.N_SLOT_EXTRA]
    bench0 = slots[enc.N_SLOT_EXTRA:2 * enc.N_SLOT_EXTRA]  # my bench slot 0
    assert bench0[0] == 1.0                      # appearThisTurn
    assert bench0[1] == pytest.approx(0.5)       # 1 pre-evolution / 2
    empty = slots[2 * enc.N_SLOT_EXTRA:3 * enc.N_SLOT_EXTRA]
    assert not empty.any()                       # empty bench slot -> zeros


def test_slot_extras_readiness():
    # fake card 1: attack 101 (fighting cost) — one fighting energy = ready
    ready = poke(1, 10, energies=[2])            # fake FIGHTING = 2
    from tests.fake_cg import EnergyType
    ready.energies = [int(EnergyType.FIGHTING)]
    me = player(active=ready)
    obs = obs_with_logs(me=me, opponent=player(active=poke(2, 20)))
    ctx, _ = encode_both(obs)
    slots = ctx[enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.N_MEM:]
    my_active = slots[:enc.N_SLOT_EXTRA]
    assert my_active[3] == 1.0                   # ready now
    assert my_active[4] > 0.0                    # best affordable dmg


def test_special_energy_pool_and_identity():
    from tests.fake_cg import EnergyType
    active = poke(1, 10)
    active.energyCards = [SimpleNamespace(id=8)]  # fake card 8 = SPECIAL_ENERGY
    active.energies = [int(EnergyType.FIGHTING)]
    me = player(active=active)
    obs = obs_with_logs(me=me, opponent=player(active=poke(2, 20)))
    ctx, _ = encode_both(obs)
    base = enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.N_MEM + 12 * enc.N_SLOT_EXTRA
    my_pool = ctx[base:base + enc.FEAT_DIM]
    opp_pool = ctx[base + enc.FEAT_DIM:base + 2 * enc.FEAT_DIM]
    assert np.array_equal(my_pool, enc.FEAT[8])
    assert not opp_pool.any()
    # per-slot special-energy count sees it too
    slots = ctx[enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.N_MEM:]
    assert slots[2] == pytest.approx(1 / 3.0)


def test_prize_pools_face_up():
    me = player()
    me.prize = [SimpleNamespace(id=7), None, None, None, None, None]
    obs = obs_with_logs(me=me)
    ctx, _ = encode_both(obs)
    base = (enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.N_MEM
            + 12 * enc.N_SLOT_EXTRA + 2 * enc.FEAT_DIM)
    my_prize = ctx[base:base + enc.FEAT_DIM + 1]
    assert np.array_equal(my_prize[:-1], enc.FEAT[7])
    assert my_prize[-1] == pytest.approx(1 / 6.0)


def test_select_extras():
    obs = obs_with_logs()
    obs.select.minCount = 1
    obs.select.maxCount = 2
    obs.select.remainDamageCounter = 3
    obs.select.remainEnergyCost = 1
    obs.select.effect = SimpleNamespace(id=7)
    obs.select.contextCard = SimpleNamespace(id=8)
    ctx, _ = encode_both(obs)
    sel = ctx[-enc.N_SELECT_EXTRA:]
    assert sel[0] == pytest.approx(0.1)
    assert sel[1] == pytest.approx(0.2)
    assert sel[2] == pytest.approx(0.3)
    assert sel[3] == pytest.approx(0.2)
    assert np.array_equal(sel[4:4 + enc.FEAT_DIM], enc.FEAT[7])
    assert np.array_equal(sel[4 + enc.FEAT_DIM:], enc.FEAT[8])


def test_global_extras():
    me = player(deck_count=30)
    opp = player(deck_count=42)
    obs = obs_with_logs(me=me, opponent=opp)
    obs.current.stadiumPlayed = True
    obs.current.retreated = False
    obs.current.turnActionCount = 4
    obs.current.firstPlayer = 0
    ctx, _ = encode_both(obs)
    base = (enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.N_MEM
            + 12 * enc.N_SLOT_EXTRA + 2 * enc.FEAT_DIM + 2 * (enc.FEAT_DIM + 1))
    g = ctx[base:base + enc.N_GLOBAL_EXTRA]
    assert g[0] == pytest.approx(42 / 60.0)      # opp deckCount
    assert g[3] == 1.0 and g[4] == 0.0           # stadiumPlayed / retreated
    assert g[5] == pytest.approx(0.4)            # turnActionCount
    assert g[6] == 1.0                           # I moved first


def test_deterministic_golden():
    """Pinned end-to-end checksum — catches accidental layout/order changes."""
    m = OppMemory()
    logs = [
        {"type": 2, "playerIndex": 1},
        {"type": 11, "playerIndex": 1, "cardId": 6, "serial": 91, "serialTarget": 55},
        {"type": 15, "playerIndex": 1, "cardId": 4, "serial": 70, "attackId": 104},
        {"type": 3, "playerIndex": 1},
    ]
    target = poke(3, 55, energies=[2])
    me = player(active=poke(1, 10, energies=[2], hp=80), hand=[hand_card(7)])
    opp = player(active=poke(4, 70), bench=[target])
    obs = obs_with_logs(logs, me=me, opponent=opp)
    ctx, ids = encode_both(obs, m)
    assert ctx.shape[0] == enc.STATE_V2_DIM + enc.N_CONTEXTS + enc.V4_EXTRA_DIM
    block = ctx[enc.STATE_V2_DIM + enc.N_CONTEXTS:]
    # memory head: attack 104 printed 30 dmg, cost 1, weakness-doubled to 60
    assert block[0] == pytest.approx(0.1)
    assert block[1] == pytest.approx(0.2)
    assert block[2] == pytest.approx(0.2)
    hot = block[3:3 + enc.N_MEM_ATTACH_HOT]
    assert hot[2] == 1.0                         # attach target = opp bench 0
    assert list(ids[-enc.N_MEM_IDS:]) == [0, 0, 0, 6, 4]
