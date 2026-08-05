"""M43: --shaping value — Phi = the frozen start's V(s) (BACKLOG #13(c) step 2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.collector import _load_phi_net, _value_potential, collect
from rl.encoders import N_CONTEXTS, N_STATE_IDS_V3, OPTION_V3_DIM, STATE_V2_DIM
from rl.plan import PLAN_DIM
from rl.policy import OptionScorerV3


def _net(option_dim=OPTION_V3_DIM):
    torch.manual_seed(0)
    return OptionScorerV3(n_state_ids=N_STATE_IDS_V3, option_dim=option_dim)


def _decision(rng, n_opts=3):
    sc = rng.random(STATE_V2_DIM + N_CONTEXTS).astype(np.float32)
    sids = rng.integers(0, 1268, N_STATE_IDS_V3).astype(np.int32)
    opts = rng.random((n_opts, OPTION_V3_DIM)).astype(np.float32)
    oids = rng.integers(0, 1268, (n_opts, 2)).astype(np.int32)
    plan = rng.random(PLAN_DIM).astype(np.float32)
    return sc, sids, opts, oids, plan


def test_value_potential_is_the_nets_value_head():
    net = _net()
    sc, sids, opts, oids, plan = _decision(np.random.default_rng(1))
    phi = _value_potential(net, sc, sids, opts, oids, plan)
    with torch.no_grad():
        _, expected = net(torch.from_numpy(sc).unsqueeze(0),
                          torch.from_numpy(plan).unsqueeze(0),
                          torch.from_numpy(sids.astype(np.int64)).unsqueeze(0),
                          torch.from_numpy(opts).unsqueeze(0),
                          torch.from_numpy(oids.astype(np.int64)).unsqueeze(0))
    assert phi == pytest.approx(float(expected))
    assert -1.0 <= phi <= 1.0        # value head is Tanh-bounded


def test_load_phi_net_freezes_the_frozen_start(tmp_path):
    net = _net()
    ckpt = tmp_path / "start.pt"
    torch.save(net.state_dict(), ckpt)
    phi_net = _load_phi_net(ckpt, net.state_dict())
    assert not phi_net.training
    assert all(not p.requires_grad for p in phi_net.parameters())


def test_load_phi_net_rejects_arch_mismatch(tmp_path):
    ckpt = tmp_path / "other.pt"
    torch.save(_net(option_dim=OPTION_V3_DIM - 4).state_dict(), ckpt)
    with pytest.raises(ValueError, match="architecture"):
        _load_phi_net(ckpt, _net().state_dict())


def test_collect_value_shaping_requires_explicit_phi_ckpt(tmp_path):
    with pytest.raises(ValueError, match="FROZEN START"):
        collect(1, "checkpoints/whatever.pt", 1, out_dir=tmp_path,
                race_shaping=0.1, shaping="value")
