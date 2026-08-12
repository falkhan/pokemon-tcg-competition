"""M43: --shaping value — Phi = the frozen start's V(s) (BACKLOG #13(c) step 2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.collector import SHAPING_GAMMA, _load_phi_net, _shaping_step, _value_potential, collect
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


# --- the invariant shaping form (M43 review finding #5) ------------------

def test_shaping_gamma_pins_ppo_gamma():
    """SHAPING_GAMMA is a copy (ppo.py imports the collector, so the collector
    cannot import it back); this test is the pin that keeps them equal."""
    from rl.ppo import GAMMA
    assert SHAPING_GAMMA == GAMMA


def test_shaping_step_form():
    # each increment is exactly coef * (gamma*phi' - phi)
    assert _shaping_step(0.4, 0.7, 0.05) == pytest.approx(
        0.05 * (SHAPING_GAMMA * 0.7 - 0.4))
    assert _shaping_step(0.4, 0.7, 0.05, gamma=0.5) == pytest.approx(
        0.05 * (0.5 * 0.7 - 0.4))


def test_shaping_terminal_term_zeroes_the_potential():
    # F_T with phi(terminal)=0 is -coef*phi_last regardless of gamma
    assert _shaping_step(0.83, 0.0, 0.05) == pytest.approx(-0.05 * 0.83)
    assert _shaping_step(-0.6, 0.0, 0.05, gamma=0.1) == pytest.approx(0.05 * 0.6)


def test_shaping_telescopes_at_gamma_one():
    # undiscounted sum over any phi sequence (incl. the terminal step)
    # collapses to -coef*phi_0 — no residual for ending in a high-phi state
    rng = np.random.default_rng(7)
    phis = rng.uniform(-1.0, 1.0, size=40)
    coef = 0.05
    total = sum(_shaping_step(phis[i - 1], phis[i], coef, gamma=1.0)
                for i in range(1, len(phis)))
    total += _shaping_step(phis[-1], 0.0, coef, gamma=1.0)
    assert total == pytest.approx(-coef * phis[0])


def test_shaping_telescopes_discounted():
    # with per-step discounting, sum_t gamma^t * F_t = -coef*phi_0 exactly
    # (each phi_t appears once as +gamma^t*coef*gamma*phi_t and once as
    # -gamma^(t+1)*coef*phi_t)
    rng = np.random.default_rng(11)
    phis = rng.uniform(-1.0, 1.0, size=25)
    coef, g = 0.05, SHAPING_GAMMA
    steps = [_shaping_step(phis[i - 1], phis[i], coef) for i in range(1, len(phis))]
    steps.append(_shaping_step(phis[-1], 0.0, coef))
    total = sum((g ** t) * f for t, f in enumerate(steps))
    assert total == pytest.approx(-coef * phis[0])
