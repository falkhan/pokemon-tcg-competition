"""migrate_v3_to_v4: the warm-start invariant, fifth use (M21).

The migrated net must equal the source v3 net EXACTLY on real v3 inputs no
matter what garbage sits in the new v4 block / memory-id slots — every new
input column of state_enc.0 is zero-initialized.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.encoders import (EMBED_DIM, N_CONTEXTS, N_STATE_IDS_V3, N_STATE_IDS_V4,
                         OPTION_V3_DIM, STATE_V2_DIM, V4_EXTRA_DIM)
from rl.plan import PLAN_DIM
from rl.plan_iter import migrate_v3_to_v4
from rl.policy import OptionScorerV3


@pytest.fixture(scope="module")
def nets():
    torch.manual_seed(0)
    v3 = OptionScorerV3(plan_dim=PLAN_DIM, n_state_ids=N_STATE_IDS_V3,
                        option_dim=OPTION_V3_DIM)
    v4 = migrate_v3_to_v4(v3.state_dict())
    return v3, v4


def _inputs(batch=3, n_opts=4, seed=1):
    rng = np.random.default_rng(seed)
    ctx3 = rng.standard_normal((batch, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32)
    garbage = rng.standard_normal((batch, V4_EXTRA_DIM)).astype(np.float32) * 9.0
    ctx4 = np.concatenate([ctx3, garbage], axis=1)
    ids3 = rng.integers(0, 1200, (batch, N_STATE_IDS_V3))
    mem_ids = rng.integers(0, 1200, (batch, N_STATE_IDS_V4 - N_STATE_IDS_V3))
    ids4 = np.concatenate([ids3, mem_ids], axis=1)
    plan = rng.standard_normal((batch, PLAN_DIM)).astype(np.float32)
    opts = rng.standard_normal((batch, n_opts, OPTION_V3_DIM)).astype(np.float32)
    oids = rng.integers(0, 1200, (batch, n_opts, 2))
    t = torch.from_numpy
    return (t(ctx3), t(ctx4), t(ids3).long(), t(ids4).long(), t(plan),
            t(opts), t(oids).long())


def test_enc_ver_buffer(nets):
    v3, v4 = nets
    assert "enc_ver" not in v3.state_dict()
    assert float(v4.state_dict()["enc_ver"]) == 4.0
    assert v4.extra_dim == V4_EXTRA_DIM and v4.n_state_ids == N_STATE_IDS_V4


def test_forward_parity_with_garbage_v4_inputs(nets):
    v3, v4 = nets
    ctx3, ctx4, ids3, ids4, plan, opts, oids = _inputs()
    logits3, value3 = v3(ctx3, plan, ids3, opts, oids)
    logits4, value4 = v4(ctx4, plan, ids4, opts, oids)
    # zero-init columns widen the matmul -> summation-order float noise
    # of O(1e-8); the invariant is exactness up to that
    assert torch.allclose(logits3, logits4, atol=1e-6, rtol=0)
    assert torch.allclose(value3, value4, atol=1e-6, rtol=0)


def test_plan_logits_parity(nets):
    v3, v4 = nets
    ctx3, ctx4, ids3, ids4, _, _, _ = _inputs(seed=2)
    cands = torch.randn(3, 5, PLAN_DIM)
    assert torch.allclose(v3.plan_logits(ctx3, ids3, cands),
                          v4.plan_logits(ctx4, ids4, cands),
                          atol=1e-6, rtol=0)


def test_migration_is_input_sensitive_where_it_should_be(nets):
    """Different V3 inputs still change the output (the copy isn't degenerate)."""
    v3, v4 = nets
    _, ctx4, _, ids4, plan, opts, oids = _inputs(seed=3)
    logits_a, _ = v4(ctx4, plan, ids4, opts, oids)
    ctx4b = ctx4.clone()
    ctx4b[:, 0] += 1.0                     # perturb a REAL v3 column
    logits_b, _ = v4(ctx4b, plan, ids4, opts, oids)
    assert not torch.allclose(logits_a, logits_b)
