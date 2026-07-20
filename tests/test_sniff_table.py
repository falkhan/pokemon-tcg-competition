"""Checkpoint-architecture sniffing (M21): every loader must rebuild the right
net for every checkpoint generation — v1, v2, v3-12id, v3-20id(+v3o), v4."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.encoders import (N_STATE_IDS, N_STATE_IDS_V3, N_STATE_IDS_V4,
                         OPTION_V2_DIM, OPTION_V3_DIM, V4_EXTRA_DIM)
from rl.plan import PLAN_DIM
from rl.plan_iter import _n_ids_of
from rl.policy import (OptionScorer, OptionScorerV2, OptionScorerV3,
                       option_dim_of)


def _ckpt(tmp_path, model, name):
    p = tmp_path / f"{name}.pt"
    torch.save(model.state_dict(), p)
    return p


GENERATIONS = [
    ("v3_12id", dict(n_state_ids=N_STATE_IDS, option_dim=OPTION_V2_DIM), 12),
    ("v3_20id", dict(n_state_ids=N_STATE_IDS_V3, option_dim=OPTION_V2_DIM), 20),
    ("v3o", dict(n_state_ids=N_STATE_IDS_V3, option_dim=OPTION_V3_DIM), 20),
    ("v4", dict(n_state_ids=N_STATE_IDS_V4, option_dim=OPTION_V3_DIM,
                extra_dim=V4_EXTRA_DIM), 25),
]


@pytest.mark.parametrize("name,kw,n_ids", GENERATIONS)
def test_n_ids_and_option_dim_sniff(name, kw, n_ids):
    sd = OptionScorerV3(plan_dim=PLAN_DIM, **kw).state_dict()
    assert _n_ids_of(sd) == n_ids
    assert option_dim_of(sd) == kw["option_dim"]
    assert ("enc_ver" in sd) == (kw.get("extra_dim", 0) > 0)


@pytest.mark.parametrize("name,kw,n_ids", GENERATIONS)
def test_ppo_load_model(tmp_path, name, kw, n_ids):
    from rl.ppo import _load_model
    src = OptionScorerV3(plan_dim=PLAN_DIM, **kw)
    model = _load_model(_ckpt(tmp_path, src, name))
    assert isinstance(model, OptionScorerV3)
    assert model.n_state_ids == n_ids
    assert model.option_dim == kw["option_dim"]
    assert model.extra_dim == kw.get("extra_dim", 0)


def test_ppo_load_model_v2_and_v1(tmp_path):
    from rl.ppo import _load_model
    assert isinstance(_load_model(_ckpt(tmp_path, OptionScorerV2(), "v2")),
                      OptionScorerV2)
    assert isinstance(_load_model(_ckpt(tmp_path, OptionScorer(), "v1")),
                      OptionScorer)


def test_v4_roundtrips_through_npz(tmp_path):
    """save_npz carries the enc_ver buffer (submission sniffs it there)."""
    from rl.policy import save_npz
    m = OptionScorerV3(plan_dim=PLAN_DIM, n_state_ids=N_STATE_IDS_V4,
                       option_dim=OPTION_V3_DIM, extra_dim=V4_EXTRA_DIM)
    save_npz(m, str(tmp_path / "w.npz"))
    w = np.load(tmp_path / "w.npz")
    assert "enc_ver" in w and float(w["enc_ver"]) == 4.0
