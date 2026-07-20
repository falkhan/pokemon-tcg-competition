"""M21: plan-head-in-PPO (rl/ppo.py plan_coef) + collector parse_pool."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rl.ppo as ppo
from rl.collector import parse_pool
from rl.encoders import N_CONTEXTS, N_STATE_IDS_V3, OPTION_V3_DIM, STATE_V2_DIM
from rl.plan import PLAN_DIM
from rl.policy import OptionScorerV3


def write_plan_shards(root):
    """v3 shards + M21 plan-PPO columns: row 0 of each shard carries a plan
    decision (2 candidates), the rest carry none."""
    rng = np.random.default_rng(9)
    for shard_idx, menu_sizes in enumerate([(3, 2), (4,)]):
        n = len(menu_sizes)
        total = sum(menu_sizes)
        ncands = np.array([2] + [0] * (n - 1), dtype=np.int32)
        np.savez_compressed(
            root / f"ppo_shard_w{shard_idx:02d}.npz",
            states=rng.random((n, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32),
            state_ids=rng.integers(0, 1268, (n, N_STATE_IDS_V3)).astype(np.int32),
            plans=rng.random((n, PLAN_DIM)).astype(np.float32),
            options=rng.random((total, OPTION_V3_DIM)).astype(np.float32),
            option_ids=rng.integers(0, 1268, (total, 2)).astype(np.int32),
            n_options=np.array(menu_sizes, dtype=np.int32),
            actions=np.zeros(n, dtype=np.int32),
            logprobs=rng.random(n).astype(np.float32),
            values=rng.random(n).astype(np.float32),
            rewards=rng.random(n).astype(np.float32),
            game_ids=np.zeros(n, dtype=np.int32),
            players=np.zeros(n, dtype=np.int32),
            plan_cands=rng.random((int(ncands.sum()), PLAN_DIM)).astype(np.float32),
            n_plan_cands=ncands,
            plan_actions=np.array([1] + [-1] * (n - 1), dtype=np.int32),
            plan_logprobs=np.array([-0.7] + [0.0] * (n - 1), dtype=np.float32),
        )


def _model():
    torch.manual_seed(0)
    return OptionScorerV3(n_state_ids=N_STATE_IDS_V3, option_dim=OPTION_V3_DIM)


def test_load_shards_carries_plan_columns(tmp_path):
    write_plan_shards(tmp_path)
    data = ppo.load_shards(tmp_path)
    assert data["plan_cands"].shape == (4, PLAN_DIM)     # 2 shards x 2 cands
    assert list(data["n_plan_cands"]) == [2, 0, 2]
    # cand_starts must be globally offset across shards
    assert list(data["cand_starts"]) == [0, 2, 2]


def test_collate_plan_selects_and_pads(tmp_path):
    write_plan_shards(tmp_path)
    data = ppo.load_shards(tmp_path)
    out = ppo.collate_plan(data, np.arange(len(data["actions"])))
    sub, cands, valid, actions, logprobs = out
    assert list(sub) == [0, 2]                # only the plan-bearing rows
    assert cands.shape == (2, 2, PLAN_DIM) and valid.all()
    assert list(actions) == [1, 1]
    assert torch.allclose(logprobs, torch.tensor([-0.7, -0.7]))
    # batches with no plan rows -> None
    assert ppo.collate_plan(data, np.array([1])) is None


def test_plan_coef_zero_leaves_plan_head_untouched(tmp_path):
    write_plan_shards(tmp_path)
    data = ppo.load_shards(tmp_path)
    n = len(data["actions"])
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()
              if k.startswith(("plan_enc", "plan_head"))}
    np.random.seed(0)
    logs = ppo.ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-2),
                          data, np.ones(n, np.float32), np.ones(n, np.float32),
                          epochs=1, batch_size=2, plan_coef=0.0)
    assert "plan_loss" not in logs
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"{k} moved with plan_coef=0"


def test_plan_coef_trains_the_plan_head(tmp_path):
    write_plan_shards(tmp_path)
    data = ppo.load_shards(tmp_path)
    n = len(data["actions"])
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()
              if k.startswith(("plan_enc", "plan_head"))}
    np.random.seed(0)
    logs = ppo.ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-2),
                          data, np.ones(n, np.float32), np.ones(n, np.float32),
                          epochs=1, batch_size=4, plan_coef=1.0)
    assert np.isfinite(logs["plan_loss"]) and np.isfinite(logs["plan_entropy"])
    after = model.state_dict()
    moved = any(not torch.equal(v, after[k]) for k, v in before.items())
    assert moved, "plan head did not receive gradient under plan_coef=1"


def test_plan_shardless_data_is_a_noop_even_with_plan_coef(tmp_path):
    """Old shards (no plan columns) + plan_coef on -> silently skipped."""
    from tests.test_ppo import write_v3_shards
    write_v3_shards(tmp_path)
    data = ppo.load_shards(tmp_path)
    n = len(data["actions"])
    model = _model()
    np.random.seed(0)
    logs = ppo.ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-3),
                          data, np.ones(n, np.float32), np.ones(n, np.float32),
                          epochs=1, batch_size=2, plan_coef=1.0)
    assert "plan_loss" not in logs           # nothing collected -> key dropped
    assert all(np.isfinite(v) for v in logs.values())


# --- parse_pool -------------------------------------------------------------

def test_parse_pool_specs_and_weights():
    specs, weights = parse_pool(
        ["solver:lucario=0.5", "generic:iono=0.25", "random:kyogre=0.25"],
        checkpoint="ck.pt")
    assert specs[0][0] == "solver" and specs[1][0] == "generic"
    assert weights == pytest.approx([0.5, 0.25, 0.25])


def test_parse_pool_mirror_token():
    specs, weights = parse_pool(["mirror=1", "solver:lucario=1"],
                                checkpoint="current.pt", learn_deck="lucario")
    assert specs[0] == ("model", "current.pt", "lucario")
    assert weights == pytest.approx([0.5, 0.5])


def test_parse_pool_past_token_dropped_when_no_checkpoints(tmp_path,
                                                           monkeypatch):
    import rl.collector as collector
    monkeypatch.setattr(collector, "ROOT", tmp_path)   # no checkpoints dir
    (tmp_path / "checkpoints").mkdir()
    specs, weights = parse_pool(["past=0.5", "solver:lucario=0.5"],
                                checkpoint="ck.pt")
    assert specs == [("solver", "lucario")]
    assert weights == pytest.approx([1.0])


# --- M22b: resolved-identity dedupe -----------------------------------------
# B3's mixture declared `solver:.../deck_20dcd3130bc0.csv=0.30` and
# `solver:lucario=0.15` as two opponents. They are one — that csv is
# byte-identical to decks/lucario.csv — so 45% of training was a single
# opponent while the config read 0.30 and 0.15.

def test_parse_pool_merges_specs_that_resolve_to_the_same_deck(monkeypatch):
    import rl.matchrunner as mr
    monkeypatch.setattr(mr, "resolve_deck",
                        lambda d: [1, 2, 3] if d in ("lucario", "alias.csv") else [9])
    specs, weights = parse_pool(
        ["solver:alias.csv=0.30", "solver:lucario=0.15", "random:kyogre=0.55"],
        checkpoint="ck.pt")
    assert len(specs) == 2, "the two solver entries are one opponent"
    assert weights == pytest.approx([0.45, 0.55])


def test_parse_pool_keeps_distinct_decks_separate(monkeypatch):
    import rl.matchrunner as mr
    monkeypatch.setattr(mr, "resolve_deck",
                        lambda d: {"lucario": [1], "iono": [2], "kyogre": [3]}[d])
    specs, weights = parse_pool(
        ["solver:lucario=0.5", "solver:iono=0.25", "random:kyogre=0.25"],
        checkpoint="ck.pt")
    assert len(specs) == 3
    assert weights == pytest.approx([0.5, 0.25, 0.25])


def test_parse_pool_same_deck_different_pilot_is_not_merged(monkeypatch):
    """solver:lucario and rule:lucario share a deck but are different opponents."""
    import rl.matchrunner as mr
    monkeypatch.setattr(mr, "resolve_deck", lambda d: [1, 2, 3])
    specs, weights = parse_pool(["solver:lucario=0.5", "rule:lucario=0.5"],
                                checkpoint="ck.pt")
    assert len(specs) == 2
    assert weights == pytest.approx([0.5, 0.5])


def test_parse_pool_unresolvable_deck_does_not_crash(monkeypatch):
    import rl.matchrunner as mr
    def _boom(d):
        raise OSError("no such deck")
    monkeypatch.setattr(mr, "resolve_deck", _boom)
    specs, weights = parse_pool(["solver:ghost=1"], checkpoint="ck.pt")
    assert specs == [("solver", "ghost")] and weights == pytest.approx([1.0])
