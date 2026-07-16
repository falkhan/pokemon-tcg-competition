"""M11 rl/plan_iter.py: warm-start invariant, twin parity, dataset mixing,
collection wiring (stubbed engine — the real path is the R-gate smoke run)."""
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import rl.plan_iter as pi
from rl.encoders import (N_CONTEXTS, N_OPTION_IDS, N_STATE_IDS, OPTION_DIM,
                         STATE_V2_DIM)
from rl.plan import PLAN_DIM
from rl.policy import OptionScorerV2, OptionScorerV3


def _rand_inputs(rng, B=3, N=5):
    return (torch.from_numpy(rng.standard_normal(
                (B, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32)),
            torch.from_numpy(rng.integers(0, 1268, (B, N_STATE_IDS))).long(),
            torch.from_numpy(rng.standard_normal(
                (B, N, OPTION_DIM)).astype(np.float32)),
            torch.from_numpy(rng.integers(0, 1268, (B, N, N_OPTION_IDS))).long())


def test_v3_zero_plan_equals_v2_after_warm_start():
    torch.manual_seed(0)
    v2 = OptionScorerV2()
    v3 = pi.load_v2_into_v3(v2.state_dict())
    rng = np.random.default_rng(1)
    sc, sids, opts, oids = _rand_inputs(rng)
    zeros = torch.zeros(sc.shape[0], PLAN_DIM)
    l2, val2 = v2(sc, sids, opts, oids)
    l3, val3 = v3(sc, zeros, sids, opts, oids)
    assert torch.allclose(l2, l3, atol=1e-6)
    assert torch.allclose(val2, val3, atol=1e-6)


def test_rl_tcg_v3_twins_are_identical():
    from tcg.network import OptionScorerV3 as TcgV3
    from tcg.network import PLAN_DIM as TCG_PLAN_DIM
    assert TCG_PLAN_DIM == PLAN_DIM
    torch.manual_seed(0)
    rl_m = OptionScorerV3()
    tcg_m = TcgV3()
    tcg_m.load_state_dict(rl_m.state_dict())
    rng = np.random.default_rng(2)
    sc, sids, opts, oids = _rand_inputs(rng)
    plan = torch.from_numpy(
        rng.standard_normal((sc.shape[0], PLAN_DIM)).astype(np.float32))
    cands = torch.from_numpy(
        rng.standard_normal((sc.shape[0], 4, PLAN_DIM)).astype(np.float32))
    for a, b_ in zip(rl_m(sc, plan, sids, opts, oids),
                     tcg_m(sc, plan, sids, opts, oids)):
        assert torch.equal(a, b_)
    assert torch.equal(rl_m.plan_logits(sc, sids, cands),
                       tcg_m.plan_logits(sc, sids, cands))


def _write_old_shard(path, rng, n=4):
    """Pre-M11 v2 shard: 9 columns, no plan data."""
    menus = [3, 2, 4, 2][:n]
    np.savez_compressed(
        path,
        states=rng.standard_normal(
            (n, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32),
        state_ids=rng.integers(0, 1268, (n, N_STATE_IDS)).astype(np.int32),
        options=rng.standard_normal(
            (sum(menus), OPTION_DIM)).astype(np.float32),
        option_ids=rng.integers(0, 1268, (sum(menus), 2)).astype(np.int32),
        n_options=np.array(menus, dtype=np.int32),
        labels=np.zeros(n, dtype=np.int32),
        game_ids=np.arange(n, dtype=np.int32) // 2,
        results=np.ones(n, dtype=np.float32),
        deck_idx=np.zeros(n, dtype=np.int32),
    )


def _write_new_shard(path, rng, n=3):
    """M11 shard: plan columns included; row 0 is a plan-decision row."""
    menus = [2, 3, 2][:n]
    ncands = [2, 0, 0][:n]
    np.savez_compressed(
        path,
        states=rng.standard_normal(
            (n, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32),
        plans=rng.standard_normal((n, PLAN_DIM)).astype(np.float32),
        state_ids=rng.integers(0, 1268, (n, N_STATE_IDS)).astype(np.int32),
        options=rng.standard_normal(
            (sum(menus), OPTION_DIM)).astype(np.float32),
        option_ids=rng.integers(0, 1268, (sum(menus), 2)).astype(np.int32),
        n_options=np.array(menus, dtype=np.int32),
        labels=np.zeros(n, dtype=np.int32),
        game_ids=np.zeros(n, dtype=np.int32),
        results=-np.ones(n, dtype=np.float32),
        deck_idx=np.ones(n, dtype=np.int32),
        plan_cands=rng.standard_normal(
            (sum(ncands), PLAN_DIM)).astype(np.float32),
        n_plan_cands=np.array(ncands, dtype=np.int32),
        plan_labels=np.array([1, -1, -1][:n], dtype=np.int32),
    )


def test_dataset_mixes_old_and_new_shards(tmp_path):
    rng = np.random.default_rng(3)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir(), new_dir.mkdir()
    _write_old_shard(old_dir / "shard_0000.npz", rng)
    _write_new_shard(new_dir / "shard_w00_0000.npz", rng)

    ds = pi.BCDatasetV3([old_dir, new_dir])
    assert len(ds) == 7
    # old rows: zero plans, no candidates, label -1
    state, plan, sids, menu, oids, label, result, didx, cands, plabel = ds[0]
    assert not plan.any() and cands.shape == (0, PLAN_DIM) and plabel == -1
    # new plan row: candidates + label survive with correct ragged offsets
    row = ds[4]
    assert row[8].shape == (2, PLAN_DIM) and row[9] == 1
    assert ds[5][8].shape == (0, PLAN_DIM)

    batch = pi.collate_v3([ds[i] for i in range(len(ds))])
    (states, plans, sids_t, options, oids_t, valid, labels, results,
     deck_idx, plan_cands, plan_valid, plan_labels) = batch
    assert states.shape[0] == 7 and plans.shape == (7, PLAN_DIM)
    assert plan_cands.shape[1] >= 2
    assert plan_valid[4].sum() == 2 and plan_labels[4] == 1
    assert plan_valid[0].sum() == 0 and plan_labels[0] == -1


class _FakeStudent:
    """act -> always option 0; act_plan -> always candidate 0."""

    def load_state_dict(self, sd):
        pass

    def eval(self):
        pass

    def act(self, *a, **kw):
        return [0]

    def act_plan(self, *a, **kw):
        return 0


def _stub_engine(monkeypatch, selected, n_prompts=2):
    from tests import builders as b
    from tests.fake_cg import OptionType, SelectContext

    obs = b.observation(options=[b.option(OptionType.END),
                                 b.option(OptionType.END)],
                        context=SelectContext.MAIN)
    script = {"n": 0}

    def fake_start(d0, d1):
        return {"current": {"result": -1, "yourIndex": 0}}, \
            SimpleNamespace(errorPlayer=-1, errorType=None)

    def fake_select(picks):
        selected.append(picks)
        script["n"] += 1
        done = script["n"] >= n_prompts
        return {"current": {"result": 0 if done else -1, "yourIndex": 0}}

    monkeypatch.setattr(pi, "battle_start", fake_start)
    monkeypatch.setattr(pi, "battle_select", fake_select)
    monkeypatch.setattr(pi, "battle_finish", lambda: None)
    monkeypatch.setattr(pi, "to_observation_class", lambda d: obs)
    monkeypatch.setattr(pi, "encode_state_v2",
                        lambda cur, deck: (np.zeros(4, np.float32),
                                           np.zeros(3, np.int32)))
    monkeypatch.setattr(pi, "encode_context",
                        lambda ctx: np.zeros(2, np.float32))
    monkeypatch.setattr(pi, "encode_option_v2",
                        lambda o, ob: (np.zeros(5, np.float32),
                                       np.zeros(2, np.int32)))
    # teacher solve: the line says option 1 then option 0
    monkeypatch.setattr(pi, "_solve",
                        lambda obs_, deck, dl: (1.0, [[1], [0]], [obs, obs]))
    monkeypatch.setattr(pi, "derive_plan", lambda line, trail, root: None)
    monkeypatch.setattr(pi, "enumerate_plans", lambda o: [None])
    return obs


def _pop_file(tmp_path):
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": [[1] * 60]}))
    return pop


def test_collect_expert_executes_and_labels_the_line(tmp_path, monkeypatch):
    selected = []
    _stub_engine(monkeypatch, selected)
    pi.collect("expert", 1, _pop_file(tmp_path), tmp_path / "out", workers=1)
    assert selected == [[1], [0]]                # the solver line was executed
    shard = np.load(next((tmp_path / "out").glob("*.npz")))
    assert shard["labels"].tolist() == [1, 0]    # ...and labeled
    assert shard["plan_labels"].tolist()[0] == 0  # null plan row recorded
    assert shard["n_plan_cands"].tolist() == [1, 0]


def test_collect_ei_student_advances_teacher_labels(tmp_path, monkeypatch):
    selected = []
    _stub_engine(monkeypatch, selected)
    monkeypatch.setattr(pi, "OptionScorerV3", _FakeStudent)
    ckpt = tmp_path / "student.pt"
    torch.save({}, ckpt)
    pi.collect("ei", 1, _pop_file(tmp_path), tmp_path / "out",
               checkpoint=str(ckpt), tau=1.0, workers=1)
    assert selected == [[0], [0]]                # student's picks drove the game
    shard = np.load(next((tmp_path / "out").glob("*.npz")))
    assert shard["labels"].tolist() == [1, 1]    # teacher labels every prompt
