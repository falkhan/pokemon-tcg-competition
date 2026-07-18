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


def _write_new_shard(path, rng, n=3, weights=None):
    """M11 shard: plan columns included; row 0 is a plan-decision row.
    weights (M18a) is written only when given — pre-M18 shards lack it."""
    menus = [2, 3, 2][:n]
    ncands = [2, 0, 0][:n]
    extra = ({} if weights is None
             else {"weights": np.asarray(weights, dtype=np.float32)})
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
        **extra,
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
    (state, plan, sids, menu, oids, label, result, didx, cands, plabel,
     weight) = ds[0]
    assert not plan.any() and cands.shape == (0, PLAN_DIM) and plabel == -1
    # new plan row: candidates + label survive with correct ragged offsets
    row = ds[4]
    assert row[8].shape == (2, PLAN_DIM) and row[9] == 1
    assert ds[5][8].shape == (0, PLAN_DIM)

    batch = pi.collate_v3([ds[i] for i in range(len(ds))])
    (states, plans, sids_t, options, oids_t, valid, labels, results,
     deck_idx, plan_cands, plan_valid, plan_labels, weights) = batch
    assert states.shape[0] == 7 and plans.shape == (7, PLAN_DIM)
    assert plan_cands.shape[1] >= 2
    assert plan_valid[4].sum() == 2 and plan_labels[4] == 1
    assert plan_valid[0].sum() == 0 and plan_labels[0] == -1
    assert weights.tolist() == [1.0] * 7          # weightless shards -> ones


def test_dataset_weights_default_and_passthrough(tmp_path):
    rng = np.random.default_rng(9)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir(), new_dir.mkdir()
    _write_old_shard(old_dir / "shard_0000.npz", rng)          # no weights
    _write_new_shard(new_dir / "shard_w00_0000.npz", rng,
                     weights=[10.0, 1.0, 1.0])
    ds = pi.BCDatasetV3([old_dir, new_dir])
    assert ds[0][10] == 1.0                       # weightless -> default 1
    assert ds[4][10] == 10.0                      # stored weight passes through
    weights = pi.collate_v3([ds[i] for i in range(len(ds))])[12]
    assert weights.shape == (7,)
    assert weights[4] == 10.0 and weights[0] == 1.0


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


def _stub_engine(monkeypatch, selected, n_prompts=2, solve_score=2e5):
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
    monkeypatch.setattr(pi, "encode_state_v3",
                        lambda cur, deck: (np.zeros(4, np.float32),
                                           np.zeros(3, np.int32)))
    monkeypatch.setattr(pi, "encode_context",
                        lambda ctx: np.zeros(2, np.float32))
    monkeypatch.setattr(pi, "encode_option_v2",
                        lambda o, ob: (np.zeros(5, np.float32),
                                       np.zeros(2, np.int32)))
    # teacher solve: the line says option 1 then option 0
    monkeypatch.setattr(pi, "_solve",
                        lambda obs_, deck, dl: (solve_score, [[1], [0]],
                                                [obs, obs]))
    monkeypatch.setattr(pi, "derive_plan", lambda line, trail, root: None)
    monkeypatch.setattr(pi, "enumerate_plans", lambda o: [None])
    # the greedy fallback: stub at its source (imported inside _collect_chunk)
    monkeypatch.setattr("rl.generic_pilot.make_generic_pilot",
                        lambda d: lambda od: [0, 1])
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


def test_collect_expert_below_bar_defers_to_greedy(tmp_path, monkeypatch):
    # Rung 0'': a line below MIN_OVERRIDE_SCORE is NOT executed or labeled —
    # the greedy fallback labels (option 0 here) and the plan stays null.
    selected = []
    _stub_engine(monkeypatch, selected, solve_score=1.0)
    pi.collect("expert", 1, _pop_file(tmp_path), tmp_path / "out", workers=1)
    assert selected == [[0], [0]]                # greedy pick, not the line
    shard = np.load(next((tmp_path / "out").glob("*.npz")))
    assert shard["labels"].tolist() == [0, 0]
    assert not shard["plans"].any()              # no committed plan anywhere
    assert shard["plan_labels"].tolist()[0] == 0  # null plan row


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


def test_v3o_masked_identity_equals_v3h():
    # M16 warm-start invariant: option-identity net on legacy-encoded options
    # (zero-padded numerics, unchanged ids) == pre-M16 net exactly.
    from rl.encoders import OPTION_V3_DIM
    torch.manual_seed(5)
    v3h = OptionScorerV3()
    v3o = pi.load_v3h_into_v3o(v3h.state_dict())
    assert v3o.option_dim == OPTION_V3_DIM
    rng = np.random.default_rng(6)
    sc, sids, opts, oids = _rand_inputs(rng)
    pad = torch.zeros(opts.shape[0], opts.shape[1],
                      OPTION_V3_DIM - opts.shape[2])
    plan = torch.zeros(sc.shape[0], PLAN_DIM)
    l_old, v_old = v3h(sc, plan, sids, opts, oids)
    l_new, v_new = v3o(sc, plan, sids, torch.cat([opts, pad], dim=2), oids)
    assert torch.allclose(l_old, l_new, atol=1e-6)
    assert torch.allclose(v_old, v_new, atol=1e-6)
    # plan head shares only state-side weights — must be untouched
    cands = torch.from_numpy(rng.standard_normal(
        (sc.shape[0], 4, PLAN_DIM)).astype(np.float32))
    assert torch.allclose(v3h.plan_logits(sc, sids, cands),
                          v3o.plan_logits(sc, sids, cands), atol=1e-6)


def test_v3o_from_v3h_hand_aware_source():
    # migration must preserve a 20-id source's state width too
    from rl.encoders import N_STATE_IDS_V3, OPTION_V3_DIM
    torch.manual_seed(7)
    src = pi.load_v3_into_v3h(OptionScorerV3().state_dict())
    v3o = pi.load_v3h_into_v3o(src.state_dict())
    assert v3o.n_state_ids == N_STATE_IDS_V3
    assert v3o.option_dim == OPTION_V3_DIM


def test_v3h_zero_hand_equals_legacy_v3():
    # M15 warm-start invariant: hand-aware(zero hand ids) == legacy net.
    from rl.encoders import N_STATE_IDS_V3
    torch.manual_seed(3)
    legacy = OptionScorerV3()
    v3h = pi.load_v3_into_v3h(legacy.state_dict())
    assert v3h.n_state_ids == N_STATE_IDS_V3
    rng = np.random.default_rng(4)
    sc, sids, opts, oids = _rand_inputs(rng)
    sids20 = torch.cat([sids, torch.zeros(sids.shape[0],
                                          N_STATE_IDS_V3 - N_STATE_IDS,
                                          dtype=torch.long)], dim=1)
    plan = torch.zeros(sc.shape[0], PLAN_DIM)
    l_old, v_old = legacy(sc, plan, sids, opts, oids)
    l_new, v_new = v3h(sc, plan, sids20, opts, oids)
    assert torch.allclose(l_old, l_new, atol=1e-6)
    assert torch.allclose(v_old, v_new, atol=1e-6)


def _vs_stats():
    return {"vs_calls": 0, "vs_errors": 0, "vs_margins": []}


def _vs_harness(monkeypatch, score=1e6):
    """Patch the solver imports _value_solve_factory binds at call time. The
    fake solve CALLS leaf_value on the real obs — so the real state encoder
    runs against the real vnet widths (the M17 crash path)."""
    import rl.turn_solver as ts
    from tests import builders as b

    def fake_solve(obs_, deck_, deadline_s=0.0, dev=False, leaf_value=None,
                   **kw):
        leaf_value(obs_)
        return score, [[0]], []

    monkeypatch.setattr(ts, "solve_turn_line", fake_solve)
    monkeypatch.setattr(
        ts, "score_leaf",
        lambda snap, obs_, dev=False, leaf_value=None, **kw: 0.0)
    monkeypatch.setattr(ts, "_root_snapshot", lambda o: None)
    obs = b.observation()
    obs.search_begin_input = object()
    return obs


def test_value_solve_hand_aware_net_uses_v3_ids(monkeypatch):
    # THE M17 regression test: a 20-id value net must be fed encode_state_v3
    # states. Pre-fix, leaf_value hard-coded encode_state_v2 -> RuntimeError
    # on every call, silently swallowed -> SETUP 0 across 10k games.
    torch.manual_seed(9)
    vnet = pi.load_v3_into_v3h(OptionScorerV3().state_dict())
    obs = _vs_harness(monkeypatch)
    stats = _vs_stats()
    vs = pi._value_solve_factory(vnet, {"me": 0}, stats)
    line, trail = vs(obs, [1] * 60)
    assert line == [[0]]
    assert stats["vs_errors"] == 0 and stats["vs_calls"] == 1
    assert stats["vs_margins"] == [1e6]


def test_value_solve_legacy_net_uses_v2_ids(monkeypatch):
    # the M16 configuration (12-id osv3_setupval2) must keep working
    torch.manual_seed(10)
    vnet = OptionScorerV3()
    obs = _vs_harness(monkeypatch)
    stats = _vs_stats()
    vs = pi._value_solve_factory(vnet, {"me": 0}, stats)
    line, _ = vs(obs, [1] * 60)
    assert line == [[0]]
    assert stats["vs_errors"] == 0


def test_value_solve_margin_gate(monkeypatch):
    # a line below the margin is measured (margin recorded) but NOT committed;
    # a lower --vs-margin threshold flips the same line to a commit
    torch.manual_seed(11)
    vnet = OptionScorerV3()
    obs = _vs_harness(monkeypatch, score=100.0)
    stats = _vs_stats()
    vs = pi._value_solve_factory(vnet, {"me": 0}, stats)
    assert vs(obs, [1] * 60) == (None, None)      # 100 < default 200
    assert stats["vs_margins"] == [100.0]
    vs_low = pi._value_solve_factory(vnet, {"me": 0}, _vs_stats(),
                                     vs_margin=50.0)
    line, _ = vs_low(obs, [1] * 60)
    assert line == [[0]]


def test_value_solve_errors_counted_not_swallowed(monkeypatch, capsys):
    import rl.turn_solver as ts
    from tests import builders as b

    def boom(*a, **kw):
        raise RuntimeError("width mismatch")

    monkeypatch.setattr(ts, "solve_turn_line", boom)
    monkeypatch.setattr(ts, "score_leaf", lambda *a, **kw: 0.0)
    monkeypatch.setattr(ts, "_root_snapshot", lambda o: None)
    obs = b.observation()
    obs.search_begin_input = object()
    stats = _vs_stats()
    vs = pi._value_solve_factory(OptionScorerV3(), {"me": 0}, stats)
    assert vs(obs, [1] * 60) == (None, None)
    assert vs(obs, [1] * 60) == (None, None)
    assert stats["vs_errors"] == 2 and stats["vs_calls"] == 2
    assert "width mismatch" in capsys.readouterr().err   # first-error print


def test_relabel_writes_disagreement_weights(tmp_path):
    rng = np.random.default_rng(12)
    src_dir = tmp_path / "plan"
    src_dir.mkdir()
    _write_new_shard(src_dir / "shard_w00_0000.npz", rng)
    torch.manual_seed(13)
    model = OptionScorerV3()
    ckpt = tmp_path / "student.pt"
    torch.save(model.state_dict(), ckpt)

    pi.relabel([src_dir], str(ckpt), weight=10.0)

    src = np.load(src_dir / "shard_w00_0000.npz")
    dst = np.load(tmp_path / "plan_w" / "shard_w00_0000.npz")
    for k in src.files:                     # original columns byte-identical
        assert np.array_equal(src[k], dst[k])
    assert dst["weights"].dtype == np.float32
    starts = np.cumsum(src["n_options"]) - src["n_options"]
    for i in range(len(src["labels"])):     # weights == per-row act() argmax
        s, m = starts[i], src["n_options"][i]
        pick = model.act(src["states"][i], src["plans"][i],
                         src["state_ids"][i], src["options"][s:s + m],
                         src["option_ids"][s:s + m], 1, greedy=True)[0]
        assert dst["weights"][i] == (1.0 if pick == src["labels"][i] else 10.0)


def test_encode_state_v3_hand_ids():
    from tests import builders as b
    from rl.encoders import N_STATE_IDS_V3, encode_state_v3
    me = b.player(active=b.pokemon(1), hand=[b.hand_card(7), b.hand_card(3),
                                            b.hand_card(6)])
    obs = b.observation(me=me)
    num, ids = encode_state_v3(obs.current, [1] * 60)
    assert ids.shape == (N_STATE_IDS_V3,)
    assert sorted(ids[12:15].tolist()) == [3, 6, 7]   # sorted hand ids
    assert not ids[15:].any()                          # zero padding
