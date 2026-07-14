"""Old-vs-new parity: rl/bc.py vs tcg/behavior_cloning.py (dataset + collate).

collect_games / train need the real engine / real data volume — import-smoke
only; the shard-loading math and the padded collate are fully pinned here.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

import rl.bc as old
import tcg.behavior_cloning as new


def make_shards(root):
    """Two shards with ragged menus and per-shard game_ids starting at 0
    (exercising the cross-shard game_base / option_base remapping)."""
    from rl.encoders import N_CONTEXTS, OPTION_DIM, STATE_DIM
    rng = np.random.default_rng(3)
    bc_dir = root / "data" / "bc"
    bc_dir.mkdir(parents=True)
    for shard_idx, menu_sizes in enumerate([(3, 2, 4), (2, 5)]):
        columns = {"states": [], "options": [], "n_options": [], "labels": [],
                   "game_ids": [], "results": []}
        for i, n_options in enumerate(menu_sizes):
            columns["states"].append(
                rng.standard_normal(STATE_DIM + N_CONTEXTS).astype(np.float32))
            columns["options"].append(
                rng.standard_normal((n_options, OPTION_DIM)).astype(np.float32))
            columns["n_options"].append(n_options)
            columns["labels"].append(i % n_options)
            columns["game_ids"].append(i // 2)
            columns["results"].append(float((-1) ** i))
        np.savez_compressed(
            bc_dir / f"shard_{shard_idx:04d}.npz",
            states=np.stack(columns["states"]),
            options=np.concatenate(columns["options"]),
            n_options=np.array(columns["n_options"], dtype=np.int32),
            labels=np.array(columns["labels"], dtype=np.int32),
            game_ids=np.array(columns["game_ids"], dtype=np.int32),
            results=np.array(columns["results"], dtype=np.float32),
        )
    return bc_dir


def test_bcdataset_parity(tmp_path, monkeypatch):
    bc_dir = make_shards(tmp_path)
    # The old dataset reads Path(ROOT/"data"/"bc") from the module global.
    monkeypatch.setattr(old, "ROOT", tmp_path)
    old_dataset = old.BCDataset()
    new_dataset = new.BCDataset(data_dir=bc_dir)

    assert len(old_dataset) == len(new_dataset) == 5
    for name in ("states", "starts", "game_ids", "options", "n_options",
                 "labels", "results"):
        assert np.array_equal(getattr(old_dataset, name), getattr(new_dataset, name)), name

    for i in range(len(old_dataset)):
        old_row, new_row = old_dataset[i], new_dataset[i]
        for old_part, new_part in zip(old_row, new_row):
            assert np.array_equal(old_part, new_part)

    # Hand-check the cross-shard remapping: shard 1's rows must offset their
    # option starts by shard 0's option rows (3+2+4=9) and their game ids by
    # shard 0's max game id + 1 (= 2).
    assert new_dataset.starts.tolist() == [0, 3, 5, 9, 11]
    assert new_dataset.game_ids.tolist() == [0, 0, 1, 2, 2]


def test_collate_parity(tmp_path, monkeypatch):
    bc_dir = make_shards(tmp_path)
    monkeypatch.setattr(old, "ROOT", tmp_path)
    old_dataset = old.BCDataset()
    new_dataset = new.BCDataset(data_dir=bc_dir)

    batch_old = [old_dataset[i] for i in range(len(old_dataset))]
    batch_new = [new_dataset[i] for i in range(len(new_dataset))]
    old_tensors = old.collate(batch_old)
    new_tensors = new.collate(batch_new)
    for old_tensor, new_tensor in zip(old_tensors, new_tensors):
        assert old_tensor.dtype == new_tensor.dtype
        assert torch.equal(old_tensor, new_tensor)


# --- v2 pipeline (M7.3): dataset/collate parity + write_shard v2 columns -------

def make_v2_shards(root):
    from rl.encoders import N_CONTEXTS, N_STATE_IDS, OPTION_DIM, STATE_V2_DIM
    rng = np.random.default_rng(5)
    bc_dir = root / "bc_v2"
    bc_dir.mkdir(parents=True)
    for shard_idx, menu_sizes in enumerate([(3, 2, 4), (2, 5)]):
        cols = {k: [] for k in ("states", "state_ids", "options", "option_ids",
                                "n_options", "labels", "game_ids", "results",
                                "deck_idx")}
        for i, n_options in enumerate(menu_sizes):
            cols["states"].append(
                rng.standard_normal(STATE_V2_DIM + N_CONTEXTS).astype(np.float32))
            cols["state_ids"].append(
                rng.integers(0, 1268, N_STATE_IDS).astype(np.int32))
            cols["options"].append(
                rng.standard_normal((n_options, OPTION_DIM)).astype(np.float32))
            cols["option_ids"].append(
                rng.integers(0, 1268, (n_options, 2)).astype(np.int32))
            cols["n_options"].append(n_options)
            cols["labels"].append(i % n_options)
            cols["game_ids"].append(i // 2)
            cols["results"].append(float((-1) ** i))
            cols["deck_idx"].append(i % 3)
        from tcg.selfplay import write_shard
        write_shard(bc_dir / f"shard_{shard_idx:04d}.npz", cols,
                    new.BC_V2_INT32_COLUMNS, new.BC_FLOAT32_COLUMNS)
    return bc_dir


def test_v2_dataset_and_collate_parity(tmp_path):
    bc_dir = make_v2_shards(tmp_path)
    ds_old = old.BCDatasetV2(bc_dir)
    ds_new = new.BCDatasetV2(bc_dir)
    for attr in ("states", "state_ids", "options", "option_ids", "n_options",
                 "labels", "game_ids", "results", "deck_idx", "starts"):
        assert np.array_equal(getattr(ds_old, attr), getattr(ds_new, attr)), attr
    assert ds_old.game_ids.tolist() == [0, 0, 1, 2, 2]  # cross-shard remap

    batch = [ds_old[i] for i in range(len(ds_old))]
    a = old.collate_v2(batch)
    b = new.collate_v2(batch)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)
    states, state_ids, options, option_ids, valid, labels, results, deck_idx = a
    assert options.shape[1] == 5 and option_ids.shape == (5, 5, 2)
    assert valid.sum() == sum(ds_old.n_options)
    assert deck_idx.tolist() == [0, 1, 2, 0, 1]


def test_v2_batch_feeds_the_v2_network(tmp_path):
    # End-to-end shape check: collated v2 batch -> OptionScorerV2 forward.
    from rl.policy import OptionScorerV2
    bc_dir = make_v2_shards(tmp_path)
    ds = old.BCDatasetV2(bc_dir)
    states, sids, options, oids, valid, labels, _, _ = \
        old.collate_v2([ds[i] for i in range(3)])
    torch.manual_seed(0)
    logits, value = OptionScorerV2(hidden=16, embed=4)(states, sids, options, oids)
    assert logits.shape == (3, options.shape[1]) and value.shape == (3,)
    loss = torch.nn.functional.cross_entropy(logits.masked_fill(~valid, -1e9), labels)
    assert torch.isfinite(loss)


def test_write_shard_rejects_undeclared_v2_column(tmp_path):
    from tcg.selfplay import write_shard
    with pytest.raises(ValueError, match="no declared dtype"):
        write_shard(tmp_path / "x.npz", {"mystery": [1]}, frozenset(), frozenset())


def test_bcdatasetv2_aggregates_multiple_dirs(tmp_path):
    # DAgger training loads teacher round-0 + student-rollout dirs together.
    bc_dir = make_v2_shards(tmp_path)
    single = old.BCDatasetV2(bc_dir)
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(), d2.mkdir()
    (bc_dir / "shard_0000.npz").rename(d1 / "shard_0000.npz")
    (bc_dir / "shard_0001.npz").rename(d2 / "shard_0001.npz")
    multi = old.BCDatasetV2([d1, d2])
    for attr in ("states", "options", "labels", "game_ids", "starts", "results"):
        assert np.array_equal(getattr(single, attr), getattr(multi, attr)), attr


# --- M9 outcome-weighted cloning (AWR-lite) ---------------------------------

def test_weighted_policy_loss_beta0_is_plain_ce():
    torch.manual_seed(1)
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    results = torch.tensor([1.0, -1.0] * 4)
    plain = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(old.weighted_policy_loss(logits, labels, results, 0.0),
                          plain)


def test_weighted_policy_loss_upweights_win_decisions():
    torch.manual_seed(2)
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    results = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])
    beta = 1.0
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    w = torch.exp(beta * results)
    expected = (w / w.mean() * ce).mean()
    got = old.weighted_policy_loss(logits, labels, results, beta)
    assert torch.allclose(got, expected)
    # a win decision's CE moves the weighted loss more than a loss decision's
    assert (w[0] / w.mean()) > 1.0 > (w[2] / w.mean())


def test_weighted_policy_loss_normalization_keeps_scale():
    # All-same-outcome batches must reduce to the plain mean CE (weights = 1),
    # so beta cannot silently rescale the effective lr.
    torch.manual_seed(3)
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 3, (5,))
    for r in (1.0, -1.0):
        results = torch.full((5,), r)
        assert torch.allclose(
            old.weighted_policy_loss(logits, labels, results, 2.0),
            torch.nn.functional.cross_entropy(logits, labels))


# --- M9 DAgger collection: student acts, teacher labels ---------------------

def _scripted_battle(monkeypatch, prompts, winner=0):
    """Wire collect_games_v2's engine seam to a fixed prompt list. Each prompt
    is a builders observation; the battle serves them in order to seat 0, then
    ends with `winner`. Returns the list of picks battle_select received."""
    from types import SimpleNamespace
    selections = []
    frames = [{"current": {"result": -1, "yourIndex": 0}, "obs": o} for o in prompts]
    frames.append({"current": {"result": winner, "yourIndex": 0}})
    it = iter(frames[1:])
    monkeypatch.setattr(old, "battle_start",
                        lambda d0, d1: (frames[0], SimpleNamespace(errorPlayer=-1)))
    monkeypatch.setattr(old, "battle_select",
                        lambda picks: (selections.append(picks), next(it))[1])
    monkeypatch.setattr(old, "battle_finish", lambda: None)
    monkeypatch.setattr(old, "to_observation_class", lambda d: d["obs"])
    return selections


def test_dagger_student_acts_teacher_labels(tmp_path, monkeypatch):
    import json
    from tests.builders import observation, option, player
    from tests.fake_cg import OptionType, SelectContext

    deck = [1] * 10 + [3] * 4 + [6] * 40 + [7] * 6
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": [deck]}))
    menu = [option(OptionType.END), option(OptionType.END)]
    prompts = [observation(player(), player(), context=SelectContext.MAIN,
                           options=menu) for _ in range(2)]
    selections = _scripted_battle(monkeypatch, prompts, winner=0)

    monkeypatch.setattr(old, "_teacher_pilot", lambda t, d, i: lambda od: [0, 1])
    monkeypatch.setattr(old, "_student_pilot", lambda s, d, i: lambda od: [1, 0])

    out = tmp_path / "dagger"
    agreement = old.collect_games_v2(1, pop, out_dir=out, student="stu.pt")

    assert selections == [[1], [1]]          # the STUDENT's pick drove the game
    assert agreement == 0.0                  # student always disagreed
    shard = np.load(next(out.glob("*.npz")))
    assert shard["labels"].tolist() == [0, 0]     # ...but the TEACHER labeled
    assert shard["results"].tolist() == [1.0, 1.0]  # seat 0 decided and won


def test_teacher_collection_unchanged_without_student(tmp_path, monkeypatch):
    import json
    from tests.builders import observation, option, player
    from tests.fake_cg import OptionType, SelectContext

    deck = [1] * 10 + [3] * 4 + [6] * 40 + [7] * 6
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": [deck]}))
    prompts = [observation(player(), player(), context=SelectContext.MAIN,
                           options=[option(OptionType.END), option(OptionType.END)])]
    selections = _scripted_battle(monkeypatch, prompts, winner=1)
    monkeypatch.setattr(old, "_teacher_pilot", lambda t, d, i: lambda od: [1, 0])

    out = tmp_path / "teacher"
    assert old.collect_games_v2(1, pop, out_dir=out) is None
    assert selections == [[1]]               # teacher both acts and labels
    shard = np.load(next(out.glob("*.npz")))
    assert shard["labels"].tolist() == [1]
    assert shard["results"].tolist() == [-1.0]    # seat 0 decided, seat 1 won


def test_load_population_resolves_names_and_paths(tmp_path):
    import json
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": ["lucario", "decks/iono.csv"]}))
    decks = old.load_population(pop)
    assert len(decks) == 2 and all(len(d) == 60 for d in decks)
    assert decks == new.load_population(pop)
