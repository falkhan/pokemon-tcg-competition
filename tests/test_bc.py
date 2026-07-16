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


def test_load_population_resolves_names_and_paths(tmp_path):
    import json
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": ["lucario", "decks/iono.csv"]}))
    decks = old.load_population(pop)
    assert len(decks) == 2 and all(len(d) == 60 for d in decks)
    assert decks == new.load_population(pop)


def test_collect_dagger_student_advances_teacher_labels(tmp_path, monkeypatch):
    """M9 Leg 2 wiring: the shard's labels come from the TEACHER while the
    game advances on the STUDENT's (sampled) picks. Engine, encoders, and the
    model are stubbed — the real path is exercised by the R-gate smoke run."""
    import json
    from types import SimpleNamespace

    from tests import builders as b

    pop = tmp_path / "population.json"
    pop.write_text(json.dumps({"decks": [[1] * 60]}))
    ckpt = tmp_path / "student.pt"
    torch.save({}, ckpt)

    class FakeStudent:
        def load_state_dict(self, sd):
            pass

        def eval(self):
            pass

        def act(self, *a, **kw):
            return [0]                       # student always picks option 0

    selected = []
    script = {"n": 0}

    def fake_battle_start(d0, d1):
        return {"current": {"result": -1, "yourIndex": 0}}, \
            SimpleNamespace(errorPlayer=-1, errorType=None)

    def fake_battle_select(picks):
        selected.append(picks)
        script["n"] += 1
        done = script["n"] >= 2
        return {"current": {"result": 0 if done else -1, "yourIndex": 0}}

    obs = b.observation(options=[b.option(0), b.option(1)])
    monkeypatch.setattr(old, "OptionScorerV2", FakeStudent)
    monkeypatch.setattr(old, "battle_start", fake_battle_start)
    monkeypatch.setattr(old, "battle_select", fake_battle_select)
    monkeypatch.setattr(old, "battle_finish", lambda: None)
    monkeypatch.setattr(old, "to_observation_class", lambda d: obs)
    monkeypatch.setattr(old, "encode_state_v2",
                        lambda cur, deck: (np.zeros(4, np.float32),
                                           np.zeros(3, np.int32)))
    monkeypatch.setattr(old, "encode_context", lambda ctx: np.zeros(2, np.float32))
    monkeypatch.setattr(old, "encode_option_v2",
                        lambda o, ob: (np.zeros(5, np.float32),
                                       np.zeros(2, np.int32)))
    monkeypatch.setattr(old, "_teacher_pilot",
                        lambda teacher, deck, inst: lambda od: [1])  # teacher: 1

    old.collect_dagger(1, str(ckpt), pop, out_dir=tmp_path / "dagger")

    assert selected == [[0], [0]]            # student's picks drove the game
    shard = np.load(tmp_path / "dagger" / "shard_0000.npz")
    assert shard["labels"].tolist() == [1, 1]  # teacher's picks are the labels
