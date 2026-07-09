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
