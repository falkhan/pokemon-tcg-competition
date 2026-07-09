"""Old-vs-new parity for the self-play collector's pure parts.

The game-collection loop itself needs the real engine + multiprocessing and is
NOT testable here (import-smoke only); write_shard and default_pool are.
"""
import pytest

np = pytest.importorskip("numpy")

import rl.collector as old
import tcg.selfplay as new


def ppo_style_columns():
    rng = np.random.default_rng(0)
    n_decisions, state_dim, option_dim = 7, 4, 3
    columns = {name: [] for name in ("states", "options", "n_options", "actions",
                                     "logprobs", "values", "rewards", "game_ids",
                                     "players")}
    for i in range(n_decisions):
        n_options = int(rng.integers(1, 5))
        columns["states"].append(rng.standard_normal(state_dim).astype(np.float32))
        columns["options"].append(
            rng.standard_normal((n_options, option_dim)).astype(np.float32))
        columns["n_options"].append(n_options)
        columns["actions"].append(int(rng.integers(0, n_options)))
        columns["logprobs"].append(float(rng.standard_normal()))
        columns["values"].append(float(rng.standard_normal()))
        columns["rewards"].append(float(rng.standard_normal()))
        columns["game_ids"].append(i // 3)
        columns["players"].append(i % 2)
    return columns


def write_ppo_shard_old_style(path, cols):
    # transcription of the inline np.savez_compressed in rl/collector.py _play_worker
    np.savez_compressed(
        path,
        states=np.stack(cols["states"]),
        options=np.concatenate(cols["options"]),
        n_options=np.array(cols["n_options"], dtype=np.int32),
        actions=np.array(cols["actions"], dtype=np.int32),
        logprobs=np.array(cols["logprobs"], dtype=np.float32),
        values=np.array(cols["values"], dtype=np.float32),
        rewards=np.array(cols["rewards"], dtype=np.float32),
        game_ids=np.array(cols["game_ids"], dtype=np.int32),
        players=np.array(cols["players"], dtype=np.int32),
    )


def assert_npz_equal(path_a, path_b):
    with np.load(path_a) as a, np.load(path_b) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert a[key].dtype == b[key].dtype, key
            assert np.array_equal(a[key], b[key]), key


def test_write_shard_ppo_layout(tmp_path):
    columns = ppo_style_columns()
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    write_ppo_shard_old_style(old_path, columns)
    new.write_shard(new_path, columns, new.PPO_INT32_COLUMNS, new.PPO_FLOAT32_COLUMNS)
    assert_npz_equal(old_path, new_path)


def test_write_shard_bc_layout(tmp_path):
    rng = np.random.default_rng(1)
    shard = {"states": [rng.standard_normal(4).astype(np.float32) for _ in range(5)],
             "options": [rng.standard_normal((n, 3)).astype(np.float32)
                         for n in (2, 1, 4, 3, 2)],
             "n_options": [2, 1, 4, 3, 2],
             "labels": [0, 0, 3, 1, 1],
             "game_ids": [0, 0, 1, 1, 2],
             "results": [1.0, 1.0, -1.0, -1.0, 0.0]}
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    # transcription of the inline flush() in rl/bc.py collect_games
    np.savez_compressed(
        old_path,
        states=np.stack(shard["states"]),
        options=np.concatenate(shard["options"]),
        n_options=np.array(shard["n_options"], dtype=np.int32),
        labels=np.array(shard["labels"], dtype=np.int32),
        game_ids=np.array(shard["game_ids"], dtype=np.int32),
        results=np.array(shard["results"], dtype=np.float32),
    )
    # same sets as tcg.behavior_cloning.BC_*_COLUMNS (importing that needs torch)
    new.write_shard(new_path, shard, frozenset({"n_options", "labels", "game_ids"}),
                    frozenset({"results"}))
    assert_npz_equal(old_path, new_path)


def test_write_shard_value_training_layout(tmp_path):
    rng = np.random.default_rng(2)
    states = [rng.standard_normal(6).astype(np.float32) for _ in range(4)]
    outcomes = [1.0, -1.0, -1.0, 0.0]
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    # transcription of the inline np.savez_compressed in rl/value_train.py collect
    np.savez_compressed(old_path, states=np.stack(states).astype(np.float32),
                        outcomes=np.array(outcomes, dtype=np.float32))
    new.write_shard(new_path, {"states": states, "outcomes": outcomes},
                    int32_columns=frozenset(),
                    float32_columns=frozenset({"outcomes"}))
    assert_npz_equal(old_path, new_path)


def test_write_shard_rejects_undeclared_columns(tmp_path):
    with pytest.raises(ValueError):
        new.write_shard(tmp_path / "x.npz", {"mystery": [1]},
                        frozenset(), frozenset())


def test_default_pool_parity():
    # Deterministic today: checkpoints/ has no ppo_it*.pt promoted selves.
    old_specs, old_weights = old.default_pool("checkpoints/bc_v1.pt")
    new_specs, new_weights = new.default_pool("checkpoints/bc_v1.pt")
    assert old_specs == new_specs
    assert old_weights == new_weights


def test_constants_parity():
    assert new.PRIZE_SHAPING == old.PRIZE_SHAPING
    assert new.LEARN_DECK == old.LEARN_DECK
    assert new.OUT_DIR == old.OUT_DIR
