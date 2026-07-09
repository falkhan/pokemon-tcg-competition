"""Old-vs-new parity: rl/ppo.py vs tcg/ppo.py.

The full-update parity test doubles as the pin on the PRESERVED loss quirk:
``loss = policy_loss * VALUE_COEF * value_loss - ...`` multiplies where the
docstring says add. If anyone "fixes" it in only one copy, the identically
seeded models diverge and this file fails. The train() iteration loop needs
the real engine — import-smoke only.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

import rl.ppo as old
import tcg.ppo as new


def test_hyperparameters_parity():
    assert (old.GAMMA, old.LAM, old.CLIP_EPS, old.ENTROPY_COEF, old.VALUE_COEF) == \
           (new.GAMMA, new.LAM, new.CLIP_EPS, new.ENTROPY_COEF, new.VALUE_COEF)


def write_shards(ppo_dir, n_shards=2):
    from rl.encoders import N_CONTEXTS, OPTION_DIM, STATE_DIM
    rng = np.random.default_rng(4)
    ppo_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(n_shards):
        n_decisions = 6 + shard_idx
        n_options = rng.integers(1, 6, size=n_decisions)
        np.savez_compressed(
            ppo_dir / f"ppo_shard_w{shard_idx:02d}.npz",
            states=rng.standard_normal(
                (n_decisions, STATE_DIM + N_CONTEXTS)).astype(np.float32),
            options=rng.standard_normal(
                (int(n_options.sum()), OPTION_DIM)).astype(np.float32),
            n_options=n_options.astype(np.int32),
            actions=(rng.integers(0, n_options)).astype(np.int32),
            logprobs=rng.standard_normal(n_decisions).astype(np.float32),
            values=rng.standard_normal(n_decisions).astype(np.float32),
            rewards=rng.standard_normal(n_decisions).astype(np.float32),
            game_ids=np.sort(rng.integers(0, 3, size=n_decisions)).astype(np.int32),
            players=rng.integers(0, 2, size=n_decisions).astype(np.int32),
        )


def test_load_shards_parity(tmp_path):
    write_shards(tmp_path)
    old_data = old.load_shards(tmp_path)
    new_data = new.load_shards(tmp_path)
    assert sorted(old_data) == sorted(new_data)
    for key in old_data:
        assert np.array_equal(old_data[key], new_data[key]), key


def test_trajectory_slices_parity(tmp_path):
    write_shards(tmp_path)
    data = new.load_shards(tmp_path)
    old_slices = old.trajectory_slices(data)
    new_slices = new.trajectory_slices(data)
    assert len(old_slices) == len(new_slices)
    for old_rows, new_rows in zip(old_slices, new_slices):
        assert np.array_equal(old_rows, new_rows)
    # Within one slice, rows stay chronological (ascending original index).
    for rows in new_slices:
        assert np.all(np.diff(rows) > 0)


@pytest.mark.parametrize("length", [1, 3, 50])
def test_compute_gae_parity(length):
    rng = np.random.default_rng(length)
    rewards = rng.standard_normal(length).astype(np.float32)
    values = rng.standard_normal(length).astype(np.float32)
    old_adv, old_ret = old.compute_gae(rewards, values)
    new_adv, new_ret = new.compute_gae(rewards, values)
    assert np.array_equal(old_adv, new_adv)
    assert np.array_equal(old_ret, new_ret)


def test_compute_gae_hand_computed():
    # lam=1, gamma=1 reduces to "sum of future rewards minus V" (the docstring's
    # sanity anchor), hand-checked on a 3-step toy.
    rewards = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    values = np.array([0.5, 0.25, -0.5], dtype=np.float32)
    advantages, returns = new.compute_gae(rewards, values, gamma=1.0, lam=1.0)
    future_sums = np.array([0.0, -1.0, -1.0], dtype=np.float32)
    assert np.allclose(advantages, future_sums - values)
    assert np.allclose(returns, future_sums)


def test_collate_ppo_parity(tmp_path):
    write_shards(tmp_path)
    data = new.load_shards(tmp_path)
    n_rows = len(data["actions"])
    advantages = np.arange(n_rows, dtype=np.float32)
    returns = -np.arange(n_rows, dtype=np.float32)
    rows = np.array([0, 3, 5, n_rows - 1])
    old_batch = old.collate_ppo(data, rows, advantages, returns)
    new_batch = new.collate_ppo(data, rows, advantages, returns)
    for old_tensor, new_tensor in zip(old_batch, new_batch):
        assert old_tensor.dtype == new_tensor.dtype
        assert torch.equal(old_tensor, new_tensor)


def run_update(module, tmp_path):
    """One identically-seeded ppo_update run; returns (logs, state_dict)."""
    data = module.load_shards(tmp_path)
    advantages = np.zeros(len(data["actions"]), dtype=np.float32)
    returns = np.zeros_like(advantages)
    for rows in module.trajectory_slices(data):
        adv, ret = module.compute_gae(data["rewards"][rows], data["values"][rows])
        advantages[rows], returns[rows] = adv, ret

    torch.manual_seed(0)
    model = module.OptionScorer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # ppo_update shuffles with the GLOBAL numpy RNG; pin both RNGs right before.
    np.random.seed(0)
    torch.manual_seed(0)
    logs = module.ppo_update(model, optimizer, data, advantages, returns,
                             epochs=2, batch_size=4)
    return logs, model.state_dict()


def test_ppo_update_parity_pins_the_loss_quirk(tmp_path):
    """Full-update parity. This intentionally FAILS if the suspected loss bug
    (`policy_loss * VALUE_COEF * value_loss`, see tcg/ppo.py) is fixed in one
    copy but not the other — the quirk is preserved deliberately."""
    write_shards(tmp_path)
    old_logs, old_state = run_update(old, tmp_path)
    new_logs, new_state = run_update(new, tmp_path)
    assert old_logs == new_logs
    assert list(old_state.keys()) == list(new_state.keys())
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key
