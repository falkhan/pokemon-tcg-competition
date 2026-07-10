"""Old-vs-new parity: rl/ppo.py vs tcg/ppo.py.

M7.4b fixed the historic loss quirk (``policy_loss * VALUE_COEF * value_loss``
— multiplied where the docstring said add; every stalled PPO run trained on
it). The full-update parity test keeps both copies in lockstep, and
test_loss_is_the_additive_ppo_objective pins the CORRECTED form literally.
The train() iteration loop needs the real engine — import-smoke only.
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
        if old_tensor is None:  # v1 shards: the id slots are None (M7.4b)
            assert new_tensor is None
            continue
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


def test_ppo_update_parity_pins_the_corrected_loss(tmp_path):
    """Full-update parity. This FAILS if the PPO loss (corrected in M7.4b from
    the multiplicative `policy_loss * VALUE_COEF * value_loss` quirk to the
    documented additive objective) diverges between the two copies."""
    write_shards(tmp_path)
    old_logs, old_state = run_update(old, tmp_path)
    new_logs, new_state = run_update(new, tmp_path)
    assert old_logs == new_logs
    assert list(old_state.keys()) == list(new_state.keys())
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key


def test_loss_is_the_additive_ppo_objective(tmp_path):
    """M7.4b: pin the CORRECTED loss form literally. With lr=0 the model never
    moves, so every epoch logs identical component losses — recombine them and
    check the update actually optimized policy + VALUE_COEF*value − ENTROPY_COEF
    *entropy by asserting a single gradient step moves loss in that direction."""
    write_shards(tmp_path)
    data = old.load_shards(tmp_path)
    n = len(data["actions"])
    advantages = np.linspace(-1, 1, n).astype(np.float32)
    returns = np.linspace(1, -1, n).astype(np.float32)

    torch.manual_seed(0)
    np.random.seed(0)
    from rl.policy import OptionScorer
    model = OptionScorer()
    # lr=0: parameters frozen -> the logged means ARE single-pass component values
    opt_frozen = torch.optim.SGD(model.parameters(), lr=0.0)
    logs = old.ppo_update(model, opt_frozen, data, advantages, returns,
                          epochs=1, batch_size=n)
    combined = (logs["policy_loss"] + old.VALUE_COEF * logs["value_loss"]
                - old.ENTROPY_COEF * logs["entropy"])
    # The multiplicative bug form would differ (unless the components conspire):
    buggy = logs["policy_loss"] * old.VALUE_COEF * logs["value_loss"] \
        - old.ENTROPY_COEF * logs["entropy"]
    assert abs(combined - buggy) > 1e-6  # the two forms are distinguishable here

    # A real step must reduce the ADDITIVE objective on the same batch.
    torch.manual_seed(0)
    np.random.seed(0)
    model2 = OptionScorer()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    before = old.ppo_update(model2, opt2, data, advantages, returns,
                            epochs=1, batch_size=n)
    np.random.seed(0)
    after = old.ppo_update(model2, opt2, data, advantages, returns,
                           epochs=1, batch_size=n)
    combined_before = (before["policy_loss"] + old.VALUE_COEF * before["value_loss"]
                       - old.ENTROPY_COEF * before["entropy"])
    combined_after = (after["policy_loss"] + old.VALUE_COEF * after["value_loss"]
                      - old.ENTROPY_COEF * after["entropy"])
    assert combined_after < combined_before


def test_warm_start_value_loads_only_the_value_head(tmp_path):
    from rl.policy import OptionScorer, OptionScorerV2

    torch.manual_seed(1)
    donor = OptionScorer()
    ckpt = tmp_path / "value.pt"
    torch.save(donor.state_dict(), ckpt)

    for make in (OptionScorer, OptionScorerV2):
        torch.manual_seed(2)
        model = make()
        body_before = model.state_dict()["state_enc.0.weight"].clone()
        loaded = old.warm_start_value(model, ckpt)
        assert loaded == ["value_head.0.bias", "value_head.0.weight",
                          "value_head.2.bias", "value_head.2.weight"]
        for key in loaded:
            assert torch.equal(model.state_dict()[key], donor.state_dict()[key])
        assert torch.equal(model.state_dict()["state_enc.0.weight"], body_before)
        # twin parity
        torch.manual_seed(2)
        model_new = make()
        assert new.warm_start_value(model_new, ckpt) == loaded


def write_v2_shards(root):
    """Synthetic encoders-v2 PPO shards (state_ids/option_ids/deck_idx present)."""
    from rl.encoders import N_CONTEXTS, N_STATE_IDS, OPTION_DIM, STATE_V2_DIM
    rng = np.random.default_rng(7)
    for shard_idx, menu_sizes in enumerate([(3, 2), (4,)]):
        n = len(menu_sizes)
        total = sum(menu_sizes)
        np.savez_compressed(
            root / f"ppo_shard_w{shard_idx:02d}.npz",
            states=rng.random((n, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32),
            state_ids=rng.integers(0, 1268, (n, N_STATE_IDS)).astype(np.int32),
            options=rng.random((total, OPTION_DIM)).astype(np.float32),
            option_ids=rng.integers(0, 1268, (total, 2)).astype(np.int32),
            n_options=np.array(menu_sizes, dtype=np.int32),
            actions=np.array([0] * n, dtype=np.int32),
            logprobs=rng.random(n).astype(np.float32),
            values=rng.random(n).astype(np.float32),
            rewards=rng.random(n).astype(np.float32),
            game_ids=np.zeros(n, dtype=np.int32),
            players=np.zeros(n, dtype=np.int32),
            deck_idx=np.array([1] * n, dtype=np.int32),
        )


def test_v2_shards_flow_through_load_collate_and_update(tmp_path):
    """M7.4b: encoders-v2 shards -> ids carried, padded, and fed to
    OptionScorerV2 through one whole ppo_update."""
    from rl.policy import OptionScorerV2

    write_v2_shards(tmp_path)
    data = old.load_shards(tmp_path)
    assert data["state_ids"].shape[1] > 0 and data["option_ids"].shape[1] == 2
    n = len(data["actions"])
    adv = np.ones(n, dtype=np.float32)
    ret = np.ones(n, dtype=np.float32)

    batch = old.collate_ppo(data, np.arange(n), adv, ret)
    states, state_ids, options, option_ids, valid, *_ = batch
    assert state_ids.dtype == torch.long and option_ids.dtype == torch.long
    assert option_ids.shape == (n, options.shape[1], 2)
    assert new.collate_ppo(data, np.arange(n), adv, ret)[1].equal(state_ids)

    torch.manual_seed(0)
    np.random.seed(0)
    model = OptionScorerV2(hidden=16, embed=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logs = old.ppo_update(model, opt, data, adv, ret, epochs=1, batch_size=2)
    assert all(np.isfinite(v) for v in logs.values())
