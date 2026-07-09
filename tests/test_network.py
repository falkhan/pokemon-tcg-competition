"""Old-vs-new parity: rl/policy.py vs tcg/network.py.

The checkpoint-key pin is the load-bearing test: every saved ``.pt`` and
exported ``.npz`` keys off the submodule attribute names, so the new class
must produce an identical ``state_dict`` under the same seed.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

import rl.policy as old
import tcg.network as new


def make_pair():
    torch.manual_seed(0)
    old_model = old.OptionScorer()
    torch.manual_seed(0)
    new_model = new.OptionScorer()
    return old_model, new_model


def seeded_batch(batch=3, n_options=5):
    from rl.encoders import N_CONTEXTS, OPTION_DIM, STATE_DIM
    rng = np.random.default_rng(42)
    state_ctx = rng.standard_normal((batch, STATE_DIM + N_CONTEXTS)).astype(np.float32)
    options = rng.standard_normal((batch, n_options, OPTION_DIM)).astype(np.float32)
    return state_ctx, options


def test_state_dict_keys_and_init_parity():
    old_model, new_model = make_pair()
    old_sd, new_sd = old_model.state_dict(), new_model.state_dict()
    assert list(old_sd.keys()) == list(new_sd.keys())
    for key in old_sd:
        assert torch.equal(old_sd[key], new_sd[key]), key


def test_forward_parity():
    old_model, new_model = make_pair()
    state_ctx, options = seeded_batch()
    old_logits, old_value = old_model(torch.from_numpy(state_ctx),
                                      torch.from_numpy(options))
    new_logits, new_value = new_model(torch.from_numpy(state_ctx),
                                      torch.from_numpy(options))
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_value, new_value)


def test_act_greedy_parity():
    old_model, new_model = make_pair()
    state_ctx, options = seeded_batch(batch=1)
    for k in (1, 2, 4):
        assert (old_model.act(state_ctx[0], options[0], k, greedy=True)
                == new_model.act(state_ctx[0], options[0], k, greedy=True))


def test_act_sampling_parity():
    # Gumbel sampling consumes the global torch RNG stream; identical seeds
    # before each call must give identical picks.
    old_model, new_model = make_pair()
    state_ctx, options = seeded_batch(batch=1)
    for seed in (0, 1, 7):
        torch.manual_seed(seed)
        old_picks = old_model.act(state_ctx[0], options[0], 2)
        torch.manual_seed(seed)
        new_picks = new_model.act(state_ctx[0], options[0], 2)
        assert old_picks == new_picks


def test_save_npz_parity(tmp_path):
    old_model, new_model = make_pair()
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    old.save_npz(old_model, str(old_path))
    new.save_npz(new_model, str(new_path))
    with np.load(old_path) as old_npz, np.load(new_path) as new_npz:
        assert sorted(old_npz.files) == sorted(new_npz.files)
        for key in old_npz.files:
            assert np.array_equal(old_npz[key], new_npz[key]), key
