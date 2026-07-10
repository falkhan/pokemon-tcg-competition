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


# --- OptionScorerV2 (M7.3) -----------------------------------------------------

def _v2_pair(seed=0):
    torch.manual_seed(seed)
    a = old.OptionScorerV2(hidden=32, embed=4)
    torch.manual_seed(seed)
    b = new.OptionScorerV2(hidden=32, embed=4)
    return a, b


def _v2_batch(batch=3, n_opt=5, seed=1):
    from rl import encoders as enc
    rng = np.random.default_rng(seed)
    return (rng.random((batch, enc.STATE_V2_DIM + enc.N_CONTEXTS), dtype=np.float32),
            rng.integers(0, enc.N_CARD_IDS, (batch, enc.N_STATE_IDS)),
            rng.random((batch, n_opt, enc.OPTION_V2_DIM), dtype=np.float32),
            rng.integers(0, enc.N_CARD_IDS, (batch, n_opt, enc.N_OPTION_IDS)))


def test_v2_state_dict_and_forward_parity():
    a, b = _v2_pair()
    assert list(a.state_dict()) == list(b.state_dict())
    assert "embedding.weight" in a.state_dict()
    sc, sid, op, oid = _v2_batch()
    la, va = a(torch.from_numpy(sc), torch.from_numpy(sid).long(),
               torch.from_numpy(op), torch.from_numpy(oid).long())
    lb, vb = b(torch.from_numpy(sc), torch.from_numpy(sid).long(),
               torch.from_numpy(op), torch.from_numpy(oid).long())
    assert la.shape == (3, 5) and va.shape == (3,)
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_v2_padding_id_is_frozen_zero():
    a, _ = _v2_pair()
    assert torch.equal(a.embedding.weight[0], torch.zeros(4))  # padding_idx row


def test_v2_save_npz_round_trips():
    a, _ = _v2_pair()
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "v2.npz"
        old.save_npz(a, str(p))
        arrays = np.load(p)
        assert "embedding.weight" in arrays
        assert arrays["embedding.weight"].shape == a.embedding.weight.shape


def test_v2_overfits_a_tiny_batch():
    # Verification item 7: the wider inputs + embeddings can actually learn.
    torch.manual_seed(0)
    model = old.OptionScorerV2(hidden=32, embed=4)
    sc, sid, op, oid = _v2_batch(batch=8, n_opt=4, seed=2)
    labels = torch.arange(8) % 4
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        logits, _ = model(torch.from_numpy(sc), torch.from_numpy(sid).long(),
                          torch.from_numpy(op), torch.from_numpy(oid).long())
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optim.zero_grad(); loss.backward(); optim.step()
    logits, _ = model(torch.from_numpy(sc), torch.from_numpy(sid).long(),
                      torch.from_numpy(op), torch.from_numpy(oid).long())
    assert (logits.argmax(dim=1) == labels).float().mean() == 1.0
