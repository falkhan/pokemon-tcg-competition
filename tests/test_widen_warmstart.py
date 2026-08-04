"""The warm-start invariant for the M41b width retrain (rl.plan_iter).

The census that motivated this found three appended blocks no checkpoint had
ever trained on (M41 100-104, M42 105-114, M41b 115-142). Consuming them means
loading a width-100 net into a width-143 one, and the ONLY thing that makes
the resulting experiment readable is the invariant these tests pin:

    at initialisation, the wide net reproduces the narrow net EXACTLY.

If that holds, a null result means "the features did not help"; if it silently
does not, a null means "the warm start scrambled a champion" and we would draw
the opposite conclusion from the same number.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402

from rl.encoders import (EMBED_DIM, N_OPTION_IDS, OPTION_M28_DIM,  # noqa: E402
                         OPTION_M41B_DIM)
from rl.plan_iter import widen_option_dim  # noqa: E402
from rl.policy import OptionScorerV3  # noqa: E402

NARROW = OPTION_M28_DIM          # 100 — what every checkpoint on the box is
WIDE = OPTION_M41B_DIM           # 143 — what the encoder emits today


def _narrow_net(seed=0, n_state_ids=20):
    torch.manual_seed(seed)
    net = OptionScorerV3(n_state_ids=n_state_ids, option_dim=NARROW)
    net.eval()
    return net


def _batch(net, n_opt=6, batch=3, width=NARROW, seed=1):
    """Random inputs at `width`; the extra columns are zero when width > NARROW
    so the two nets are being asked the SAME question."""
    rng = np.random.default_rng(seed)
    state_ctx = torch.from_numpy(
        rng.standard_normal((batch, net.state_enc[0].in_features
                             - net.n_state_ids * EMBED_DIM
                             - net.plan_dim)).astype(np.float32))
    plan = torch.from_numpy(
        rng.standard_normal((batch, net.plan_dim)).astype(np.float32))
    state_ids = torch.from_numpy(
        rng.integers(0, 50, (batch, net.n_state_ids)).astype(np.int64))
    options = torch.zeros((batch, n_opt, width), dtype=torch.float32)
    options[:, :, :NARROW] = torch.from_numpy(
        rng.standard_normal((batch, n_opt, NARROW)).astype(np.float32))
    option_ids = torch.from_numpy(
        rng.integers(0, 50, (batch, n_opt, N_OPTION_IDS)).astype(np.int64))
    return state_ctx, plan, state_ids, options, option_ids


def test_day_one_is_behaviourally_identical():
    """THE invariant, stated at the precision it actually holds.

    The weight copy is exact, but the two forward passes run GEMMs of
    different shape, so the reduction ORDER differs and the logits can
    disagree in the last bits. Measured on this fixture: max |diff| =
    1.49e-08, which is ~8x BELOW float32 eps (1.19e-07); the value head is
    exactly equal. So "day one is exactly equivalent" — the phrase the plan
    inherits from the earlier warm starts — is true behaviourally and false
    bitwise, and the distinction is worth pinning rather than glossing:
    the pilot argsorts logits, so what must be preserved is the RANKING, and
    that is asserted exactly below.
    """
    narrow = _narrow_net()
    wide = widen_option_dim(narrow.state_dict(), WIDE)
    wide.eval()
    assert wide.option_dim == WIDE

    sc, plan, sids, opts_n, oids = _batch(narrow)
    opts_w = torch.zeros((*opts_n.shape[:2], WIDE), dtype=torch.float32)
    opts_w[:, :, :NARROW] = opts_n

    with torch.no_grad():
        logits_n, value_n = narrow(sc, plan, sids, opts_n, oids)
        logits_w, value_w = wide(sc, plan, sids, opts_w, oids)

    assert torch.equal(value_n, value_w)                    # exact
    assert (logits_n - logits_w).abs().max().item() < np.finfo(np.float32).eps
    # the decision-relevant property: identical argmax AND identical full
    # ordering, so every pilot that argsorts serves the same actions
    assert torch.equal(logits_n.argmax(-1), logits_w.argmax(-1))
    assert torch.equal(logits_n.argsort(-1), logits_w.argsort(-1))


def test_the_new_columns_start_with_exactly_zero_influence():
    """The appended features must EARN their influence from the gradient.
    Firing them at init must not move a single logit."""
    narrow = _narrow_net(seed=3)
    wide = widen_option_dim(narrow.state_dict(), WIDE)
    wide.eval()
    sc, plan, sids, opts, oids = _batch(narrow, width=WIDE, seed=7)

    hot = opts.clone()
    hot[:, :, NARROW:] = 3.14           # light up every new column, hard
    with torch.no_grad():
        base_logits, base_value = wide(sc, plan, sids, opts, oids)
        hot_logits, hot_value = wide(sc, plan, sids, hot, oids)
    assert torch.equal(base_logits, hot_logits)
    assert torch.equal(base_value, hot_value)
    # ...and the weights that would carry them are exactly zero
    assert torch.count_nonzero(
        wide.option_enc[0].weight[:, NARROW:WIDE]) == 0


def test_the_embedding_block_moves_with_the_width():
    """The id-embedding columns live AFTER the numeric ones, so widening must
    shift them right — not overwrite them with the new numeric block. That is
    the bug this shape of migration exists to avoid."""
    narrow = _narrow_net(seed=5)
    old_w = narrow.option_enc[0].weight.detach().clone()
    wide = widen_option_dim(narrow.state_dict(), WIDE)
    new_w = wide.option_enc[0].weight.detach()
    n_embed = N_OPTION_IDS * EMBED_DIM
    assert torch.equal(old_w[:, :NARROW], new_w[:, :NARROW])
    assert torch.equal(old_w[:, NARROW:NARROW + n_embed],
                       new_w[:, WIDE:WIDE + n_embed])


def test_widening_to_the_same_width_is_a_plain_load():
    narrow = _narrow_net(seed=11)
    same = widen_option_dim(narrow.state_dict(), NARROW)
    for a, b in zip(narrow.state_dict().values(), same.state_dict().values()):
        assert torch.equal(a, b)


def test_narrowing_is_refused():
    """A narrower target is a RE-LAYOUT, not an append — exactly the § 2c
    silent mis-slice. It must raise rather than produce a plausible net."""
    narrow = _narrow_net(seed=13)
    with pytest.raises(ValueError, match="refusing to narrow"):
        widen_option_dim(narrow.state_dict(), NARROW - 10)


def test_every_non_option_tensor_survives_untouched():
    """Only option_enc.0.weight may change. A migration that quietly reset the
    value head or the trunk would look like a warm start and behave like a
    scratch train."""
    narrow = _narrow_net(seed=17)
    sd = narrow.state_dict()
    wide_sd = widen_option_dim(sd, WIDE).state_dict()
    for key, tensor in sd.items():
        if key == "option_enc.0.weight":
            continue
        assert torch.equal(tensor, wide_sd[key]), key


# --- the mixed-width hazard guard -------------------------------------------

def _shard_dir(tmp_path, name, width, rows=4):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "shard_0000.npz",
             options=np.zeros((rows, width), dtype=np.float32))
    return str(d)


def test_mixed_width_shards_are_refused(tmp_path):
    """A widening run on mixed-width shards must ABORT. The pad shim would
    zero-fill the narrow ones, asserting the appended features are 0 where
    they are not — which would quietly manufacture the null result the
    widening exists to test."""
    from rl.plan_iter import _refuse_mixed_option_widths
    dirs = [_shard_dir(tmp_path, "old", NARROW),
            _shard_dir(tmp_path, "new", WIDE)]
    with pytest.raises(SystemExit, match="MIXED option width"):
        _refuse_mixed_option_widths(dirs)


def test_uniform_width_shards_pass(tmp_path):
    from rl.plan_iter import _refuse_mixed_option_widths
    dirs = [_shard_dir(tmp_path, "a", WIDE), _shard_dir(tmp_path, "b", WIDE)]
    _refuse_mixed_option_widths(dirs)          # must not raise
