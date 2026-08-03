"""G-14 (M40): the serve/train input-parity check in scripts/ship_verify.py.

Ship_verify's checks 1-5 verify that the artifact is INTERNALLY consistent —
twins match, deck md5 matches, weights match the checkpoint, rule names resolve,
rules fire. Every one of them passes on a bundle whose network is fed an input
distribution it never saw in training, which is the live state of Ship A and
Ship B and has been true since M24.

Two regressions of exactly that shape landed in M40 (S5 encoder-version drift,
S6 plan-head mismatch), so check 6 tests the general form. These unit tests
synthesize bundles and corpora rather than using real ones, so they pin the
DECISION BOUNDARY rather than today's accident.
"""
import numpy as np
import pytest

from rl.encoders import (EMBED_DIM, N_CONTEXTS, N_STATE_IDS_V3, N_STATE_IDS_V4,
                         OPTION_M28_DIM, STATE_V2_DIM, V4_EXTRA_DIM)
from rl.plan import PLAN_DIM
from scripts.ship_verify import input_parity_checks

V3_CTX = STATE_V2_DIM + N_CONTEXTS
V4_CTX = V3_CTX + V4_EXTRA_DIM
FIXES_LIVE = 'os.environ.get(\n        "PKM_ATTACH_FIXES",\n        "conserve,racemode2,racemode4")'
FIXES_PZ = 'os.environ.get(\n        "PKM_ATTACH_FIXES",\n        "conserve,racemode2,racemode4,planzero")'


def _bundle(tmp_path, v4=False, name="policy_weights.npz"):
    ctx = V4_CTX if v4 else V3_CTX
    n_ids = N_STATE_IDS_V4 if v4 else N_STATE_IDS_V3
    w_in = ctx + PLAN_DIM + n_ids * EMBED_DIM
    z = {
        "embedding.weight": np.zeros((10, EMBED_DIM), np.float32),
        "state_enc.0.weight": np.zeros((8, w_in), np.float32),
        "plan_enc.0.weight": np.zeros((8, PLAN_DIM), np.float32),
        "option_enc.0.weight": np.zeros(
            (8, OPTION_M28_DIM + 2 * EMBED_DIM), np.float32),
    }
    if v4:
        z["enc_ver"] = np.asarray(4.0, np.float32)
    p = tmp_path / name
    np.savez(p, **z)
    return p


def _corpus(tmp_path, name, *, v4=False, plan_frac=None, rows=20,
            pad_v4_rows=0):
    """A shard dir. plan_frac=None writes NO `plans` column (what every
    replay-derived corpus actually looks like); a float writes that fraction
    of non-zero plan rows."""
    d = tmp_path / name
    d.mkdir()
    ctx = V4_CTX if v4 else V3_CTX
    n_ids = N_STATE_IDS_V4 if v4 else N_STATE_IDS_V3
    states = np.zeros((rows, ctx), np.float32)
    states[:, :V3_CTX] = 1.0
    if v4:
        # OppMemory.features writes v[3 + hot] = 1.0 with hot in 0..6, so the
        # attach-target one-hot at v4-block offsets 3..9 SUMS to 1 — which
        # slot is hot varies by row. Vary it here: a fixture that always sets
        # offset 3 would agree with a check that only tests offset 3, which is
        # exactly the bug real data caught.
        states[:, V3_CTX:] = 1.0
        states[:, V3_CTX + 3:V3_CTX + 10] = 0.0
        for r in range(rows):
            states[r, V3_CTX + 3 + (r % 7)] = 1.0
        if pad_v4_rows:
            states[:pad_v4_rows, V3_CTX:] = 0.0     # the zero-pad masquerade
    cols = dict(states=states,
                state_ids=np.zeros((rows, n_ids), np.int32),
                options=np.zeros((rows, OPTION_M28_DIM), np.float32),
                labels=np.zeros(rows, np.int32))
    if plan_frac is not None:
        plans = np.zeros((rows, PLAN_DIM), np.float32)
        plans[:int(rows * plan_frac), 0] = 1.0
        cols["plans"] = plans
    np.savez(d / "shard_0000.npz", **cols)
    return d


def _result(checks, prefix):
    for label, ok, detail in checks:
        if label.startswith(prefix):
            return ok, detail
    raise AssertionError(f"no check {prefix} in {[c[0] for c in checks]}")


# --- 6a: the S6 defect ------------------------------------------------------

def test_serving_a_plan_against_a_planless_corpus_fails(tmp_path):
    """The live state of Ship A and Ship B. This is the check that did not
    exist from M24 to M40."""
    checks = input_parity_checks(_bundle(tmp_path), FIXES_LIVE,
                                 [_corpus(tmp_path, "planless")])
    ok, detail = _result(checks, "6a")
    assert not ok
    assert "OUT OF DISTRIBUTION" in detail and "planzero" in detail


def test_planzero_against_a_planless_corpus_passes(tmp_path):
    checks = input_parity_checks(_bundle(tmp_path), FIXES_PZ,
                                 [_corpus(tmp_path, "planless")])
    assert _result(checks, "6a")[0]


def test_serving_a_plan_against_a_plan_carrying_corpus_passes(tmp_path):
    # 0.16 is the measured occupancy of a real plan-carrying corpus; the null
    # plan is a legitimate frequent choice, so the bar is not 1.0.
    checks = input_parity_checks(
        _bundle(tmp_path), FIXES_LIVE,
        [_corpus(tmp_path, "planful", plan_frac=0.16, rows=100)])
    assert _result(checks, "6a")[0]


def test_planzero_against_a_plan_carrying_corpus_fails(tmp_path):
    # The mirror-image mistake: zeroing the plan when the corpus DID carry
    # plans throws away a trained input. Both directions must be caught.
    checks = input_parity_checks(
        _bundle(tmp_path), FIXES_PZ,
        [_corpus(tmp_path, "planful", plan_frac=0.16, rows=100)])
    assert not _result(checks, "6a")[0]


# --- 6b: the S5 defect ------------------------------------------------------

def test_v4_bundle_against_v3_corpus_fails_on_width(tmp_path):
    """The mismatch BCDatasetV3's pad shim hides: it zero-pads narrow states
    to the widest shard, so this loads and trains silently."""
    checks = input_parity_checks(_bundle(tmp_path, v4=True), FIXES_LIVE,
                                 [_corpus(tmp_path, "v3corpus")])
    ok, detail = _result(checks, "6b")
    assert not ok
    assert "zero-pads" in detail


def test_v4_bundle_against_v4_corpus_passes_on_width(tmp_path):
    checks = input_parity_checks(_bundle(tmp_path, v4=True), FIXES_LIVE,
                                 [_corpus(tmp_path, "v4corpus", v4=True)])
    assert _result(checks, "6b")[0]


def test_mixed_width_dirs_are_reported_per_dir_not_pooled(tmp_path):
    # Pooling is what hides the mismatch at train time; the check must not
    # repeat the mistake. One good dir + one bad dir = FAIL.
    checks = input_parity_checks(
        _bundle(tmp_path, v4=True), FIXES_LIVE,
        [_corpus(tmp_path, "good", v4=True), _corpus(tmp_path, "bad")])
    ok, detail = _result(checks, "6b")
    assert not ok and "bad" in detail and "good" not in detail.split("MISMATCH")[1]


# --- 6d: zero-padded rows masquerading as v4 --------------------------------

def test_zero_padded_v4_rows_are_detected(tmp_path):
    """A v4-width corpus can still contain rows whose 341-dim block is all
    zeros — a vector encode_ctx_v4 can never emit, because the attach-target
    one-hot always has exactly one bit set. In a retention mix those rows are
    the champion half, so the block becomes a corpus-identity feature that is
    always 'new corpus' at serve time."""
    checks = input_parity_checks(
        _bundle(tmp_path, v4=True), FIXES_LIVE,
        [_corpus(tmp_path, "padded", v4=True, rows=20, pad_v4_rows=7)])
    ok, detail = _result(checks, "6d")
    assert not ok and ":7" in detail


def test_clean_v4_corpus_passes_the_integrity_check(tmp_path):
    checks = input_parity_checks(_bundle(tmp_path, v4=True), FIXES_LIVE,
                                 [_corpus(tmp_path, "clean", v4=True)])
    assert _result(checks, "6d")[0]


def test_v3_bundle_skips_the_v4_integrity_check(tmp_path):
    checks = input_parity_checks(_bundle(tmp_path), FIXES_LIVE,
                                 [_corpus(tmp_path, "c")])
    assert not any(label.startswith("6d") for label, _, _ in checks)


# --- plumbing ---------------------------------------------------------------

def test_missing_corpus_is_a_failure_not_a_skip(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    checks = input_parity_checks(_bundle(tmp_path), FIXES_LIVE, [empty])
    assert any(not ok for _, ok, _ in checks)


def test_serve_fix_prefix_is_discoverable_by_the_fix_name_check():
    # Check 3 resolves fix names reflectively out of rl.plan by prefix. A new
    # prefix class that it does not know reports the token as a typo and the
    # ship gate blocks on a correct config.
    import rl.plan as rp
    known = {v for k, v in vars(rp).items()
             if k.startswith(("PLAY_FIX_", "ATTACH_FIX_", "SERVE_FIX_"))
             and isinstance(v, str)}
    assert rp.SERVE_FIX_PLANZERO in known


@pytest.mark.parametrize("frac,serves,expect", [
    (0.00, True, False), (0.04, True, False), (0.06, True, True),
    (0.00, False, True), (0.02, False, False),
])
def test_plan_thresholds_are_the_documented_boundary(tmp_path, frac, serves,
                                                     expect):
    d = _corpus(tmp_path, f"c{frac}{serves}", plan_frac=frac, rows=100)
    checks = input_parity_checks(_bundle(tmp_path),
                                 FIXES_LIVE if serves else FIXES_PZ, [d])
    assert _result(checks, "6a")[0] is expect
