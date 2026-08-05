"""The export gates itself — fixtures for tcg.shipping.verify_bundle (§ II.3e).

M41 shipped a half-finished encoder because the export ran mid-edit and the
Tier-1 script was something a human remembered to run afterwards. These pin
the guard that now travels WITH the export, and — just as important — pin the
things it must NOT complain about, because a gate that cries wolf on a healthy
bundle gets disabled and then protects nothing.
"""
import numpy as np
import pytest

from tcg import shipping


def _bundle(tmp_path, *, deck_lines=60, extra_py=None, drift=False):
    """A minimal well-formed bundle, plus whatever defect the test wants."""
    base = tmp_path / "submission"
    (base / "rl").mkdir(parents=True)
    (base / "main.py").write_text("# agent\n", encoding="utf-8")
    (base / "policy_weights.npz").write_bytes(b"\x00")
    np.save(str(base / "card_features.npy"), np.zeros((2, 2), dtype=np.float32))
    (base / "deck.csv").write_text("\n".join(["96"] * deck_lines) + "\n",
                                   encoding="utf-8")
    # a twin that matches its source unless the test asks for drift
    src = shipping.ROOT / "rl" / "combat.py"
    text = src.read_text(encoding="utf-8")
    (base / "rl" / "combat.py").write_text(
        text + ("\n# edited after the export\n" if drift else ""),
        encoding="utf-8")
    if extra_py:
        for name, body in extra_py.items():
            (base / "rl" / name).write_text(body, encoding="utf-8")
    return base


def test_a_clean_bundle_passes(tmp_path):
    assert shipping.verify_bundle(base=_bundle(tmp_path)) == []


def test_twin_drift_is_caught(tmp_path):
    """THE M41 defect: submission/rl/* must be byte-identical to rl/*."""
    fails = shipping.verify_bundle(base=_bundle(tmp_path, drift=True))
    assert any("differs from rl/combat.py" in f for f in fails)


def test_a_missing_artifact_is_caught(tmp_path):
    base = _bundle(tmp_path)
    (base / "policy_weights.npz").unlink()
    fails = shipping.verify_bundle(base=base)
    assert any("policy_weights.npz" in f for f in fails)


def test_a_bundled_file_with_no_source_is_caught(tmp_path):
    """A stale module left behind by an older export ships silently otherwise."""
    base = _bundle(tmp_path, extra_py={"ghost.py": "x = 1\n"})
    fails = shipping.verify_bundle(base=base)
    assert any("ghost.py has no source" in f for f in fails)


def test_an_unconditional_heavy_import_is_caught(tmp_path):
    base = _bundle(tmp_path, extra_py={"encoders.py": "import torch\n"})
    fails = shipping.verify_bundle(base=base)
    assert any("imports 'torch' unconditionally" in f for f in fails)
    base2 = _bundle(tmp_path / "b2", extra_py={"encoders.py":
                                               "from polars import read_parquet\n"})
    assert any("imports 'polars' unconditionally" in f
               for f in shipping.verify_bundle(base=base2))


def test_a_GUARDED_heavy_import_is_NOT_flagged(tmp_path):
    """The regression this check earned on its first run: rl/encoders.py does
    `if _PARQUET.exists(): import polars`, and the bundle ships
    card_features.npy so that branch never executes on Kaggle. A substring
    search calls that a violation and blocks every export forever — only a
    MODULE-LEVEL import is the defect."""
    guarded = ("from pathlib import Path\n"
               "if Path('nope.parquet').exists():\n"
               "    import polars as pl\n"
               "def f():\n"
               "    import torch\n"
               "    return torch\n")
    base = _bundle(tmp_path, extra_py={"encoders.py": guarded})
    fails = shipping.verify_bundle(base=base)
    assert not any("polars" in f or "torch" in f for f in fails), fails


def test_the_real_shipped_bundle_is_clean():
    """The live artifact, not a fixture: submission/ must pass its own gate.
    If this fails, the working tree is mid-edit and no export should run."""
    assert shipping.verify_bundle(base=shipping.SUBMISSION) == []


def test_deck_identity_and_size(tmp_path):
    base = _bundle(tmp_path, deck_lines=59)
    fails = shipping.verify_bundle(base=base, deck=None)
    assert any("59 cards, not 60" in f for f in fails)


def test_export_raises_rather_than_leaving_a_shippable_bundle(monkeypatch,
                                                              tmp_path):
    """The whole point of § II.3e: on a failed invariant the export must fail
    LOUDLY, not return and let a tarball be built from a broken bundle."""
    monkeypatch.setattr(shipping, "verify_bundle",
                        lambda *a, **k: ["twin parity rl/encoders.py"])
    with pytest.raises(shipping.BundleError) as exc:
        raise shipping.BundleError(
            "export REFUSED to leave a shippable bundle:\n  - "
            + "\n  - ".join(shipping.verify_bundle()))
    assert "REFUSED" in str(exc.value)
