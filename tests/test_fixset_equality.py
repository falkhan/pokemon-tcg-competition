"""M43 review finding #4: gate-vs-ship fix-set equality.

The gate measures a matchrunner kind token's fix package (e.g. `model-c-pkgz`
= conserve,racemode2,racemode4,planzero) while the shipped bundle reads the
PKM_ATTACH_FIXES default hardcoded in submission/main.py — two unlinked copies.
`tcg.shipping export --fixes` templates the gated package into the default;
`scripts/ship_verify.py --gate-arm` requires set-equality between the two.
These tests pin both halves without building a real bundle.
"""
import re

import pytest

pytest.importorskip("torch")

import tcg.shipping as shp
from scripts.ship_verify import gate_fix_package

SHIP_VERIFY_RE = r'"PKM_ATTACH_FIXES",\s*\n?\s*"([^"]*)"'

MAIN_PY_SNIPPET = (
    "_ATTACH_FIXES = frozenset(\n"
    "    f for f in os.environ.get(\n"
    '        "PKM_ATTACH_FIXES",\n'
    '        "conserve,planzero,ash,ashguard").split(",") if f)\n'
)


def test_set_bundle_fixes_rewrites_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(shp, "SUBMISSION", tmp_path)
    (tmp_path / "main.py").write_text(MAIN_PY_SNIPPET, encoding="utf-8")
    shp._set_bundle_fixes("conserve,racemode2,racemode4,planzero")
    src = (tmp_path / "main.py").read_text(encoding="utf-8")
    # ship_verify's own regex must see exactly the gated set
    m = re.search(SHIP_VERIFY_RE, src)
    assert m is not None
    assert set(m.group(1).split(",")) == {
        "conserve", "racemode2", "racemode4", "planzero"}
    assert "ashguard" not in src


def test_set_bundle_fixes_refuses_a_missing_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(shp, "SUBMISSION", tmp_path)
    (tmp_path / "main.py").write_text("print('no fix string here')\n",
                                      encoding="utf-8")
    with pytest.raises(shp.BundleError, match="anchor"):
        shp._set_bundle_fixes("conserve")


def test_set_bundle_fixes_empty_string_means_no_fixes(tmp_path, monkeypatch):
    monkeypatch.setattr(shp, "SUBMISSION", tmp_path)
    (tmp_path / "main.py").write_text(MAIN_PY_SNIPPET, encoding="utf-8")
    shp._set_bundle_fixes("")
    m = re.search(SHIP_VERIFY_RE,
                  (tmp_path / "main.py").read_text(encoding="utf-8"))
    assert m is not None and m.group(1) == ""


def test_gate_fix_package_resolves_the_m43_arm_kinds():
    assert gate_fix_package(
        "model-c-pkgz:checkpoints/x.pt:alakazam_v2_h4") == {
            "conserve", "racemode2", "racemode4", "planzero"}
    assert gate_fix_package("model-pz:checkpoints/y.pt:decks/ogerpon.csv") \
        == {"planzero"}


def test_gate_fix_package_bare_model_is_the_empty_set():
    assert gate_fix_package("model:checkpoints/x.pt:lucario") == frozenset()


def test_gate_fix_package_unknown_kind_is_none_not_empty():
    # an unknown kind must FAIL verification, never read as "no fixes"
    assert gate_fix_package("model-typo-kind:x.pt:lucario") is None


def test_export_signature_carries_fixes():
    import inspect
    assert "fixes" in inspect.signature(shp.export).parameters


def test_ship_verify_3b_success_path_formats(tmp_path, monkeypatch):
    """The 3b equality check's success/detail f-string executes end-to-end —
    caught live 2026-08-12: a NameError in the detail string survived unit
    tests that only covered gate_fix_package."""
    import subprocess
    import sys
    from pathlib import Path
    ROOT = Path(shp.__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "scripts/ship_verify.py",
         "--checkpoint", "definitely_missing.pt", "--deck", "alakazam_v2_h4",
         "--corpus", str(tmp_path),
         "--gate-arm", "model-c-pkgz:checkpoints/x.pt:alakazam_v2_h4"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    # the run FAILS (missing checkpoint/corpus) but must not crash: 3b's
    # verdict line prints with both sets formatted
    assert "3b. gate/ship fix-set equality" in proc.stdout
    assert "NameError" not in proc.stderr
