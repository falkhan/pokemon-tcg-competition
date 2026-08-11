"""M43 review findings #1/#3: the rl.ppo CLI must refuse inert shaping and
expose the Phi anchor.

Finding #3: `--shaping value` (or dev) without `--race-shaping > 0` used to
parse fine and run a full leg UNSHAPED — the collector builds a phi net only
when both are set (rl/collector.py gates on `race_shaping and shaping`).
A silently-unshaped pre-registered arm is the failure mode; the CLI now
hard-fails instead.

Finding #1: Phi was hardcoded to --start. The Phase-0 kill branch of
docs/M43-plan.md warm-starts the critic from a different checkpoint via
--value-ckpt; --phi-ckpt lets Phi follow it instead of silently staying on
--start.

The parser lives in rl/ppo.py's __main__ block, so the error paths are
exercised via subprocess (argparse p.error exits 2 before train() runs —
nothing heavy executes).
"""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "rl.ppo", *argv],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_value_shaping_without_coef_is_rejected():
    proc = _run_cli("--shaping", "value", "--iterations", "1")
    assert proc.returncode == 2
    assert "no-op without --race-shaping" in proc.stderr


def test_dev_shaping_without_coef_is_rejected():
    proc = _run_cli("--shaping", "dev", "--iterations", "1")
    assert proc.returncode == 2
    assert "no-op without --race-shaping" in proc.stderr


def test_phi_ckpt_requires_value_shaping():
    proc = _run_cli("--phi-ckpt", "m39_retain_b.pt", "--iterations", "1")
    assert proc.returncode == 2
    assert "--phi-ckpt only applies under --shaping value" in proc.stderr


def test_help_documents_phi_ckpt():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "--phi-ckpt" in proc.stdout


def test_train_accepts_phi_ckpt_kwarg():
    from rl.ppo import train
    assert "phi_ckpt" in inspect.signature(train).parameters
