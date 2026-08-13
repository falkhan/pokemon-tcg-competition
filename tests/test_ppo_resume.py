"""M44 Step 3: pausable PPO (3a/3b/3c) + the loss-cause classifier (3g).

The state round-trip is exercised headless — `_save_train_state` /
`_load_train_state` touch only torch. The CLI guards run via subprocess
exactly like tests/test_ppo_cli.py (argparse p.error exits 2 before anything
heavy runs). The 3g classifier is pure-dict, no engine needed.
"""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from rl.collector import CAUSE_BUCKETS, _terminal_cause
from rl.policy import OptionScorer
from rl.ppo import _load_train_state, _save_train_state, train

ROOT = Path(__file__).resolve().parent.parent

META = {"start": "seed.pt", "iterations": 25, "learn_deck": "lucario",
        "opponents": ["mirror=0.5", "rule:iono=0.5"]}


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "rl.ppo", *argv],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def _stepped_model_and_opt():
    """A tiny OptionScorer + AdamW with one REAL step taken, so exp_avg /
    exp_avg_sq exist — an empty optimizer state would round-trip trivially."""
    torch.manual_seed(0)
    model = OptionScorer(hidden=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = sum((p ** 2).sum() for p in model.parameters())
    loss.backward()
    opt.step()
    return model, opt


# --- 3a/3b: state round-trip ------------------------------------------------

def test_save_load_round_trip(tmp_path):
    model, opt = _stepped_model_and_opt()
    path = tmp_path / "ppo_state_x.pt"
    _save_train_state(path, model, opt, next_it=2, best_wr=0.625, meta=META)
    assert not path.with_suffix(".pt.tmp").exists(), "atomic tmp not cleaned"

    state = _load_train_state(path, META)
    assert state["next_it"] == 2
    assert state["best_wr"] == 0.625
    assert state["meta"] == META

    for k, v in model.state_dict().items():
        assert torch.equal(state["model"][k], v.cpu()), f"model sd drifted: {k}"

    # restoring into a FRESH model+opt reproduces the optimizer state exactly
    torch.manual_seed(1)                       # different init on purpose
    model2 = OptionScorer(hidden=8)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    model2.load_state_dict(state["model"])
    opt2.load_state_dict(state["opt"])
    s1, s2 = opt.state_dict(), opt2.state_dict()
    assert s1["param_groups"] == s2["param_groups"]
    for pid, pstate in s1["state"].items():
        for k, v in pstate.items():
            got = s2["state"][pid][k]
            if isinstance(v, torch.Tensor):
                assert torch.equal(got, v), f"opt state drifted: {pid}/{k}"
            else:
                assert got == v


@pytest.mark.parametrize("key,bad", [("start", "other.pt"),
                                     ("iterations", 99),
                                     ("learn_deck", "ogerpon")])
def test_meta_mismatch_refused(tmp_path, key, bad):
    model, opt = _stepped_model_and_opt()
    path = tmp_path / "ppo_state_x.pt"
    _save_train_state(path, model, opt, next_it=1, best_wr=0.5, meta=META)
    with pytest.raises(ValueError, match=f"meta mismatch on '{key}'"):
        _load_train_state(path, {**META, key: bad})


def test_opponents_not_enforced(tmp_path):
    # the pool legitimately changes as league legs complete (docs/M44-plan.md
    # Step 5: subsequent legs pool the UPDATED net) — never a refusal
    model, opt = _stepped_model_and_opt()
    path = tmp_path / "ppo_state_x.pt"
    _save_train_state(path, model, opt, next_it=1, best_wr=0.5, meta=META)
    state = _load_train_state(path, {**META, "opponents": ["mirror=1.0"]})
    assert state["next_it"] == 1


def test_missing_state_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_train_state(tmp_path / "absent.pt", META)


# --- 3b CLI guards ----------------------------------------------------------

def test_resume_requires_tag():
    proc = _run_cli("--resume", "--iterations", "1")
    assert proc.returncode == 2
    assert "--resume requires --tag" in proc.stderr


def test_resume_missing_state_refused():
    proc = _run_cli("--resume", "--tag", "zz_no_such_state_zz",
                    "--iterations", "1")
    assert proc.returncode == 2
    assert "no state file" in proc.stderr


def test_help_documents_resume():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "--resume" in proc.stdout


def test_train_accepts_resume_kwarg():
    assert "resume" in inspect.signature(train).parameters


# --- 3g: loss-cause classifier ----------------------------------------------

def _terminal_obs(reason):
    return {"current": {"result": 0},
            "logs": [{"type": 5, "player": 0},
                     {"type": 23, "result": 0, "reason": reason}]}


@pytest.mark.parametrize("reason,cause", [(1, "prizes"), (2, "deckout"),
                                          (3, "benchout"), (4, "effect")])
def test_terminal_cause_by_engine_reason(reason, cause):
    assert _terminal_cause(_terminal_obs(reason)) == cause


def test_terminal_cause_missing_log_is_other():
    # obs.logs is per-seat (rl/memory.py) — the RESULT log can land in the
    # other seat's window; that is a bucket, not a guess
    assert _terminal_cause({"current": {"result": 1}, "logs": []}) == "other"
    assert _terminal_cause({"current": {"result": 1}}) == "other"
    assert _terminal_cause({"current": {"result": 1}, "logs": None}) == "other"


def test_cause_buckets_cover_classifier_range():
    for reason in (1, 2, 3, 4, None, 99):
        assert _terminal_cause(_terminal_obs(reason)) in CAUSE_BUCKETS
