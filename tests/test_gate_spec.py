"""Golden fixtures for scripts/gate_spec.py (M41b § II.3d).

The refusals are the feature, so they are what these pin: a spec edited after
the run, a short cell, a missing cell, and an unstamped legacy battery must
all fail to produce a verdict. A gate runner that emits a number anyway is
worse than the hand-rolled decoders it replaces, because it looks principled.
"""
import json

import pytest

import scripts.gate_spec as gs

SPEC = {
    "name": "t_gate",
    "question": "does the arm beat the control?",
    "arm": "model:checkpoints/arm.pt:deck",
    "control": "model:checkpoints/ctl.pt:deck",
    "beds": [{"name": "grim", "spec": "model:checkpoints/bed.pt:grim"}],
    "n_per_cell": 4,
    "seed": 1,
    "bars": {"pass": 0.02, "kill": 0.0},
}


def _write_spec(tmp_path, **over):
    spec = {**SPEC, **over}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def _write_cell(out, side, bed, results, digest, n_declared=None):
    """A battery checkpoint as `run_pairs` writes it, stamped like `run` does."""
    out.mkdir(parents=True, exist_ok=True)
    header = {"pairs": [], "workers": 8, "seed": 1}
    if digest is not None:
        header["extra"] = {"gate": "t_gate", "spec_hash": digest}
    lines = [json.dumps(header),
             json.dumps({"job": 0, "pair": 0, "results": results,
                         "margins": [None] * len(results)})]
    gs.cell_path(out, side, bed).write_text("\n".join(lines) + "\n")


# --- the hash ---------------------------------------------------------------

def test_hash_ignores_formatting_but_not_the_bar(tmp_path):
    """Reformatting a spec is not tampering; moving a bar is."""
    base = gs.spec_hash(SPEC)
    assert gs.spec_hash({**SPEC, "question": "reworded, same decision"}) == base
    assert gs.spec_hash({**SPEC, "bars": {"pass": 0.01, "kill": 0.0}}) != base
    assert gs.spec_hash({**SPEC, "n_per_cell": 8}) != base
    assert gs.spec_hash({**SPEC, "arm": "model:other.pt:deck"}) != base
    assert gs.spec_hash({**SPEC, "seed": 2}) != base


def test_a_spec_must_pre_register_both_bars(tmp_path):
    path, _ = _write_spec(tmp_path, bars={"pass": 0.02})
    with pytest.raises(SystemExit):
        gs.load_spec(path)


def test_a_kill_bar_above_the_pass_bar_is_unreadable(tmp_path):
    path, _ = _write_spec(tmp_path, bars={"pass": 0.0, "kill": 0.05})
    with pytest.raises(SystemExit):
        gs.load_spec(path)


# --- the refusals -----------------------------------------------------------

def test_decode_refuses_a_spec_edited_after_the_run(tmp_path, capsys):
    """THE property § II.3d exists for. The battery is stamped with the
    original hash; the bar is then lowered so a losing arm would pass."""
    out = tmp_path / "run"
    original = gs.spec_hash(SPEC)
    _write_cell(out, "arm", "grim", [0, 0, 1, 1], original)
    _write_cell(out, "control", "grim", [0, 1, 1, 1], original)

    moved, _ = _write_spec(tmp_path, bars={"pass": -0.99, "kill": -1.0})
    rc = gs.cmd_decode(moved, out)
    text = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in text and "changed after the run" in text
    assert "VERDICT" not in text          # no number is emitted at all


def test_decode_refuses_a_cell_short_of_the_pre_registered_n(tmp_path, capsys):
    """Stopping a battery early once it looks good is the same defect wearing
    a different hat."""
    out = tmp_path / "run"
    path, spec = _write_spec(tmp_path)
    digest = gs.spec_hash(spec)
    _write_cell(out, "arm", "grim", [0, 0], digest)          # 2 of 4
    _write_cell(out, "control", "grim", [0, 1, 1, 1], digest)
    assert gs.cmd_decode(path, out) == 2
    assert "pre-registered 4" in capsys.readouterr().out


def test_decode_refuses_a_missing_cell(tmp_path, capsys):
    out = tmp_path / "run"
    path, spec = _write_spec(tmp_path)
    _write_cell(out, "arm", "grim", [0, 0, 1, 1], gs.spec_hash(spec))
    assert gs.cmd_decode(path, out) == 2
    assert "missing cell" in capsys.readouterr().out


def test_decode_refuses_an_unstamped_legacy_battery(tmp_path, capsys):
    """A pre-M41b checkpoint carries no spec hash; it cannot be retrofitted
    into a pre-registered gate after the fact."""
    out = tmp_path / "run"
    path, spec = _write_spec(tmp_path)
    _write_cell(out, "arm", "grim", [0, 0, 1, 1], None)
    _write_cell(out, "control", "grim", [0, 1, 1, 1], gs.spec_hash(spec))
    assert gs.cmd_decode(path, out) == 2
    assert "ran under spec None" in capsys.readouterr().out


# --- the verdicts -----------------------------------------------------------

def _clean_run(tmp_path, arm, control, **over):
    out = tmp_path / "run"
    path, spec = _write_spec(tmp_path, **over)
    digest = gs.spec_hash(spec)
    _write_cell(out, "arm", "grim", arm, digest)
    _write_cell(out, "control", "grim", control, digest)
    return path, out


def test_a_matching_run_decodes_to_pass(tmp_path, capsys):
    # arm 4/4, control 0/4 -> delta +1.0, far above the pass bar
    path, out = _clean_run(tmp_path, [0, 0, 0, 0], [1, 1, 1, 1])
    assert gs.cmd_decode(path, out) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_a_matching_run_decodes_to_kill(tmp_path, capsys):
    path, out = _clean_run(tmp_path, [1, 1, 1, 1], [0, 0, 0, 0])
    assert gs.cmd_decode(path, out) == 1
    assert "VERDICT: KILL" in capsys.readouterr().out


def test_between_the_bars_is_inconclusive_not_a_rounded_up_pass(tmp_path,
                                                                capsys):
    """delta lands strictly between kill and pass: the pre-registered outcome
    is 'we do not know'."""
    path, out = _clean_run(tmp_path, [0, 0, 0, 1], [0, 0, 1, 1],
                           bars={"pass": 0.5, "kill": 0.0})
    assert gs.cmd_decode(path, out) == 1
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_draws_count_half_matching_the_matchrunner_decode_law(tmp_path):
    assert gs.wr([0, 1]) == 0.5
    assert gs.wr([2, 2]) == 0.5
    assert gs.wr([0, 2, 1, 1]) == pytest.approx(0.375)


def test_z_is_zero_when_the_arms_agree():
    assert gs.two_proportion_z(0.5, 100, 0.5, 100) == 0.0
    assert gs.two_proportion_z(0.6, 500, 0.5, 500) > 2.0
