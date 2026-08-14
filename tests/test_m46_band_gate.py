"""Golden fixtures for scripts/m46_band_gate.py (M46 A0).

What these pin, in order of importance: (1) the tamper chain — editing the
WEIGHTS after cells ran must produce a refusal, because the band hash rides
inside the derived spec name that gate_spec stamps into every cell; (2) the
share-weighted delta/SE arithmetic against a hand computation (the copy of
m40_decide.py:248-300 must stay numerically identical to the frozen
original); (3) a family kill overruling a pooled pass; (4) the
weight-bootstrap PASS->INCONCLUSIVE downgrade; (5) advisory beds being
structurally outside the verdict; (6) the committed m46 specs themselves.
"""
import json

import pytest

pytest.importorskip("numpy")

import scripts.gate_spec as gs
import scripts.m46_band_gate as bg

BAND = {
    "name": "t_band",
    "question": "test band",
    "arm": "model:checkpoints/arm.pt:deck",
    "control": "model:checkpoints/ctl.pt:deck",
    "beds": [
        {"name": "x_d1", "spec": "model:checkpoints/x1.pt:deck", "family": "x"},
        {"name": "x_d2", "spec": "model:checkpoints/x2.pt:deck", "family": "x"},
        {"name": "y_d1", "spec": "model:checkpoints/y1.pt:deck", "family": "y"},
        {"name": "anchor", "spec": "rule:iono", "family": None,
         "advisory": True},
    ],
    "n_per_cell": 4,
    "seeds": [1],
    "weights": {"n_games": 10, "source": "test", "games": {"x": 7, "y": 3}},
    "bars": {"pass": 0.02, "kill": 0.0, "family_z": -2.0},
    "bootstrap": {"B": 200, "seed": 7, "stability_floor": 0.95},
    "coverage_floor": 0.80,
}

WIN, LOSS, DRAW = [0] * 4, [1] * 4, [2] * 4
SPLIT = [0, 0, 1, 1]


def _write_band(tmp_path, **over):
    spec = {**BAND, **over}
    path = tmp_path / "band.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def _write_cells(out, spec, results_by_bed):
    """Cells for every seed x side x bed, stamped as `run` stamps them.
    results_by_bed: {bed: (arm_results, control_results)}."""
    for seed in spec["seeds"]:
        digest = gs.spec_hash(bg.derive_spec(spec, seed))
        sdir = bg.seed_dir(out, seed)
        sdir.mkdir(parents=True, exist_ok=True)
        for side_i, side in enumerate(("arm", "control")):
            for bed in spec["beds"]:
                res = results_by_bed[bed["name"]][side_i]
                header = {"pairs": [], "workers": 8, "seed": seed,
                          "extra": {"gate": spec["name"],
                                    "spec_hash": digest}}
                lines = [json.dumps(header),
                         json.dumps({"job": 0, "pair": 0, "results": res})]
                gs.cell_path(sdir, side, bed["name"]).write_text(
                    "\n".join(lines), encoding="utf-8")


ALL_EVEN = {"x_d1": (SPLIT, SPLIT), "x_d2": (SPLIT, SPLIT),
            "y_d1": (SPLIT, SPLIT), "anchor": (SPLIT, SPLIT)}


# --- hash discipline ---------------------------------------------------------

def test_band_hash_ignores_prose_and_covers_decisions():
    base = bg.band_hash(BAND)
    assert bg.band_hash({**BAND, "question": "reworded"}) == base
    assert bg.band_hash({**BAND, "note": "added"}) == base
    for tamper in (
            {"weights": {**BAND["weights"],
                         "games": {"x": 6, "y": 4}}},
            {"seeds": [1, 2]},
            {"bars": {**BAND["bars"], "family_z": -2.5}},
            {"bootstrap": {**BAND["bootstrap"], "stability_floor": 0.9}},
            {"acceptance": {"expect": "KILL", "max_z": -1.96}},
            {"null_calibration": True},
            {"beds": BAND["beds"][:3]}):
        assert bg.band_hash({**BAND, **tamper}) != base, tamper


def test_weights_edit_after_run_is_a_refusal(tmp_path, capsys):
    """THE chain property: cells stamped under the original weights refuse to
    decode once weights.games changes, because the band hash is inside the
    derived spec name that gate_spec hashed and stamped."""
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, ALL_EVEN)
    assert bg.cmd_decode(path, out) != 2          # sanity: decodes as stamped
    path, _ = _write_band(
        tmp_path, weights={**BAND["weights"], "games": {"x": 6, "y": 4}})
    capsys.readouterr()
    assert bg.cmd_decode(path, out) == 2
    text = capsys.readouterr().out
    assert "REFUSED" in text and "VERDICT" not in text


def test_refusals_missing_short_foreign(tmp_path, capsys):
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, ALL_EVEN)
    gs.cell_path(bg.seed_dir(out, 1), "arm", "y_d1").unlink()
    assert bg.cmd_decode(path, out) == 2
    _write_cells(out, spec, ALL_EVEN)
    short = {**ALL_EVEN, "x_d1": ([0, 1], SPLIT)}
    _write_cells(out, spec, short)
    assert bg.cmd_decode(path, out) == 2


# --- the weighted read -------------------------------------------------------

def test_weighted_math_matches_hand_computation(tmp_path, capsys):
    """x: two draws pooling to fa=0.75 fc=0.25; y: fa=0.25 fc=0.75.
    shares .7/.3 -> delta = .7*.5 + .3*(-.5) = +0.2; var per the m40 law."""
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, {
        "x_d1": (WIN, LOSS), "x_d2": (SPLIT, SPLIT),
        "y_d1": ([0, 1, 1, 1], [0, 0, 0, 1]), "anchor": (SPLIT, SPLIT)})
    bg.cmd_decode(path, out)
    text = capsys.readouterr().out
    fa_x, fc_x, n_x = 0.75, 0.25, 8
    fa_y, fc_y, n_y = 0.25, 0.75, 4
    delta = 0.7 * (fa_x - fc_x) + 0.3 * (fa_y - fc_y)
    var = (0.7 ** 2 * (fa_x * (1 - fa_x) / n_x + fc_x * (1 - fc_x) / n_x)
           + 0.3 ** 2 * (fa_y * (1 - fa_y) / n_y + fc_y * (1 - fc_y) / n_y))
    got_delta, got_se, _, _ = bg._weighted_delta(
        bg._family_table(spec, bg._collect(spec, out)[0])[0],
        {"x": 0.7, "y": 0.3})
    assert abs(got_delta - delta) < 1e-9
    assert abs(got_se - var ** 0.5) < 1e-9
    assert f"delta {delta:+.4f}" in text


def test_draws_count_half_and_draw_pooling(tmp_path):
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, {
        "x_d1": ([0, 2, 1, 1], SPLIT),        # wr 0.375, the decode law cell
        "x_d2": (WIN, SPLIT),
        "y_d1": (SPLIT, SPLIT), "anchor": (SPLIT, SPLIT)})
    fams, _ = bg._family_table(spec, bg._collect(spec, out)[0])
    fa, na, _, _ = bg._family_rates(fams["x"])
    assert na == 8 and abs(fa - (0.375 + 1.0) / 2) < 1e-9


def test_family_kill_overrules_pooled_pass(tmp_path, capsys):
    """x hugely positive carries the pool; y at z=-2.83 must kill anyway."""
    path, spec = _write_band(tmp_path, seeds=[1, 2])
    out = tmp_path / "out"
    _write_cells(out, spec, {
        "x_d1": (WIN, LOSS), "x_d2": (WIN, LOSS),
        "y_d1": (LOSS, WIN), "anchor": (SPLIT, SPLIT)})
    rc = bg.cmd_decode(path, out)
    text = capsys.readouterr().out
    assert rc == 1
    assert "FAMILY-KILL: y" in text and "VERDICT: KILL" in text


def test_delta_bars_and_exit_codes(tmp_path, capsys):
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, {                      # clean pass, both positive
        "x_d1": (WIN, SPLIT), "x_d2": (WIN, SPLIT),
        "y_d1": (WIN, SPLIT), "anchor": (SPLIT, SPLIT)})
    assert bg.cmd_decode(path, out) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out
    _write_cells(out, spec, {                      # uniform negative -> kill
        "x_d1": (SPLIT, WIN), "x_d2": (SPLIT, WIN),
        "y_d1": (SPLIT, WIN), "anchor": (SPLIT, SPLIT)})
    assert bg.cmd_decode(path, out) == 1
    assert "VERDICT: KILL" in capsys.readouterr().out


def test_weight_unstable_pass_downgrades(tmp_path, capsys):
    """Point delta +0.2 passes, but with n_games=10 the multinomial redraw
    flips the class often enough to breach the 0.95 floor."""
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    _write_cells(out, spec, {
        "x_d1": (WIN, SPLIT), "x_d2": (WIN, SPLIT),
        "y_d1": (LOSS, SPLIT), "anchor": (SPLIT, SPLIT)})
    rc = bg.cmd_decode(path, out)
    text = capsys.readouterr().out
    assert "WEIGHT-UNSTABLE" in text and "VERDICT: INCONCLUSIVE" in text
    assert rc == 1
    bg.cmd_decode(path, out)                       # deterministic under seed
    assert (capsys.readouterr().out.count("verdict agreement")
            and "WEIGHT-UNSTABLE" in text)


def test_advisory_isolation(tmp_path, capsys):
    path, spec = _write_band(tmp_path)
    out = tmp_path / "out"
    for anchor in ((WIN, LOSS), (LOSS, WIN)):
        _write_cells(out, spec, {**ALL_EVEN, "anchor": anchor})
        bg.cmd_decode(path, out)
        text = capsys.readouterr().out
        assert "delta +0.0000 +/-" in text        # weighted pool unmoved
        assert "ADVISORY" in text


# --- modes -------------------------------------------------------------------

def test_null_mode(tmp_path, capsys):
    path, spec = _write_band(tmp_path, null_calibration=True)
    out = tmp_path / "out"
    _write_cells(out, spec, ALL_EVEN)              # quiet null
    assert bg.cmd_decode(path, out) == 0
    text = capsys.readouterr().out
    assert "NULL CALIBRATION" in text
    assert "recommended calibrated family_z" in text
    assert "\nVERDICT:" not in text
    _write_cells(out, spec, {**ALL_EVEN,           # y fires at z=-2.83
                             "y_d1": (LOSS, WIN)})
    assert bg.cmd_decode(path, out) == 1


def test_acceptance_mode(tmp_path, capsys):
    acc = {"expect": "KILL", "max_z": -1.96, "reason": "live truth"}
    path, spec = _write_band(tmp_path, seeds=[1, 2], acceptance=acc)
    out = tmp_path / "out"
    _write_cells(out, spec, {                      # strong negative, non-
        "x_d1": (SPLIT, WIN), "x_d2": (SPLIT, WIN),  # degenerate rates so
        "y_d1": (SPLIT, WIN), "anchor": (SPLIT, SPLIT)})  # the SE is real
    assert bg.cmd_decode(path, out) == 0
    assert "ACCEPTANCE: MET" in capsys.readouterr().out
    _write_cells(out, spec, {                      # arm wins: invalid
        "x_d1": (WIN, LOSS), "x_d2": (WIN, LOSS),
        "y_d1": (WIN, LOSS), "anchor": (SPLIT, SPLIT)})
    assert bg.cmd_decode(path, out) == 1
    assert "INSTRUMENT INVALID" in capsys.readouterr().out


# --- validator + the committed specs ----------------------------------------

def test_validator_refuses_inconsistencies(tmp_path):
    for tamper, msg in (
            ({"weights": {**BAND["weights"], "games": {"x": 7, "y": 2}}},
             "n_games"),
            ({"bars": {**BAND["bars"], "family_z": 2.0}}, "negative"),
            ({"seeds": []}, "seeds"),
            ({"beds": BAND["beds"][:1]}, "no bed"),
            ({"beds": [{**BAND["beds"][3], "family": "x"}] + BAND["beds"][:3]},
             "advisory")):
        path, _ = _write_band(tmp_path, **tamper)
        with pytest.raises(SystemExit, match=msg):
            bg.load_band_spec(path)


def test_committed_m46_specs_validate():
    from pathlib import Path
    specs_dir = Path(bg.ROOT) / "docs" / "specs"
    posted = sorted(specs_dir.glob("m46_*.json"))
    assert posted, "the committed m46 band specs are missing"
    postmortem_shares = {"grim": 0.386, "mirror": 0.229, "wall": 0.060,
                         "lucario": 0.060, "stall": 0.048, "dragapult": 0.036}
    for path in posted:
        spec = bg.load_band_spec(path)             # full validator pass
        w = spec["weights"]
        assert w["n_games"] == 83
        assert sum(w["games"].values()) == 83
        for fam, share in postmortem_shares.items():
            assert abs(w["games"][fam] / 83 - share) < 5e-3, (path, fam)
        advisory = {b["name"] for b in spec["beds"]
                    if b.get("advisory", False)}
        assert advisory == {"tuned", "iono"}
