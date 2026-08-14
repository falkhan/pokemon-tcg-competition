"""M44 league ranking script: BT fit correctness, cache identity, refusals."""
import json
import math
from pathlib import Path

import pytest

import scripts.m44_league as lg

ROOT = Path(__file__).resolve().parent.parent


def _expected_grid(true_r: list[float], n_per_cell: float):
    """W from the EXACT Bradley-Terry rates of constructed ratings — the MLE
    must then recover the constructed ratings (real-valued wins are legal in
    the MM iteration)."""
    P = len(true_r)
    pi = [10 ** (r / 400.0) for r in true_r]
    w = [[0.0] * P for _ in range(P)]
    n = [[0.0] * P for _ in range(P)]
    for i in range(P):
        for j in range(P):
            if i != j:
                n[i][j] = n_per_cell
                w[i][j] = n_per_cell * pi[i] / (pi[i] + pi[j])
    return w, n


def test_bt_recovers_constructed_order_and_pins_anchor():
    true_r = [1000.0, 1120.0, 880.0]
    w, n = _expected_grid(true_r, 400)
    r, se = lg.bt_fit(w, n, pin_idx=0, pin=1000.0)
    assert r[0] == pytest.approx(1000.0, abs=1e-9)      # pin is EXACT
    for got, want in zip(r, true_r):
        assert got == pytest.approx(want, abs=1.0)
    assert sorted(range(3), key=lambda i: -r[i]) == [1, 0, 2]
    assert all(s > 0 and math.isfinite(s) for s in se)


def test_bt_draws_count_half_all_draws_is_flat():
    # every game a draw -> W[i][j] = n/2 everywhere -> all ratings = the pin
    P, n_games = 4, 200
    w = [[n_games / 2 if i != j else 0.0 for j in range(P)] for i in range(P)]
    n = [[float(n_games) if i != j else 0.0 for j in range(P)] for i in range(P)]
    r, _ = lg.bt_fit(w, n, pin_idx=0, pin=1000.0)
    for x in r:
        assert x == pytest.approx(1000.0, abs=1e-6)


def test_bt_degenerate_record_is_refused():
    w, n = _expected_grid([1000.0, 1100.0], 100)
    w[0] = [0.0, 0.0]                          # player 0 never won anything
    with pytest.raises(SystemExit, match="degenerate"):
        lg.bt_fit(w, n, pin_idx=0, pin=1000.0)


PA = {"id": "K", "md5": "aaa111", "deck": "alakazam_v2_h4"}
PB = {"id": "O", "md5": "bbb222", "deck": "ogerpon"}


def test_cell_name_is_order_invariant():
    assert lg.cell_name(PA, PB, 200, 0) == lg.cell_name(PB, PA, 200, 0)
    assert lg.pair_digest(PA, PB) == lg.pair_digest(PB, PA)
    assert lg.cell_extra(PA, PB) == lg.cell_extra(PB, PA)
    # digest tracks identity: a different net md5 = a different cell
    assert (lg.pair_digest({**PA, "md5": "changed"}, PB)
            != lg.pair_digest(PA, PB))


def test_read_pair_cells_pools_matching_and_refuses_mismatch(tmp_path):
    name = lg.cell_name(PA, PB, 4, 0)
    header = {"pairs": [], "workers": 2, "seed": 0,
              "extra": lg.cell_extra(PA, PB)}
    rows = [{"job": 0, "pair": 0, "results": [0, 1], "reasons": [1, 2]},
            {"job": 1, "pair": 0, "results": [2, 0], "reasons": [None, 3]}]
    (tmp_path / name).write_text(
        "\n".join(json.dumps(x) for x in [header, *rows]) + "\n")
    # a second cell (top-up) pools in
    name2 = lg.cell_name(PA, PB, 2, 1)
    (tmp_path / name2).write_text(
        "\n".join(json.dumps(x) for x in
                  [header, {"job": 0, "pair": 0, "results": [1],
                            "reasons": [4]}]) + "\n")
    results, reasons = lg.read_pair_cells(tmp_path, PA, PB)
    # pooling order across cell files is not part of the contract — the
    # (result, reason) PAIRING is
    assert sorted(zip(results, reasons), key=str) == sorted(
        [(0, 1), (1, 2), (2, None), (0, 3), (1, 4)], key=str)

    # tampered header md5 -> loud refusal naming the file
    bad = dict(header, extra={**lg.cell_extra(PA, PB), "a_md5": "evil"})
    (tmp_path / name).write_text(
        "\n".join(json.dumps(x) for x in [bad, *rows]) + "\n")
    with pytest.raises(SystemExit, match=name.replace(".", r"\.")):
        lg.read_pair_cells(tmp_path, PA, PB)


def test_roster_on_disk_validates():
    # the real spec file: specs parse, md5 pins match the checkpoints on disk
    players = lg.load_roster(ROOT / "docs/specs/m44_roster.json")
    ids = {p["id"] for p in players}
    assert {"K", "O", "G", "tuned", "iono"} <= ids
    assert sum(1 for p in players if "pin" in p) == 1
