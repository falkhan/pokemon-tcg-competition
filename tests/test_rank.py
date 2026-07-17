"""M12 rl/rank.py: pair math, dataset/collate, spec parsing."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import rl.matchrunner as mr
import rl.rank as rk
from rl.encoders import N_CONTEXTS, N_STATE_IDS, OPTION_DIM, STATE_V2_DIM


def test_pair_stats_margin_and_direction():
    logits = torch.tensor([[3.0, 1.0, 2.0]])
    scores = torch.tensor([[1000.0, 100.0, float("nan")]])
    # one valid ordered pair (0 over 1: gap 900 >= 200); logit agrees
    loss, correct, total = rk._pair_stats(logits, scores, rk.RANK_MARGIN)
    assert (correct, total) == (1, 1) and loss is not None
    # reversed logits: same pair counted, now wrong
    loss2, correct2, total2 = rk._pair_stats(
        torch.tensor([[1.0, 3.0, 2.0]]), scores, rk.RANK_MARGIN)
    assert (correct2, total2) == (0, 1)
    assert loss2.item() > loss.item()
    # sub-margin gap: no pairs
    _, _, t3 = rk._pair_stats(logits, torch.tensor([[100.0, 50.0, 60.0]]),
                              rk.RANK_MARGIN)
    assert t3 == 0


def _write_rank_shard(path, rng, n=3):
    menus = [3, 2, 4][:n]
    np.savez_compressed(
        path,
        states=rng.standard_normal(
            (n, STATE_V2_DIM + N_CONTEXTS)).astype(np.float32),
        state_ids=rng.integers(0, 1268, (n, N_STATE_IDS)).astype(np.int32),
        options=rng.standard_normal(
            (sum(menus), OPTION_DIM)).astype(np.float32),
        option_ids=rng.integers(0, 1268, (sum(menus), 2)).astype(np.int32),
        n_options=np.array(menus, dtype=np.int32),
        sib_scores=np.concatenate([
            np.array([1000.0, 100.0, np.nan], np.float32),
            np.array([np.nan, 500.0], np.float32),
            np.array([2000.0, 100.0, np.nan, 700.0], np.float32)]),
        game_ids=np.array([0, 0, 1], dtype=np.int32),
        results=np.ones(n, dtype=np.float32),
        deck_idx=np.zeros(n, dtype=np.int32),
    )


def test_rank_dataset_and_collate(tmp_path):
    rng = np.random.default_rng(7)
    d = tmp_path / "rank"
    d.mkdir()
    _write_rank_shard(d / "shard_w00_0000.npz", rng)
    ds = rk.RankDataset(d)
    assert len(ds) == 3
    sc, sids, menu, oids, sib = ds[2]
    assert menu.shape[0] == 4 and np.isnan(sib[2])
    states, sids_t, options, oids_t, scores = rk.collate_rank(
        [ds[i] for i in range(3)])
    assert options.shape == (3, 4, OPTION_DIM)
    assert torch.isnan(scores[0, 3])          # padding is NaN (never a pair)
    assert scores[2, 0] == 2000.0


def test_parse_spec_rank_kind():
    assert mr.parse_spec("rank:checkpoints/osv3_rank1.pt:lucario") == \
        ("rank", "checkpoints/osv3_rank1.pt", "lucario")
    assert mr.spec_deck(("rank", "x.pt", "lucario")) == "lucario"


def test_parse_spec_vsolver_kind():
    assert mr.parse_spec("vsolver:checkpoints/osv3_setupval1.pt:lucario") == \
        ("vsolver", "checkpoints/osv3_setupval1.pt", "lucario")
    assert mr.spec_deck(("vsolver", "x.pt", "lucario")) == "lucario"


def test_score_leaf_leaf_value_replaces_tail_keeps_anchors():
    import rl.turn_solver as ts
    from tests import builders as b
    me = b.player(active=b.pokemon(1, energies=[6]), bench=[b.pokemon(4)])
    opp = b.player(active=b.pokemon(4, hp=300))
    root = b.observation(me=me, opponent=opp)
    snap = ts._root_snapshot(root)
    # non-terminal leaf: learned tail replaces heuristics entirely
    leaf = b.observation(me=me, opponent=opp)
    base = ts.score_leaf(snap, leaf)
    learned = ts.score_leaf(snap, leaf, leaf_value=lambda o: 1234.0)
    assert learned == 1234.0                 # prizes 0 + tail replaced
    assert learned != base or base == 1234.0
    # terminal anchors ignore leaf_value
    won = b.observation(me=me, opponent=opp, result=0)
    assert ts.score_leaf(snap, won, leaf_value=lambda o: 1234.0) == ts.W_WIN
