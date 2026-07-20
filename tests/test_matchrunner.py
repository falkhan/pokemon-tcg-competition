"""M7.2 match runner: spec parsing, slot-fair series, job math, delegation.

The engine game loop itself is [ENGINE] (import-smoke only, house convention);
everything here runs through the injected game_fn seam.
"""
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import rl.deck_search as ds  # noqa: E402
import rl.matchrunner as mr  # noqa: E402

DECKS = Path(__file__).resolve().parent.parent / "decks"
LUCARIO = [int(x) for x in (DECKS / "lucario.csv").read_text().split()]


@pytest.mark.parametrize("s,expected", [
    ("generic:lucario", ("generic", "lucario")),
    ("random:kyogre", ("random", "kyogre")),
    ("solver:lucario", ("solver", "lucario")),
    ("solver-dev:lucario", ("solver-dev", "lucario")),
    ("mcts:checkpoints/bc_v1_value_search.pt:kyogre:32",
     ("mcts", "checkpoints/bc_v1_value_search.pt", "kyogre", 32)),
    ("rule:iono", ("rule", "iono", "iono")),
    ("rule:lucario:kyogre", ("rule", "lucario", "kyogre")),
    ("rule:dragapult", ("rule", "dragapult", "dragapult")),
    ("model:checkpoints/bc_v1.pt:kyogre", ("model", "checkpoints/bc_v1.pt", "kyogre")),
    ("solver2:lucario", ("solver2", "lucario")),
    ("solver2a:lucario", ("solver2a", "lucario")),
    ("generic2b:kyogre", ("generic2b", "kyogre")),
    ("ext:bundles/buddy:decks/lucario.csv",
     ("ext", "bundles/buddy", "decks/lucario.csv")),
])
def test_parse_spec_kinds(s, expected):
    assert mr.parse_spec(s) == expected


@pytest.mark.parametrize("junk", ["generic", "banana:x", "model:onlyckpt", "rule:a:b:c"])
def test_parse_spec_rejects_junk(junk):
    with pytest.raises(ValueError, match="spec"):
        mr.parse_spec(junk)


def test_resolve_and_spec_deck_contract():
    assert mr.resolve_deck("lucario") == LUCARIO
    assert mr.resolve_deck(str(DECKS / "lucario.csv")) == LUCARIO
    assert mr.resolve_deck(LUCARIO) == LUCARIO
    assert mr.spec_deck(("rule", "lucario", "kyogre")) == "kyogre"
    assert mr.spec_deck(("generic", "lucario")) == "lucario"
    # deck_search's historic helpers still delegate here
    assert ds._resolve_deck("lucario") == LUCARIO


def _fake_pilots(monkeypatch, record=None):
    """make_pilot without the engine: callable is the instance name, deck resolved."""
    def fake(spec, instance):
        if record is not None:
            record.append((spec, instance))
        return instance, mr.resolve_deck(mr.spec_deck(spec))
    monkeypatch.setattr(mr, "make_pilot", fake)


def test_play_series_is_slot_fair_and_maps_draws(monkeypatch):
    _fake_pilots(monkeypatch)
    seats = []

    def game_fn(fn0, fn1, deck0, deck1, stats):
        seats.append((fn0, fn1))
        return 0  # seat 0 always wins

    r = mr.play_series(("generic", "lucario"), ("generic", "iono"), 4, game_fn=game_fn)
    # a sits seat g%2: wins games 0,2 (seat 0) and loses 1,3 (b took seat 0)
    assert r == [0, 1, 0, 1]
    assert seats[0][0].endswith("_a") and seats[1][0].endswith("_b")
    r = mr.play_series(("generic", "lucario"), ("generic", "iono"), 2,
                       game_fn=lambda *a: 2)
    assert r == [2, 2]


def test_play_series_uses_distinct_pilot_instances(monkeypatch):
    record = []
    _fake_pilots(monkeypatch, record)
    mr.play_series(("rule", "lucario", "lucario"), ("rule", "lucario", "lucario"), 1,
                   seed=7, game_fn=lambda *a: 0)
    names = [inst for _, inst in record]
    assert len(names) == 2 and names[0] != names[1]  # module-state isolation


def test_play_series_accumulates_per_side_stats(monkeypatch):
    _fake_pilots(monkeypatch)

    def game_fn(fn0, fn1, deck0, deck1, stats):
        stats[0] = {"moves": 10, "time_s": 0.1, "errors": 0}
        stats[1] = {"moves": 8, "time_s": 0.4, "errors": 1}
        return 0

    stats: dict = {}
    mr.play_series(("generic", "lucario"), ("generic", "iono"), 2,
                   game_fn=game_fn, stats=stats)
    # a sat seat 0 then seat 1 -> gets 10+8 moves; b the mirror
    assert stats["a"] == {"moves": 18, "time_s": 0.5, "errors": 1}
    assert stats["b"] == {"moves": 18, "time_s": 0.5, "errors": 1}


def test_play_series_collects_latency_samples_when_asked(monkeypatch):
    _fake_pilots(monkeypatch)

    def game_fn(fn0, fn1, deck0, deck1, stats):
        assert stats.get("collect_samples")   # the flag reaches _engine_game
        stats[0] = {"moves": 2, "time_s": 0.3, "errors": 0, "samples": [0.1, 0.2]}
        stats[1] = {"moves": 1, "time_s": 0.5, "errors": 0, "samples": [0.5]}
        return 0

    stats: dict = {"collect_samples": True}
    mr.play_series(("generic", "lucario"), ("generic", "iono"), 2,
                   game_fn=game_fn, stats=stats)
    # a sat seat 0 then seat 1 -> its samples are game0-seat0 + game1-seat1
    assert stats["a"]["samples"] == [0.1, 0.2, 0.5]
    assert stats["b"]["samples"] == [0.5, 0.1, 0.2]
    assert stats["a"]["moves"] == 3 and stats["a"]["time_s"] == pytest.approx(0.8)

    def plain_game(fn0, fn1, deck0, deck1, stats):
        assert "collect_samples" not in stats  # flag only propagates when set
        stats[0] = {"moves": 2, "time_s": 0.3, "errors": 0}
        stats[1] = {"moves": 1, "time_s": 0.5, "errors": 0}
        return 0

    plain: dict = {}
    mr.play_series(("generic", "lucario"), ("generic", "iono"), 2,
                   game_fn=plain_game, stats=plain)
    assert "samples" not in plain["a"]         # opt-in only


def test_percentile_nearest_rank():
    ms = list(range(1, 101))                   # 1..100
    assert mr.percentile(ms, 50) == 50
    assert mr.percentile(ms, 99) == 99
    assert mr.percentile(ms, 100) == 100
    assert mr.percentile([7.0], 99) == 7.0
    assert mr.percentile([], 99) == 0.0


def test_make_pilot_solver_wraps_the_generic_pilot():
    fn, deck = mr.make_pilot(("solver", "lucario"), "t0")
    assert callable(fn) and len(deck) == 60


def test_make_pilot_fixed_kinds_build():
    # M9 Leg 1: every pilot-v2 variant builds and resolves its deck.
    for kind in mr._FIXED_KINDS:
        fn, deck = mr.make_pilot((kind, "lucario"), f"t_{kind}")
        assert callable(fn) and len(deck) == 60
    assert mr.spec_deck(("solver2", "lucario")) == "lucario"
    assert mr.spec_deck(("ext", "bundles/buddy", "kyogre")) == "kyogre"


def test_series_wr_counts_draws_as_half():
    assert mr.series_wr([0, 0, 1, 2]) == pytest.approx(0.625)
    assert mr.series_wr([]) == 0.0


def test_make_jobs_even_chunks_cover_all_games():
    pairs = [(("generic", "a"), ("generic", "b"), 40),
             (("generic", "a"), ("generic", "c"), 6)]
    jobs = mr._make_jobs(pairs, workers=4)
    per_pair = {0: 0, 1: 0}
    seeds = set()
    for pair_idx, _a, _b, n, seed in jobs:
        per_pair[pair_idx] += n
        seeds.add(seed)
        assert n % 2 == 0 or n == per_pair[pair_idx]  # even chunks keep slot-fairness
    assert per_pair == {0: 40, 1: 6}
    assert len(seeds) == len(jobs)  # distinct seeds
    assert mr._make_jobs([], 4) == []


def test_run_pairs_inprocess_with_game_fn(monkeypatch):
    _fake_pilots(monkeypatch)
    pairs = [(("generic", "lucario"), ("generic", "iono"), 4),
             (("generic", "iono"), ("generic", "kyogre"), 2)]
    results = mr.run_pairs(pairs, workers=1, game_fn=lambda *a: 0)
    assert results == [[0, 1, 0, 1], [0, 1]]
    with pytest.raises(ValueError, match="workers"):
        mr.run_pairs(pairs, workers=4, game_fn=lambda *a: 0)


def test_deck_search_wrappers_delegate(monkeypatch):
    calls = []

    def fake_series(spec_a, spec_b, n, seed=0, game_fn=None, stats=None):
        calls.append((spec_a, spec_b, n))
        return [0] * n

    monkeypatch.setattr(mr, "play_series", fake_series)
    assert ds.matchup([1] * 60, [2] * 60, 4, agent="iono") == [0] * 4
    assert calls[-1] == (("rule", "iono", [1] * 60), ("rule", "iono", [2] * 60), 4)
    assert ds._play_vs_spec([1] * 60, ("random", "kyogre"), 3) == [0] * 3
    assert calls[-1] == (("generic", [1] * 60), ("random", "kyogre"), 3)
    assert ds._play_vs_spec([1] * 60, ("random", "kyogre"), 3, pilot="lucario") == [0] * 3
    assert calls[-1][0] == ("rule", "lucario", [1] * 60)


def test_resolve_deck_falls_back_to_repo_root(monkeypatch, tmp_path):
    # League specs store ROOT-relative paths; they must resolve from any cwd
    # (the committed league.json broke on Windows with absolute sandbox paths).
    monkeypatch.chdir(tmp_path)
    assert mr.resolve_deck("decks/lucario.csv") == LUCARIO


def test_model_pilot_loads_pre_m3_checkpoints(tmp_path):
    # bc_v1 predates the M3 combat features: its state input is N_COMBAT
    # narrower and make_pilot must adapt (slice the combat block out) instead
    # of crashing with a shape mismatch (the measured selftest failure).
    torch = pytest.importorskip("torch")
    from rl.encoders import N_COMBAT, N_CONTEXTS, STATE_DIM
    from rl.policy import OptionScorer

    old_dim = STATE_DIM + N_CONTEXTS - N_COMBAT
    ckpt = tmp_path / "old.pt"
    torch.save(OptionScorer(state_ctx_dim=old_dim).state_dict(), ckpt)
    fn, deck = mr.make_pilot(("model", str(ckpt), "kyogre"), "t0")
    assert callable(fn) and len(deck) == 60

    current = tmp_path / "new.pt"
    torch.save(OptionScorer().state_dict(), current)
    fn, _ = mr.make_pilot(("model", str(current), "kyogre"), "t1")
    assert callable(fn)

    weird = tmp_path / "weird.pt"
    torch.save(OptionScorer(state_ctx_dim=old_dim - 5).state_dict(), weird)
    with pytest.raises(ValueError, match="matches neither"):
        mr.make_pilot(("model", str(weird), "kyogre"), "t2")


def test_model_pilot_loads_v2_checkpoints(tmp_path):
    # M7.3: OptionScorerV2 checkpoints (embedding.weight present) get the
    # encoders-v2 path with the deck closed over for deck-context features.
    torch = pytest.importorskip("torch")
    from rl.policy import OptionScorerV2

    ckpt = tmp_path / "osv2.pt"
    torch.save(OptionScorerV2().state_dict(), ckpt)
    fn, deck = mr.make_pilot(("model", str(ckpt), "kyogre"), "v2")
    assert callable(fn) and len(deck) == 60


# --- M8.1: chunk checkpointing / resume in run_pairs ---------------------------

class _FakePool:
    """In-process stand-in for the spawn Pool (checkpoint tests are offline)."""

    def __init__(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, iterable):
        return (fn(x) for x in iterable)


def _patch_pool(monkeypatch, series_fn):
    from types import SimpleNamespace
    monkeypatch.setattr(mr, "play_series", series_fn)
    monkeypatch.setattr(mr.mp, "get_context",
                        lambda kind: SimpleNamespace(Pool=_FakePool))


def test_run_pairs_checkpoint_writes_and_resumes(tmp_path, monkeypatch):
    calls = []
    _patch_pool(monkeypatch, lambda a, b, n, seed=0, **k:
                calls.append((n, seed)) or [0] * n)
    pairs = [(("generic", "x"), ("generic", "y"), 8)]
    ck = tmp_path / "run.jsonl"

    r1 = mr.run_pairs(pairs, workers=2, seed=5, checkpoint=str(ck))
    assert [len(r) for r in r1] == [8]
    n_calls = len(calls)
    assert n_calls > 1                       # really chunked
    assert len(ck.read_text().splitlines()) == 1 + n_calls  # header + chunks

    r2 = mr.run_pairs(pairs, workers=2, seed=5, checkpoint=str(ck))
    assert len(calls) == n_calls             # fully resumed: zero new games
    assert [len(r) for r in r2] == [8]


def test_run_pairs_checkpoint_resumes_after_crash(tmp_path, monkeypatch):
    boom = {"after": 2}

    def flaky(a, b, n, seed=0, **k):
        if boom["after"] == 0:
            raise RuntimeError("worker died")
        boom["after"] -= 1
        return [0] * n

    _patch_pool(monkeypatch, flaky)
    pairs = [(("generic", "x"), ("generic", "y"), 8)]
    ck = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError):
        mr.run_pairs(pairs, workers=2, seed=5, checkpoint=str(ck))
    survived = len(ck.read_text().splitlines()) - 1
    assert survived == 2                     # completed chunks persisted

    calls = []
    _patch_pool(monkeypatch, lambda a, b, n, seed=0, **k:
                calls.append(n) or [0] * n)
    r = mr.run_pairs(pairs, workers=2, seed=5, checkpoint=str(ck))
    assert [len(x) for x in r] == [8]
    assert len(calls) + survived == 4        # only the missing chunks re-ran


def test_run_pairs_checkpoint_rejects_different_run(tmp_path, monkeypatch):
    _patch_pool(monkeypatch, lambda a, b, n, seed=0, **k: [0] * n)
    pairs = [(("generic", "x"), ("generic", "y"), 8)]
    ck = tmp_path / "run.jsonl"
    mr.run_pairs(pairs, workers=2, seed=5, checkpoint=str(ck))
    with pytest.raises(ValueError, match="different run"):
        mr.run_pairs(pairs, workers=2, seed=6, checkpoint=str(ck))


def test_run_pairs_seed_threads_into_chunk_seeds():
    pairs = [(("generic", "x"), ("generic", "y"), 8)]
    seeds_a = {j[4] for j in mr._make_jobs(pairs, 2, seed_base=1001)}
    seeds_b = {j[4] for j in mr._make_jobs(pairs, 2, seed_base=1002)}
    assert seeds_a != seeds_b                # --seed finally changes something
