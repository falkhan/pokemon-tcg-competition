"""Shared direct-engine match runner — ONE battle loop for every evaluator (M7.2).

Before this module, three separate battle-loop implementations drifted apart:
rl/collector.py's worker, rl/deck_search.py's matchup (hardcoded Lucario pilots),
and rl/eval.py. This is now the canonical home of the OpponentSpec vocabulary and
the slot-fair game loop; deck_search delegates here, the collector migrates in
M7.3 when it gains per-game deck sampling (it records training tensors mid-game —
a recording concern layered on top of match running), and eval.play_games stays
on kaggle_environments because it exercises the SHIPPED file agents (the deploy
surface, cross-checked against this fast path in M7-plan verification item 5).

Engine constraint (cg/sim.py holds one live battle per process): a worker runs
many battles sequentially (battle_start -> loop -> battle_finish per game);
parallelism is process-level via a spawn Pool, with only picklable str/int job
tuples crossing the boundary — the rl/collector.py pattern.

Opponent specs (picklable tuples; deck = decks/ name | csv path | list of ids):
  ("rule",  agent, deck)    rule expert brain ("lucario"/"iono"/"tuned") on a deck
  ("model", ckpt, deck)     neural pilot (greedy OptionScorer) from a checkpoint
  ("generic", deck)         the deck-agnostic rule pilot
  ("solver", deck)          generic pilot + within-turn combo solver (M7.4a)
  ("solver-dev", deck)      solver + the development tier (M8.1: setup search)
  ("solver-model", ckpt, deck)  neural pilot + the lethal combo solver (M9)
  ("random", deck)          uniform-random legal moves

Usage:
  python -m rl.matchrunner play --a generic:lucario --b random:kyogre -n 60
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK_DIR = ROOT / "decks"

OpponentSpec = tuple


def resolve_deck(deck) -> list[int]:
    """A decks/ name, a csv path (absolute or repo-ROOT-relative), or an id
    list -> 60 card ids. League specs store ROOT-relative POSIX paths so
    data/league/league.json is portable across machines/OSes."""
    if isinstance(deck, (list, tuple)):
        return list(deck)
    path = Path(deck)
    if not path.suffix:
        path = DECK_DIR / f"{deck}.csv"
    elif not path.is_absolute() and not path.exists():
        path = ROOT / path
    return [int(x) for x in path.read_text().split() if x.strip()]


def spec_deck(spec: OpponentSpec):
    """The deck slot of a spec (unresolved)."""
    return spec[2] if spec[0] in ("rule", "model", "solver-model", "mcts") else spec[1]


def parse_spec(s: str) -> OpponentSpec:
    """CLI shorthand -> spec tuple: "generic:lucario", "rule:iono[:deck]",
    "model:checkpoints/bc_v1.pt:kyogre", "random:kyogre"."""
    parts = s.split(":")
    kind = parts[0]
    if kind in ("generic", "random", "solver", "solver-dev") and len(parts) == 2:
        return (kind, parts[1])
    if kind == "mcts" and len(parts) == 4:
        return ("mcts", parts[1], parts[2], int(parts[3]))
    if kind == "rule" and len(parts) in (2, 3):
        return ("rule", parts[1], parts[2] if len(parts) == 3 else parts[1])
    if kind in ("model", "solver-model") and len(parts) == 3:
        return (kind, parts[1], parts[2])
    raise ValueError(f"cannot parse opponent spec {s!r} "
                     "(want kind:deck or rule:agent[:deck] or model:ckpt:deck)")


def make_pilot(spec: OpponentSpec, instance: str):
    """Build (agent_callable, deck_ids) for a spec. [ENGINE] for rule/model.

    `instance` must be unique per live rule pilot — teacher modules keep
    module-level mutable state (rl/teacher.py docstring).
    """
    kind = spec[0]
    if kind == "rule":
        from rl.teacher import load_teacher
        # deck=spec[1]: load_teacher's deck param only sets module.my_deck (the
        # kaggle-env deck return, unused in direct loops) and accepts names only;
        # the battle deck is resolved from the spec's own deck slot below.
        return load_teacher(instance, agent=spec[1], deck=spec[1]), resolve_deck(spec[2])
    if kind == "generic":
        from rl.generic_pilot import make_generic_pilot
        ids = resolve_deck(spec[1])
        return make_generic_pilot(ids), ids
    if kind == "solver":
        from rl.turn_solver import make_solver_pilot
        ids = resolve_deck(spec[1])
        return make_solver_pilot(ids, instance=instance), ids
    if kind == "solver-dev":
        from rl.turn_solver import make_solver_pilot
        ids = resolve_deck(spec[1])
        return make_solver_pilot(ids, instance=instance, dev=True), ids
    if kind == "solver-model":
        # ("solver-model", ckpt, deck) — the M9 hybrid: the neural checkpoint
        # plays every prompt EXCEPT where the lethal triggers T1–T4 fire and
        # the turn solver finds a prize line (the solver's whole edge over the
        # generic pilot; the checkpoint inherits it without retraining).
        from rl.turn_solver import make_solver_pilot
        fn, ids = make_pilot(("model", spec[1], spec[2]), instance)
        return make_solver_pilot(ids, instance=instance, inner=fn), ids
    if kind == "mcts":
        # ("mcts", ckpt, deck, n_sims) — the M8.4(b) sims-ladder instrument:
        # MCTS over the checkpoint's own policy/value with L3 archetype
        # determinization (rl/determinize.py). Dimension-aware like "model".
        import torch
        from rl.determinize import load_meta
        from rl.mcts import make_mcts_agent
        from rl.policy import OptionScorer
        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        sd = torch.load(ckpt, map_location="cpu")
        m = OptionScorer(state_ctx_dim=sd["state_enc.0.weight"].shape[1])
        m.load_state_dict(sd)
        m.eval()
        ids = resolve_deck(spec[2])
        return make_mcts_agent(m, ids, n_sims=int(spec[3]), meta=load_meta()), ids
    if kind == "random":
        rng = random.Random(hash(instance) & 0xFFFF)
        fn = lambda od: rng.sample(range(len(od["select"]["option"])),  # noqa: E731
                                   od["select"]["maxCount"])
        return fn, resolve_deck(spec[1])
    if kind == "model":
        import numpy as np
        import torch
        from cg.api import to_observation_class
        from rl.encoders import (COMBAT_SLICE, N_COMBAT, N_CONTEXTS, STATE_DIM,
                                 encode_context, encode_option, encode_state)
        from rl.policy import OptionScorer

        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt          # league specs store ROOT-relative paths
        sd = torch.load(ckpt, map_location="cpu")

        if "embedding.weight" in sd:
            # Encoders-v2 checkpoint (OptionScorerV2, M7.3): id embeddings +
            # deck-context pools — the pilot closes over its own deck list.
            from rl.encoders import encode_option_v2, encode_state_v2
            from rl.policy import OptionScorerV2
            m2 = OptionScorerV2()
            m2.load_state_dict(sd)
            m2.eval()
            deck_ids = resolve_deck(spec[2])

            def fn2(od):
                obs = to_observation_class(od)
                num, sids = encode_state_v2(obs.current, deck_ids)
                sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
                pairs = [encode_option_v2(o, obs) for o in obs.select.option]
                opts = np.stack([n for n, _ in pairs]).astype(np.float32)
                oids = np.stack([i for _, i in pairs])
                return m2.act(sc, sids, opts, oids, obs.select.maxCount, greedy=True)
            return fn2, deck_ids

        # Dimension-aware v1 load: bc_v1 predates the M3 combat features. Its
        # state input is exactly N_COMBAT narrower, and the combat block is a
        # contiguous slice of the current encoding — slicing it out
        # reconstructs the encoder the checkpoint was trained on.
        in_dim = sd["state_enc.0.weight"].shape[1]
        expected = STATE_DIM + N_CONTEXTS
        if in_dim == expected:
            cut = None
        elif in_dim == expected - N_COMBAT:
            cut = COMBAT_SLICE
        else:
            raise ValueError(
                f"{spec[1]}: state input dim {in_dim} matches neither the current "
                f"encoder ({expected}) nor the pre-M3 one ({expected - N_COMBAT})")
        m = OptionScorer(state_ctx_dim=in_dim)
        m.load_state_dict(sd)
        m.eval()

        def fn(od):
            obs = to_observation_class(od)
            sc = np.concatenate([encode_state(obs.current),
                                 encode_context(obs.select.context)]).astype(np.float32)
            if cut is not None:
                sc = np.delete(sc, np.s_[cut[0]:cut[1]])
            opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
            with torch.no_grad():
                logits, _ = m(torch.from_numpy(sc).unsqueeze(0), torch.from_numpy(opts).unsqueeze(0))
            order = torch.argsort(logits.squeeze(0), descending=True).tolist()
            return [int(i) for i in order[: obs.select.maxCount]]
        return fn, resolve_deck(spec[2])
    raise ValueError(f"unknown opponent spec kind: {spec!r}")


def _engine_game(fn0, fn1, deck0: list[int], deck1: list[int],
                 stats: dict | None = None) -> int:
    """One battle on the direct engine loop. Returns the winner seat (0/1) or 2
    for a draw. `stats`, if given, accumulates per-seat move counts, wall time,
    and agent exceptions under stats[0] / stats[1] — the G1/G6 gate inputs —
    and captures the end state under stats["final"] (the loss-forensics data:
    per-seat deck counts and prizes remaining). [ENGINE]"""
    from cg.game import battle_start, battle_select, battle_finish

    obs_dict, start = battle_start(deck0, deck1)
    if start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"battle_start rejected a deck (errorType={start.errorType})")
    try:
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            fn = fn0 if seat == 0 else fn1
            if stats is not None:
                s = stats.setdefault(seat, {"moves": 0, "time_s": 0.0, "errors": 0})
                t0 = time.perf_counter()
                try:
                    picks = fn(obs_dict)
                except Exception:  # noqa: BLE001 — G1 counts crashes, then re-raises
                    s["errors"] += 1
                    raise
                dt = time.perf_counter() - t0
                s["time_s"] += dt
                s["moves"] += 1
                if stats.get("collect_samples"):   # per-move p99 (M7.4a G6)
                    s.setdefault("samples", []).append(dt)
            else:
                picks = fn(obs_dict)
            obs_dict = battle_select([int(i) for i in picks])
        if stats is not None:
            players = obs_dict["current"]["players"]
            stats["final"] = {
                "decks": [p.get("deckCount") for p in players],
                "prizes": [len(p.get("prize", [])) for p in players],
            }
        return obs_dict["current"]["result"]
    finally:
        battle_finish()


def play_series(spec_a: OpponentSpec, spec_b: OpponentSpec, n_games: int,
                seed: int = 0, game_fn=None, stats: dict | None = None,
                on_game=None) -> list[int]:
    """Slot-fair series: a takes seat g%2. Returns 0 = a won, 1 = b won, 2 = draw
    per game. `game_fn(fn0, fn1, deck0, deck1, stats)` is the test seam (defaults
    to the [ENGINE] loop). `stats`, if given, accumulates per-SIDE ("a"/"b")
    move/time/error totals across the series; pre-seed it with
    {"collect_samples": True} to also keep every per-move latency under
    side["samples"] (the M7.4a p99 input). `on_game(g, result, seat_stats)`
    is called after each game with a's result and that game's raw stats — the
    loss-forensics hook (the CLI's --diag)."""
    game = game_fn or _engine_game
    fn_a, deck_a = make_pilot(spec_a, instance=f"mr{seed}_a")
    fn_b, deck_b = make_pilot(spec_b, instance=f"mr{seed}_b")
    collect = stats is not None and bool(stats.get("collect_samples"))
    out = []
    for g in range(n_games):
        a_seat = g % 2
        fns = (fn_a, fn_b) if a_seat == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        seat_stats: dict | None = None
        if stats is not None or on_game:
            seat_stats = {"collect_samples": True} if collect else {}
        res = game(fns[0], fns[1], decks[0], decks[1], seat_stats)
        if stats is not None:
            for seat in (0, 1):
                if seat not in seat_stats:
                    continue
                side = stats.setdefault("a" if seat == a_seat else "b",
                                        {"moves": 0, "time_s": 0.0, "errors": 0})
                for k in ("moves", "time_s", "errors"):
                    side[k] += seat_stats[seat][k]
                if collect:
                    side.setdefault("samples", []).extend(
                        seat_stats[seat].get("samples", ()))
        result = 2 if res == 2 else (0 if res == a_seat else 1)
        if on_game is not None:
            on_game(g, result, _from_a_view(seat_stats, a_seat))
        out.append(result)
    return out


def _from_a_view(seat_stats: dict, a_seat: int) -> dict:
    """Re-key one game's stats from seat indices to side-a's perspective."""
    view = {"a": seat_stats.get(a_seat), "b": seat_stats.get(1 - a_seat)}
    final = seat_stats.get("final")
    if final:
        view["final"] = {k: [v[a_seat], v[1 - a_seat]] for k, v in final.items()}
    return view


def series_wr(results: list[int]) -> float:
    """Win rate for side a, draws counting half."""
    if not results:
        return 0.0
    return (sum(1 for r in results if r == 0) + 0.5 * sum(1 for r in results if r == 2)) / len(results)


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); pure Python, no numpy."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    rank = -(-q * len(ys) // 100)                     # ceil without math
    return ys[min(len(ys) - 1, max(0, int(rank) - 1))]


# ---------------------------------------------------------------------------
# Multiprocessing across pairs (the rl/collector.py Pool pattern)
# ---------------------------------------------------------------------------
def _make_jobs(pairs: list[tuple], workers: int, seed_base: int = 1000) -> list[tuple]:
    """Split (spec_a, spec_b, n) pairs into even game chunks, ~2 jobs per worker.
    Chunks keep even sizes so each stays slot-fair. Pure — unit-tested offline.
    Job = (pair_idx, spec_a, spec_b, n_chunk, seed)."""
    total = sum(n for _, _, n in pairs)
    if total == 0:
        return []
    # target chunk size: fill ~2*workers jobs, rounded to even, min 2
    target = max(2, (total // max(1, 2 * workers) + 1) // 2 * 2)
    jobs = []
    for pair_idx, (a, b, n) in enumerate(pairs):
        done = 0
        while done < n:
            chunk = min(target, n - done)
            jobs.append((pair_idx, a, b, chunk, seed_base + len(jobs)))
            done += chunk
    return jobs


def _pair_worker(arg: tuple) -> tuple[int, int, list[int]]:
    job_idx, (pair_idx, spec_a, spec_b, n, seed) = arg
    return job_idx, pair_idx, play_series(spec_a, spec_b, n, seed=seed)


def _run_key(pairs: list[tuple], workers: int, seed: int) -> dict:
    """The checkpoint header — json-normalized so tuple/list mismatch can't
    false-negative the resume validation."""
    return json.loads(json.dumps(
        {"pairs": pairs, "workers": workers, "seed": seed}))


def run_pairs(pairs: list[tuple], workers: int = 4, game_fn=None,
              seed: int = 0, checkpoint: str | None = None) -> list[list[int]]:
    """Run [(spec_a, spec_b, n_games), ...]; returns per-pair result lists.

    workers <= 1 runs in-process (required for an injected game_fn — callables
    don't cross the spawn boundary). Otherwise a spawn Pool over even game
    chunks; only str/int tuples are pickled. [ENGINE] unless game_fn given.

    seed offsets every chunk's instance seed (before M8.1 the CLI --seed was
    silently dropped on this path; note the engine's own RNG drives game
    variance either way — repeated runs are independent samples).

    checkpoint (M8.1): a jsonl path. Line 1 pins the run key
    (pairs/workers/seed); each completed chunk appends one line as it
    finishes, and a rerun with the SAME key resumes, skipping completed
    chunks — long measurements survive crashes and pauses. A key mismatch
    raises instead of silently mixing two different runs."""
    if workers <= 1 or game_fn is not None:
        if workers > 1:
            raise ValueError("game_fn requires workers<=1 (not picklable)")
        return [play_series(a, b, n, seed=1000 + seed + k, game_fn=game_fn)
                for k, (a, b, n) in enumerate(pairs)]

    jobs = _make_jobs(pairs, workers, seed_base=1000 + seed)
    results: list[list[int]] = [[] for _ in pairs]
    done: set[int] = set()
    fh = None
    if checkpoint:
        path = Path(checkpoint)
        key = _run_key(pairs, workers, seed)
        if path.exists() and path.read_text().strip():
            lines = [json.loads(line) for line in path.read_text().splitlines()
                     if line.strip()]
            if lines[0] != key:
                raise ValueError(f"{checkpoint} belongs to a different run "
                                 "(header mismatch) — delete it or use a new path")
            for row in lines[1:]:
                done.add(row["job"])
                results[row["pair"]].extend(row["results"])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(key) + "\n")
        fh = path.open("a")

    pending = [(i, job) for i, job in enumerate(jobs) if i not in done]
    try:
        if pending:
            ctx = mp.get_context("spawn")
            with ctx.Pool(min(workers, len(pending))) as pool:
                for job_idx, pair_idx, chunk in pool.imap_unordered(_pair_worker,
                                                                    pending):
                    results[pair_idx].extend(chunk)
                    if fh is not None:
                        fh.write(json.dumps({"job": job_idx, "pair": pair_idx,
                                             "results": chunk}) + "\n")
                        fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return results


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("play", help="one spec-vs-spec series, slot-fair")
    s.add_argument("--a", required=True, help="e.g. generic:lucario, rule:iono, model:<ckpt>:<deck>")
    s.add_argument("--b", required=True)
    s.add_argument("-n", "--games", type=int, default=60)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--diag", action="store_true",
                   help="per-game end-state lines (loss forensics; forces workers=1)")
    s.add_argument("--latency", action="store_true",
                   help="per-move latency mean/p50/p99 ms per side (M7.4a G6; "
                        "forces workers=1)")
    s.add_argument("--checkpoint", default=None, metavar="FILE",
                   help="jsonl chunk checkpoint: appends per completed chunk; "
                        "rerunning the same command resumes (M8.1)")
    a = p.parse_args()

    spec_a, spec_b = parse_spec(a.a), parse_spec(a.b)
    on_game = None
    if a.diag:
        def on_game(g, result, view):
            f = view.get("final") or {}
            decks = f.get("decks", ["?", "?"])
            prizes = f.get("prizes", ["?", "?"])
            moves = (view.get("a") or {}).get("moves", "?")
            print(f"game {g:>3}: {'WLD'[result]}  moves={moves:>3}  "
                  f"deck a/b={decks[0]}/{decks[1]}  prizes-left a/b={prizes[0]}/{prizes[1]}",
                  flush=True)
    stats = {"collect_samples": True} if a.latency else None
    if a.diag or a.latency or a.workers <= 1:
        results = play_series(spec_a, spec_b, a.games, seed=a.seed,
                              stats=stats, on_game=on_game)
    else:
        results = run_pairs([(spec_a, spec_b, a.games)], workers=a.workers,
                            seed=a.seed, checkpoint=a.checkpoint)[0]
    w = sum(1 for r in results if r == 0)
    d = sum(1 for r in results if r == 2)
    print(f"{a.a} vs {a.b}: {w}W {len(results) - w - d}L {d}D over {len(results)} "
          f"(wr={series_wr(results):.3f})", flush=True)
    if a.latency:
        for side in ("a", "b"):
            ms = [1000 * x for x in (stats.get(side) or {}).get("samples", [])]
            if ms:
                print(f"side {side}: mean={sum(ms) / len(ms):.1f}ms  "
                      f"p50={percentile(ms, 50):.1f}ms  p99={percentile(ms, 99):.1f}ms  "
                      f"over {len(ms)} moves", flush=True)


if __name__ == "__main__":
    _main()
