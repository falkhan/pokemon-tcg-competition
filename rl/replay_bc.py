"""Imitation from Kaggle leaderboard replays — "clone the leaderboard" (M10).

Why: every observable self-teacher caps the student at (teacher strength x
fidelity), and our strongest one (the solver pilot) IS the shipped agent — pure
self-imitation can never clear the 0.55 bar by construction (docs/M8-report.md).
The leaderboard episodes we already harvest (rl/kaggle_ingest.py) serialize
full per-seat observations plus the answering action for BOTH seats, from
teams scoring far above ours — the only teacher pool whose strength exceeds
the ship agent and whose reasoning is fully observable.

Conversion is nearly free: the env hands each seat its option menu in
``observation.select.option`` and the pilot returns indices into it
(rl/matchrunner.py model pilot), so a replay decision encodes with the exact
inference-path calls — no engine replay, no option-list reconstruction. The
one alignment trap is the M8.0 off-by-one: the action answering step i's
select is recorded at steps[i+1] (rl/postmortem.py ``_action_for``). A second
trap postmortem never hits: the INACTIVE seat carries a stale, repeated
``select`` every step — only ``status == "ACTIVE"`` steps are decisions.

This module is [ENGINE] (encoders import rl.combat -> cg); rl/kaggle_ingest.py
stays engine-free per its docstring contract and is imported for the parse /
cache layer only. Its ``bc-shards`` audit stub is superseded by ``build`` here.

Usage:
  python -m rl.replay_bc audit                      # G0: converter integrity, zero games
  python -m rl.replay_bc roundtrip -n 6             # G1: same-code alignment gate, >=98%
  python -m rl.replay_bc align                      # drift report vs old cached subs
  python -m rl.replay_bc build --min-score 550      # shards -> data/bc_kaggle
  python -m rl.replay_bc meta-eval --a model:checkpoints/osv2_kbc1.pt:lucario -n 60
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from cg.api import to_observation_class
from rl.deck_search import DECK_SIZE
from rl.encoders import (N_CONTEXTS, encode_context, encode_option_v2,
                         encode_state_v2)
from rl.kaggle_ingest import EPISODES_PQ, KAGGLE_DIR, RAW_DIR, deck_hash, parse_episode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "bc_kaggle"

# Seats piloted by us are excluded from teacher data by default (they are the
# G1 alignment fixture instead). Source: docs submission log — the parquet
# only sees subs that have episodes fetched, so keep this list by hand.
OUR_SUBS = {54474043, 54586430, 54616196, 54621283}

# deck_idx offset for replay decks — keeps them disjoint from the training
# deck-population indices when shard dirs are mixed (rl/bc.py multi-dir).
REPLAY_DECK_BASE = 1000


# ---------------------------------------------------------------------------
# Decision iteration — the alignment rules, each failure a counted drop-reason
# ---------------------------------------------------------------------------
def iter_replay_decisions(steps: list, seat: int, drops: Counter):
    """Yield (step_idx, obs_dict, action) for every prompt `seat` answered.

    LABEL_OUT_OF_RANGE is a converter-bug alarm, not a data blemish — the
    audit gate hard-fails on it (a recorded answer that doesn't index its own
    menu means our select<->action pairing is wrong).
    """
    for i, step in enumerate(steps):
        st = step[seat] if seat < len(step) and isinstance(step[seat], dict) else None
        if st is None or st.get("status") != "ACTIVE":
            continue  # INACTIVE seats repeat the last select — not a decision
        obs = st.get("observation")
        sel = obs.get("select") if isinstance(obs, dict) else None
        cur = obs.get("current") if isinstance(obs, dict) else None
        if not sel or not isinstance(cur, dict) or not cur.get("players"):
            drops["blank_obs"] += 1  # step-0 deck prompt (select is None) lands here
            continue
        options = sel.get("option") or []
        if not options:
            drops["empty_menu"] += 1
            continue
        if not 0 <= int(sel.get("context", -1)) < N_CONTEXTS:
            drops["context_overflow"] += 1  # SelectContext beyond the one-hot head-room
            continue
        nxt = steps[i + 1][seat] if i + 1 < len(steps) and isinstance(steps[i + 1][seat], dict) else None
        action = nxt.get("action") if nxt else None
        if not (isinstance(action, list) and action
                and all(isinstance(a, (int, float)) for a in action)):
            drops["no_next_action"] += 1  # terminal select / timed-out answer
            continue
        if len(action) == DECK_SIZE:
            drops["deck_return"] += 1  # the step-1 deck action
            continue
        if not 0 <= int(action[0]) < len(options):
            drops["LABEL_OUT_OF_RANGE"] += 1
            continue
        yield i, obs, [int(a) for a in action]


def encode_decisions(steps: list, seat: int, deck_ids: list[int],
                     drops: Counter, contexts: Counter | None = None) -> list[tuple]:
    """One seat's decisions -> collect_games_v2-shaped rows (rl/bc.py):
    (state_ctx, state_ids, opts, opt_ids, label). Identical encoder calls to
    the model pilot's inference path."""
    rows = []
    for _i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        state_num, state_ids = encode_state_v2(obs.current, deck_ids)
        state_ctx = np.concatenate([state_num, encode_context(obs.select.context)])
        pairs = [encode_option_v2(o, obs) for o in obs.select.option]
        opts = np.stack([num for num, _ in pairs])
        opt_ids = np.stack([ids for _, ids in pairs])
        if contexts is not None:
            contexts[int(obs.select.context)] += 1
        rows.append((state_ctx, state_ids, opts, opt_ids, action[0]))
    return rows


def _load_raw(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def _episode_meta() -> dict[int, dict]:
    if not EPISODES_PQ.exists():
        return {}
    return {r["episode_id"]: r
            for r in pl.read_parquet(EPISODES_PQ).iter_rows(named=True)}


def _cached_episodes() -> list[tuple[int, Path]]:
    out = []
    for path in sorted(RAW_DIR.glob("episode_*.json.gz")):
        out.append((int(path.stem.split("_")[1].split(".")[0]), path))
    return out


# ---------------------------------------------------------------------------
# G0 — converter audit: zero games, hard-fails on any integrity violation
# ---------------------------------------------------------------------------
def audit() -> dict:
    """Convert every cached episode, both seats, no filters. Report decision /
    drop counts and the context histogram; raise on encode errors or
    out-of-range labels (either means the converter is wrong, not the data)."""
    drops: Counter = Counter()
    contexts: Counter = Counter()
    errors: list[str] = []
    n_decisions = n_seats = n_eps = 0

    for eid, path in _cached_episodes():
        raw = _load_raw(path)
        ep = parse_episode(raw, episode_id=eid)
        n_eps += 1
        for seat in (0, 1):
            if ep.decks[seat] is None:
                drops["seat_no_deck"] += 1
                continue
            try:
                rows = encode_decisions(raw["steps"], seat, ep.decks[seat],
                                        drops, contexts)
            except Exception as e:  # noqa: BLE001 — every encode failure is reportable
                errors.append(f"episode {eid} seat {seat}: {type(e).__name__}: {e}")
                continue
            n_seats += 1
            n_decisions += len(rows)

    print(f"G0 audit: {n_eps} episodes, {n_seats} seats encoded, "
          f"{n_decisions} decisions", flush=True)
    for reason, n in drops.most_common():
        print(f"  dropped {reason:<20} {n}", flush=True)
    top = ", ".join(f"ctx{c}:{n}" for c, n in contexts.most_common(12))
    print(f"  contexts: {top}", flush=True)
    for e in errors[:10]:
        print(f"  ENCODE ERROR: {e}", flush=True)

    ok = not errors and drops["LABEL_OUT_OF_RANGE"] == 0
    print(f"G0 verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(
            f"G0 FAILED: {len(errors)} encode errors, "
            f"{drops['LABEL_OUT_OF_RANGE']} out-of-range labels")
    return {"episodes": n_eps, "seats": n_seats, "decisions": n_decisions,
            "drops": dict(drops), "contexts": dict(contexts)}


# ---------------------------------------------------------------------------
# G1 — converter alignment. The CANONICAL gate is `roundtrip`: it compares the
# converter against the SAME pilot code that produced the actions. `align`
# compares against old cached submissions and therefore mixes converter error
# with pilot-version drift — it's a drift REPORT, not a gate (measured
# 2026-07-15: 0.81/0.85 vs the M6/M7-era subs while roundtrip was 1.0000 on
# 905 decisions — the whole gap is the pilot fixes shipped since).
# ---------------------------------------------------------------------------
def align_roundtrip(n_games: int = 6, out_dir: Path | None = None) -> dict:
    """G1 gate: play the CURRENT rules bundle vs itself via
    kaggle_environments, dump env.toJSON() (byte-identical shape to a cached
    Kaggle episode), then replay every recorded observation through a fresh
    solver pilot and the converter. Same code on both sides, so ANY
    disagreement is a converter bug. Gate: >=0.98 first-pick agreement."""
    import tempfile

    from rl.eval import play_games
    from rl.matchrunner import make_pilot

    bundle = str(ROOT / "submission_rules" / "main.py")
    tmp = tempfile.TemporaryDirectory() if out_dir is None else None
    out = Path(out_dir) if out_dir else Path(tmp.name)
    wr, _ = play_games(bundle, bundle, n_games, json_prefix=str(out / "ep"))

    drops: Counter = Counter()
    first = full = total = 0
    mismatches: list[str] = []
    for path in sorted(out.glob("ep_g*.json")):
        raw = json.loads(path.read_text())
        ep = parse_episode(raw, episode_id=0)
        for seat in (0, 1):
            if ep.decks[seat] is None:
                raise SystemExit(f"{path.name} seat {seat}: deck extraction failed")
            fn = make_pilot(("solver", ep.decks[seat]),
                            instance=f"g1_{path.name}_{seat}")[0]
            for i, obs_dict, action in iter_replay_decisions(raw["steps"], seat, drops):
                maxc = int(obs_dict["select"].get("maxCount") or 1)
                pred = [int(x) for x in fn(obs_dict)][:maxc]
                total += 1
                if pred and pred[0] == action[0]:
                    first += 1
                elif len(mismatches) < 10:
                    sel = obs_dict["select"]
                    mismatches.append(
                        f"{path.name} seat{seat} step{i} ctx={sel.get('context')} "
                        f"recorded={action} pred={pred}")
                full += pred == action[:maxc]
    if tmp:
        tmp.cleanup()

    rate = first / total if total else 0.0
    print(f"G1 roundtrip: {n_games} mirror games, {total} decisions, "
          f"first-pick {rate:.4f}, full {full / max(1, total):.4f}", flush=True)
    for m in mismatches:
        print(f"  MISMATCH: {m}", flush=True)
    ok = total > 0 and rate >= 0.98
    print(f"G1 verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("G1 FAILED — converter alignment bug (same-code "
                         "disagreement cannot be pilot drift)")
    return {"games": n_games, "decisions": total, "first_pick": rate, "pass": ok}


def align(subs: set[int] | None = None) -> dict:
    """Drift REPORT (not a gate — see section note): replay episodes of our
    old cached submissions through today's generic AND solver pilots and
    report agreement with the recorded actions. Disagreement = converter
    error (ruled out by roundtrip) + pilot code evolved since that sub —
    useful as a fingerprint of how much behavior our fixes changed."""
    from rl.generic_pilot import make_generic_pilot
    from rl.matchrunner import make_pilot

    subs = subs or (OUR_SUBS & {54474043, 54586430, 54621283})  # rules-bundle subs
    meta = _episode_meta()
    stats: dict[int, dict] = {}  # sub -> {pilot -> [first_hits, full_hits, total]}
    drops: Counter = Counter()
    pilot_errors: Counter = Counter()
    n_eps = 0

    for eid, path in _cached_episodes():
        m = meta.get(eid)
        our_seat = m.get("our_seat") if m else None
        if our_seat is None:
            continue
        sub = m.get(f"submission_id_{our_seat}")
        if sub not in subs:
            continue
        raw = _load_raw(path)
        ep = parse_episode(raw, our_submission_ids=subs, episode_id=eid)
        deck = ep.decks[our_seat]
        if deck is None:
            continue
        n_eps += 1
        pilots = {"generic": make_generic_pilot(deck),
                  "solver": make_pilot(("solver", deck), instance=f"align{eid}")[0]}
        s = stats.setdefault(int(sub), {p: [0, 0, 0] for p in pilots})

        for _i, obs_dict, action in iter_replay_decisions(raw["steps"], our_seat, drops):
            maxc = int(obs_dict["select"].get("maxCount") or 1)
            for name, fn in pilots.items():
                try:
                    pred = [int(x) for x in fn(obs_dict)][:maxc]
                except Exception:  # noqa: BLE001 — counted, not fatal
                    pilot_errors[name] += 1
                    continue
                hit_first = bool(pred) and pred[0] == action[0]
                hit_full = pred == action[:maxc]
                rec = s[name]
                rec[0] += hit_first
                rec[1] += hit_full
                rec[2] += 1

    print(f"align drift report: {n_eps} own-seat episodes over subs {sorted(subs)}",
          flush=True)
    best_by_sub: dict[int, float] = {}
    for sub, by_pilot in sorted(stats.items()):
        for name, (first, full, total) in by_pilot.items():
            if not total:
                continue
            print(f"  sub {sub} vs {name:<8} first-pick {first / total:.4f}  "
                  f"full {full / total:.4f}  (n={total})", flush=True)
            best_by_sub[sub] = max(best_by_sub.get(sub, 0.0), first / total)
    for name, n in pilot_errors.items():
        print(f"  pilot errors ({name}): {n}", flush=True)

    print(f"per-sub best first-pick (drift, lower = more behavior change since "
          f"that sub): { {s: round(v, 4) for s, v in best_by_sub.items()} }",
          flush=True)
    return {"episodes": n_eps, "best_by_sub": best_by_sub}


# ---------------------------------------------------------------------------
# build — raw cache -> data/bc_kaggle shards (v2 schema + teacher metadata)
# ---------------------------------------------------------------------------
def build(out_dir: Path = DATA_DIR, min_score: float = 550.0,
          include_ours: bool = False, winners_only: bool = False,
          min_steps: int = 0, shard_size: int = 5000,
          only_subs: set[int] | None = None,
          only_deck_hash: str | None = None) -> None:
    """Encode qualifying replay seats into BC shards.

    Shard schema = collect_games_v2 (rl/bc.py) + additive columns the trainer
    treats as optional: teacher_score (leaderboard score of the imitated
    seat), seat_won (1/0), episode_ids (provenance). game_ids are one per
    EPISODE (both seats share it) so the by-game val split stays leak-free.
    deck_idx indexes deck_registry.json at REPLAY_DECK_BASE+.

    only_subs (M23): restrict to these submission ids — the clone-one-opponent
    path. Every other seat filter still applies, so pair it with a --min-score
    at or below the target's actual score.

    only_deck_hash (M24): restrict to seats whose decklist hash starts with
    this prefix — the strong-pilots-on-OUR-deck corpus (deck_hash is the
    order-invariant sha1 from rl.kaggle_ingest; a short unambiguous prefix
    like '20dcd313' is enough)."""
    meta = _episode_meta()
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = ("states", "state_ids", "options", "option_ids", "n_options",
               "labels", "game_ids", "results", "deck_idx",
               "teacher_score", "seat_won", "episode_ids")
    shard: dict[str, list] = {k: [] for k in columns}
    shard_idx = sum(1 for _ in out_dir.glob("shard_*.npz"))
    registry: dict[str, int] = {}
    drops: Counter = Counter()
    skips: Counter = Counter()
    game_id = n_seats = n_decisions = 0

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            state_ids=np.stack(shard["state_ids"]),
            options=np.concatenate(shard["options"]),
            option_ids=np.concatenate(shard["option_ids"]),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            labels=np.array(shard["labels"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
            teacher_score=np.array(shard["teacher_score"], dtype=np.float32),
            seat_won=np.array(shard["seat_won"], dtype=np.float32),
            episode_ids=np.array(shard["episode_ids"], dtype=np.int64),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    for eid, path in _cached_episodes():
        m = meta.get(eid)
        if m is None:
            skips["episode_no_meta"] += 1
            continue
        if m.get("status_0") != "DONE" or m.get("status_1") != "DONE":
            skips["episode_not_done"] += 1
            continue
        if min_steps and (m.get("n_steps") or 0) < min_steps:
            skips["episode_short"] += 1
            continue
        raw = _load_raw(path)
        ep = parse_episode(raw, episode_id=eid)
        took_seat = False

        for seat in (0, 1):
            sub = m.get(f"submission_id_{seat}")
            score = m.get(f"updated_score_{seat}")
            reward, opp_reward = ep.rewards[seat], ep.rewards[1 - seat]
            if only_subs is not None and sub not in only_subs:
                skips["seat_not_target"] += 1
                continue
            if not include_ours and (sub in OUR_SUBS or m.get("our_seat") == seat):
                skips["seat_ours"] += 1
                continue
            if score is None or score < min_score:
                skips["seat_low_score"] += 1
                continue
            if ep.decks[seat] is None:
                skips["seat_no_deck"] += 1
                continue
            if only_deck_hash is not None and not deck_hash(
                    ep.decks[seat]).startswith(only_deck_hash):
                skips["seat_other_deck"] += 1
                continue
            if None in (reward, opp_reward):
                skips["seat_no_reward"] += 1
                continue
            won = reward > opp_reward
            if winners_only and not won:
                skips["seat_not_winner"] += 1
                continue

            rows = encode_decisions(raw["steps"], seat, ep.decks[seat], drops)
            if not rows:
                skips["seat_no_decisions"] += 1
                continue
            didx = registry.setdefault(deck_hash(ep.decks[seat]),
                                       REPLAY_DECK_BASE + len(registry))
            result = 0.0 if reward == opp_reward else (1.0 if won else -1.0)
            for state_ctx, state_ids, opts, opt_ids, label in rows:
                shard["states"].append(state_ctx)
                shard["state_ids"].append(state_ids)
                shard["options"].append(opts)
                shard["option_ids"].append(opt_ids)
                shard["n_options"].append(len(opts))
                shard["labels"].append(label)
                shard["game_ids"].append(game_id)
                shard["results"].append(result)
                shard["deck_idx"].append(didx)
                shard["teacher_score"].append(float(score))
                shard["seat_won"].append(1.0 if won else 0.0)
                shard["episode_ids"].append(eid)
            took_seat = True
            n_seats += 1
            n_decisions += len(rows)

        if took_seat:
            game_id += 1
        if len(shard["labels"]) >= shard_size:
            flush()
    flush()

    (out_dir / "deck_registry.json").write_text(json.dumps({
        "base": REPLAY_DECK_BASE,
        "params": {"min_score": min_score, "winners_only": winners_only,
                   "include_ours": include_ours, "min_steps": min_steps,
                   "only_subs": sorted(only_subs) if only_subs else None,
                   "only_deck_hash": only_deck_hash},
        "decks": [{"deck_idx": i, "hash": h}
                  for h, i in sorted(registry.items(), key=lambda kv: kv[1])],
    }, indent=2))

    print(f"build: {game_id} episodes -> {n_seats} teacher seats, "
          f"{n_decisions} decisions, {shard_idx} shards in {out_dir}", flush=True)
    for reason, n in skips.most_common():
        print(f"  skipped {reason:<22} {n}", flush=True)
    for reason, n in drops.most_common():
        print(f"  dropped {reason:<22} {n}", flush=True)
    if drops["LABEL_OUT_OF_RANGE"]:
        raise SystemExit("LABEL_OUT_OF_RANGE > 0 — converter bug, shards suspect")


# ---------------------------------------------------------------------------
# meta-eval — candidate vs the frozen harvested-meta snapshot (G4 co-gate)
# ---------------------------------------------------------------------------
def _latest_snapshot() -> Path:
    snaps = sorted((p for p in KAGGLE_DIR.glob("meta_v*")
                    if p.name.removeprefix("meta_v").isdigit()),
                   key=lambda p: int(p.name.removeprefix("meta_v")))
    if not snaps:
        raise SystemExit("no data/kaggle/meta_v* snapshot — run "
                         "`python -m rl.kaggle_ingest meta` after a harvest")
    return snaps[-1]


def meta_eval(a: str, snapshot: Path | None = None, n: int = 60,
              workers: int = 8, seed: int = 0, checkpoint: str | None = None,
              exclude_mirror: bool = True, min_games: int = 3) -> dict:
    """Play spec `a` against the OFF-MIRROR archetypes of a meta snapshot.

    M22b retarget. The old form played every snapshot deck at equal n and
    reported a manifest-weighted mean. Two defects made that number worse than
    useless:

    1. **Its dominant cell IS the mirror gate.** `deck_20dcd3130bc0.csv` (weight
       0.899) is byte-identical to `decks/lucario.csv` — md5 aef8da62… — and both
       sides are solver-piloted. So ~90% of the "meta score" re-measured, at n=60,
       exactly what the mirror gate measures at n=800. There was never a
       mirror-vs-meta trade: one quantity, two replicates, and the disagreement
       between them was read as signal across M18 and M21.
    2. **The weighted mean laundered precision.** Sum(w_i^2) = 0.815 gives an
       effective n of 147 for 480 games spent — 69% of the compute wasted — and an
       MDE of 16.3pp. B2 shipped on a 4.4pp edge from this instrument (z=0.68, 11%
       power), and its two seeds spanned 8.2pp, which is 1.00 SD of what the
       instrument produces by chance.

    So: drop any snapshot deck identical to the deck `a` pilots (the mirror gate
    covers it properly), drop archetypes seeded on fewer than `min_games` observed
    games (meta_v2's 4th deck came from a SINGLE game), and spend the whole budget
    on what is left — the only part of the field mirror cannot see.

    Reports a PANEL with per-cell CIs, not a weighted scalar. `tail_score` is
    returned but is explicitly only over the surviving decks; `coverage` says what
    fraction of the frozen field that is. Never quote tail_score as a whole-field
    number.
    """
    import math

    from rl.matchrunner import (parse_spec, resolve_deck, run_pairs, series_wr,
                                spec_deck)

    snap = Path(snapshot) if snapshot else _latest_snapshot()
    manifest = json.loads((snap / "manifest.json").read_text())
    decks = manifest["decks"]
    spec_a = parse_spec(a)
    total_w = sum(d["weight"] for d in decks) or 1.0

    try:
        own = tuple(sorted(resolve_deck(spec_deck(spec_a))))
    except (OSError, ValueError, TypeError, IndexError):
        own = None

    keep, dropped = [], []
    for d in decks:
        try:
            same = own is not None and tuple(sorted(resolve_deck(snap / d["csv"]))) == own
        except (OSError, ValueError):
            same = False
        if exclude_mirror and same:
            dropped.append((d, "IS the mirror deck — mirror gate measures it at n=800"))
        elif d.get("n_games", 0) < min_games:
            dropped.append((d, f"only {d.get('n_games', 0)} observed games — sampling artifact"))
        else:
            keep.append(d)

    budget = n * len(decks)                      # same compute as the old form
    print(f"meta-eval (tail): {a} vs {snap.name}, seed={seed}, budget={budget} games",
          flush=True)
    for d, why in dropped:
        print(f"  DROPPED {d['archetype']:<40} (weight {d['weight'] / total_w:.3f}) — {why}",
              flush=True)
    if not keep:
        print("  no off-mirror archetypes survive — this snapshot measures only the "
              "mirror, which the mirror gate already covers.", flush=True)
        return {"snapshot": snap.name, "per_deck": {}, "tail_score": None,
                "coverage": 0.0, "dropped": [d["archetype"] for d, _ in dropped]}

    per = max(budget // len(keep), 1)             # equal split — panel, not a mean
    pairs = [(spec_a, ("solver", str(snap / d["csv"])), per) for d in keep]
    results = run_pairs(pairs, workers=workers, seed=seed, checkpoint=checkpoint)

    coverage = sum(d["weight"] for d in keep) / total_w
    keep_w = sum(d["weight"] for d in keep) or 1.0
    tail_score, per_deck = 0.0, {}
    for d, res in zip(keep, results):
        wr, m = series_wr(res), len(res)
        half = 1.96 * math.sqrt(0.25 / m) if m else 0.0
        tail_score += wr * d["weight"] / keep_w
        per_deck[d["archetype"]] = {"wr": wr, "n": m,
                                    "ci95": (max(0.0, wr - half), min(1.0, wr + half))}
        print(f"  vs {d['archetype']:<40} wr={wr:.3f}  "
              f"95%CI [{max(0.0, wr - half):.3f}, {min(1.0, wr + half):.3f}]  n={m}",
              flush=True)
    print(f"tail score: {tail_score:.3f}  — covers {coverage:.1%} of the FROZEN field; "
          f"the rest is the mirror matchup (use the mirror gate, n=800).", flush=True)
    print("  NB frozen weights understate the tail: the field drifted after the "
          "snapshot (solrock 0.892->0.761, kangaskhan 0.087->0.155, drakloak "
          "0.021->0.085; chi2=33.9, p=4.4e-08), so these archetypes are ~24% of "
          "the CURRENT field.", flush=True)
    print("Do NOT quote tail score as a whole-field number, and do not compare it "
          "across candidates below its MDE.", flush=True)
    return {"snapshot": snap.name, "per_deck": per_deck, "tail_score": tail_score,
            "coverage": coverage, "dropped": [d["archetype"] for d, _ in dropped]}


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="G0: convert everything, hard-fail on integrity errors")

    s = sub.add_parser("roundtrip", help="G1: same-code alignment gate (>=98%)")
    s.add_argument("-n", "--games", type=int, default=6)

    s = sub.add_parser("align", help="drift report: today's pilots vs old cached subs")
    s.add_argument("--subs", type=int, nargs="*", default=None)

    s = sub.add_parser("build", help="encode qualifying seats into data/bc_kaggle shards")
    s.add_argument("--out", type=Path, default=DATA_DIR)
    s.add_argument("--min-score", type=float, default=550.0)
    s.add_argument("--winners-only", action="store_true")
    s.add_argument("--include-ours", action="store_true")
    s.add_argument("--min-steps", type=int, default=0)
    s.add_argument("--shard-size", type=int, default=5000)
    s.add_argument("--only-subs", type=int, nargs="+", default=None,
                   help="clone-one-opponent: keep only these submission ids")
    s.add_argument("--deck-hash", type=str, default=None,
                   help="M24: keep only seats whose deck_hash starts with "
                        "this prefix (e.g. 20dcd313 = our lucario 60)")

    s = sub.add_parser("meta-eval", help="G4: candidate vs the frozen meta snapshot")
    s.add_argument("--a", required=True, help="matchrunner spec, e.g. model:<ckpt>:lucario")
    s.add_argument("--snapshot", type=Path, default=None)
    s.add_argument("-n", "--games", type=int, default=60)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--include-mirror", action="store_true",
                   help="keep snapshot decks identical to the deck `a` pilots. OFF by "
                        "default: that cell IS the mirror gate, measured there at n=800")
    s.add_argument("--min-games", type=int, default=3,
                   help="drop archetypes seeded on fewer observed games (meta_v2's "
                        "4th deck came from a single game)")

    a = p.parse_args()
    if a.cmd == "audit":
        audit()
    elif a.cmd == "roundtrip":
        align_roundtrip(n_games=a.games)
    elif a.cmd == "align":
        align(subs=set(a.subs) if a.subs else None)
    elif a.cmd == "build":
        build(out_dir=a.out, min_score=a.min_score, winners_only=a.winners_only,
              include_ours=a.include_ours, min_steps=a.min_steps,
              shard_size=a.shard_size,
              only_subs=set(a.only_subs) if a.only_subs else None,
              only_deck_hash=a.deck_hash)
    elif a.cmd == "meta-eval":
        meta_eval(a.a, snapshot=a.snapshot, n=a.games, workers=a.workers,
                  seed=a.seed, checkpoint=a.checkpoint,
                  exclude_mirror=not a.include_mirror, min_games=a.min_games)


if __name__ == "__main__":
    _main()
