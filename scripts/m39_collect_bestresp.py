"""M39 P3 corpus B — offline best-response against the clone beds.

The plan (BACKLOG sweep #2, decision 6) specified this as "`plan_iter collect`
against the beds, keep the winning seats, low-dose fine-tune — no new
infrastructure". **That specification does not work, and the reason matters
enough to record here rather than silently work around.**

`plan_iter collect` labels every row with `_teacher_step` — the SOLVER
teacher — in both `expert` and `ei` mode; in `ei` mode the student only
*executes*, it never *labels* ([plan_iter.py:388](../rl/plan_iter.py:388)).
A corpus collected that way is a solver-teacher corpus, and M38's central
negative result is that solver-teacher labels cannot beat the winners lineage
at any dose (.123-.264 against a .517-.631 control). Using it here would have
re-run a known kill under a new name.

What best-response actually needs is the opposite label source: OUR net's own
executed actions, in the games it WON. That is this script.

Method, and its one real limitation:

  * play the arm against a fixed bed, recording every own-seat decision with
    the same encoder calls `rl/replay_bc.encode_decisions` uses, so corpus B
    is schema- and encoding-identical to corpus A and to the champion shards;
  * with probability --eps, substitute a uniformly random legal option on a
    single-pick MAIN prompt and record THAT. Without exploration the labels
    are the net's own argmax, the cross-entropy gradient at those rows is
    already near zero, and the "fine-tune" would only re-weight states — a
    much weaker experiment than the one the plan intended. Exploration plus
    an outcome filter is the mechanism that lets imitation exceed the policy
    it imitates;
  * write `results` per row (+1 / -1 / 0 for the recorded seat) rather than
    filtering here, so `plan_iter train --outcome-weight 0` performs the
    winners-only filter at train time — the same knob, and the same corpus,
    also supports the alpha ladder.

LIMITATION, stated up front: this is *filtered self-imitation*, the offline
approximation of best-response, not best-response itself (that needs a policy
gradient against the bed -- BACKLOG sweep #7, ~327M env steps). The ceiling is
our own net's best play under exploration, which is why corpus A (whose
demonstrators reach 1058) runs in parallel. Which one wins says which
constraint was binding.

Usage:
    uv run python scripts/m39_collect_bestresp.py --bed wall -n 600
    uv run python scripts/m39_collect_bestresp.py --bed grim -n 600 --seed 2
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cg.api import SelectContext, to_observation_class  # noqa: E402
from rl.encoders import (encode_context, encode_option_v2,  # noqa: E402
                         encode_state_v3)
import rl.matchrunner as mr  # noqa: E402

# The three beds carrying the deck-race loss mass the P3 lane targets. The
# arm plays its OWN list against each, seat-rotated by play_series.
BEDS = {
    "wall": "model:checkpoints/m38_bc_wall.pt:greattusk_wall",
    "grim": "model:checkpoints/m39_bc_grim.pt:grim_live",
    "archaludon": "model:checkpoints/m39_bc_archaludon.pt:archaludon",
    # panel draws, so corpus B is not collected against one lottery ticket
    # either (G-13): a corpus manufactured against a single weak draw would
    # teach lines that only beat that draw.
    "wall_d2": "model:checkpoints/m39_bed_wall_d2.pt:greattusk_wall",
    "grim_d2": "model:checkpoints/m39_bc_grim_b.pt:grim_live",
    "arch_d2": "model:checkpoints/m39_bed_arch_d2.pt:archaludon",
}
ARM = "model-conserve:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
DECK_IDX = 9000        # own bucket, clear of REPLAY_DECK_BASE and the beds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default="wall", choices=sorted(BEDS))
    ap.add_argument("-n", "--games", type=int, default=600)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.08,
                    help="exploration rate on single-pick MAIN prompts")
    ap.add_argument("--shard-size", type=int, default=5000)
    args = ap.parse_args()

    out_dir = args.out or ROOT / f"data/bc_m39_br_{args.bed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed * 7919)

    deck_ids = mr.resolve_deck("alakazam_v2_h4")
    pending: list[tuple] = []      # rows of the game in progress
    columns = ("states", "state_ids", "options", "option_ids", "n_options",
               "labels", "game_ids", "results", "deck_idx", "teacher_score",
               "seat_won", "episode_ids")
    shard = {k: [] for k in columns}
    counts = {"games": 0, "wins": 0, "rows": 0, "explored": 0, "shards": 0,
              "game_id": 0}

    def flush():
        if not shard["labels"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{counts['shards']:04d}.npz",
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
        counts["shards"] += 1
        for v in shard.values():
            v.clear()

    # --- record side a's own decisions -----------------------------------
    orig_make = mr.make_pilot

    def recording_make(spec, instance):
        fn, ids = orig_make(spec, instance)
        if not instance.endswith("_a"):
            return fn, ids

        def wrapped(od):
            obs = to_observation_class(od)
            if obs.select is None:      # deck request / game boundary
                return fn(od)
            picks = fn(od)
            picks = [int(i) for i in picks]
            n_opt = len(obs.select.option)
            if (args.eps > 0 and obs.select.maxCount == 1 and n_opt >= 2
                    and obs.select.context == SelectContext.MAIN
                    and rng.random() < args.eps):
                picks = [rng.randrange(n_opt)]
                counts["explored"] += 1
            state_num, state_ids = encode_state_v3(obs.current, deck_ids)
            state_ctx = np.concatenate(
                [state_num, encode_context(obs.select.context)]
            ).astype(np.float32)
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            pending.append((state_ctx, state_ids,
                            np.stack([x for x, _ in pairs]).astype(np.float32),
                            np.stack([i for _, i in pairs]), picks[0]))
            return picks
        return wrapped, ids

    def on_game(g, result, seat_stats):
        won = result == 0
        counts["games"] += 1
        counts["wins"] += won
        res = 0.0 if result == 2 else (1.0 if won else -1.0)
        for state_ctx, state_ids, opts, opt_ids, label in pending:
            shard["states"].append(state_ctx)
            shard["state_ids"].append(state_ids)
            shard["options"].append(opts)
            shard["option_ids"].append(opt_ids)
            shard["n_options"].append(len(opts))
            shard["labels"].append(label)
            shard["game_ids"].append(counts["game_id"])
            shard["results"].append(res)
            shard["deck_idx"].append(DECK_IDX)
            # teacher_score / seat_won / episode_ids exist so corpus B is
            # schema-identical to the harvested corpora and the two can be
            # mixed by `train --data dirA dirB` (the P3-C retention arm).
            # There is no leaderboard score behind a manufactured seat, so
            # it is recorded as 0.0 rather than as a plausible-looking number.
            shard["teacher_score"].append(0.0)
            shard["seat_won"].append(1.0 if won else 0.0)
            shard["episode_ids"].append(-1)
        counts["rows"] += len(pending)
        counts["game_id"] += 1
        pending.clear()
        if len(shard["labels"]) >= args.shard_size:
            flush()
        if counts["games"] % 100 == 0:
            print(f"  game {counts['games']}/{args.games}  "
                  f"wr={counts['wins'] / counts['games']:.3f}  "
                  f"rows={counts['rows']}", flush=True)

    mr.make_pilot = recording_make
    try:
        mr.play_series(mr.parse_spec(ARM), mr.parse_spec(BEDS[args.bed]),
                       args.games, seed=args.seed, on_game=on_game)
    finally:
        mr.make_pilot = orig_make
    flush()

    meta = {"arm": ARM, "bed": BEDS[args.bed], "games": counts["games"],
            "wins": counts["wins"], "rows": counts["rows"],
            "explored": counts["explored"], "eps": args.eps,
            "seed": args.seed, "deck_idx": DECK_IDX}
    (out_dir / "collect_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ncollected {counts['rows']} rows over {counts['games']} games "
          f"(wr {counts['wins'] / max(counts['games'], 1):.3f}, "
          f"{counts['explored']} exploratory picks) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
