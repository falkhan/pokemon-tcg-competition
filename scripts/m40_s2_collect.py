"""M40 S2 — self-play corpus collection, RESUMABLE.

Three things the M39 lane got wrong or could not do, fixed here.

1. NO RESUME. `plan_iter collect` has no resume at all — relaunching into the
   same --out clobbers existing shards (CLAUDE.md), so a crash or a pause costs
   the whole run and the workaround has been "collect the remainder into a
   FRESH dir and train on both". This collector checkpoints at CHUNK
   granularity: work is split into (bed, seed, game-range) chunks, each chunk's
   shards are written before its manifest entry is committed, and a restart
   replays the manifest and runs only what is missing. Interrupt it whenever.

2. UNIFORM NOISE IS NOT EXPLORATION. M39's `bestresp` explored by replacing the
   chosen action with a UNIFORM draw over all legal options at eps=0.08 on
   single-pick MAIN prompts. That produced a corpus the net already agreed with
   93.5% of the time, because ~95% of rows were plain greedy and the 5% that
   were not were mostly nonsense the net would never consider. This collector
   samples from the policy instead — softmax temperature, or epsilon over the
   TOP-K — so a deviation is a plausible alternative line rather than a random
   one, and the rate can be swept until agreement drops.

3. THE KILL IS AT COLLECTION, NOT AFTER TRAINING. The plan pre-registers
   "init val_acc > 0.90 kills the corpus before training". That has to be
   checked HERE: `runs/m39_retain_b.log` shows init val_acc 0.907 on the
   milestone's BEST net, so the same gate applied at the training entry would
   have killed it. The collection-time equivalent is AGREEMENT — how often the
   recorded action equals what the arm would have played anyway — which is
   measurable per chunk and is what init val_acc is proxying for.

The arm's own rules stay in the loop: the base action is whatever the full
pilot (net + fix stack) plays, and exploration is a deviation FROM that, so the
corpus is on-distribution for the agent we actually ship.

Usage:
    # start (or resume — same command, always safe)
    uv run python scripts/m40_s2_collect.py --out data/m40_s2_a --games 400 \
        --beds wall_d1 grim_d1 arch_d1 --tau 0.6

    # inspect progress without running anything
    uv run python scripts/m40_s2_collect.py --out data/m40_s2_a --status
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import SelectContext, to_observation_class  # noqa: E402
from rl.encoders import (encode_context, encode_option_v2,  # noqa: E402
                         encode_state_v3)
import rl.matchrunner as mr  # noqa: E402

# S2 step 3 — opponent diversity beyond three clones per family. The pool spans
# panel DRAWS as well as families, because G-13 showed a single draw is one
# lottery ticket and a corpus manufactured against one draw teaches lines that
# only beat that draw. `comp_*` are S3 composites: strictly stronger opponents
# (+125 ELO over the clone they wrap, measured), and the one legitimate place
# solver strength enters training data — as an OPPONENT, never as a teacher.
BEDS = {
    # mirror family — 19.4% of the live mix, and arm 1's decode showed it
    # FALLING when left out of the pool (m28 −3.0 z−2.11, mirror −2.2).
    "mirror":   "model:checkpoints/m28_winners.pt:alakazam_v2_h4",
    "m28":      "model:checkpoints/m28_winners.pt:clone54618168",
    "wall_d1":  "model:checkpoints/m38_bc_wall.pt:greattusk_wall",
    "wall_d2":  "model:checkpoints/m39_bed_wall_d2.pt:greattusk_wall",
    "wall_d3":  "model:checkpoints/m39_bed_wall_d3.pt:greattusk_wall",
    "grim_d1":  "model:checkpoints/m39_bc_grim.pt:grim_live",
    "grim_d2":  "model:checkpoints/m39_bc_grim_b.pt:grim_live",
    "grim_d3":  "model:checkpoints/m39_bed_grim_d3.pt:grim_live",
    "arch_d1":  "model:checkpoints/m39_bc_archaludon.pt:archaludon",
    "arch_d2":  "model:checkpoints/m39_bed_arch_d2.pt:archaludon",
    "arch_d3":  "model:checkpoints/m39_bed_arch_d3.pt:archaludon",
    "topgrim":  "model:checkpoints/m39_bc_topgrim.pt:grim_live",
    "topgrim_d2": "model:checkpoints/m40_bed_topgrim_d2.pt:grim_live",
    "topgrim_d3": "model:checkpoints/m40_bed_topgrim_d3.pt:grim_live",
    # M40 phase 2 — the live LOSS FAMILIES, cloned from 900-1150 seats
    # (2026-08-03). NOT live-faithful (positive control failed: they read
    # 0.56-0.90 vs live truth 0.08-0.25) but real, diverse opponents that
    # widen the pool beyond wall/grim/arch. d3 draws are the pre-registered
    # HELD-OUTS — keep them out of --beds so the gate can see overfit.
    "dragapult_d1": "model:checkpoints/m40_bed_dragapult_d1.pt:data/kaggle/dragapult_3631d393_deck.csv",
    "dragapult_d2": "model:checkpoints/m40_bed_dragapult_d2.pt:data/kaggle/dragapult_3631d393_deck.csv",
    "garchomp_d1": "model:checkpoints/m40_bed_garchomp_d1.pt:data/kaggle/garchomp_c7b3253f_deck.csv",
    "garchomp_d2": "model:checkpoints/m40_bed_garchomp_d2.pt:data/kaggle/garchomp_c7b3253f_deck.csv",
    "rocket_d1": "model:checkpoints/m40_bed_rocket_d1.pt:data/kaggle/rocket_59e27a5e_deck.csv",
    "rocket_d2": "model:checkpoints/m40_bed_rocket_d2.pt:data/kaggle/rocket_59e27a5e_deck.csv",
    # M43 Lane B: the ogerpon self-play mirror — the pre-registered B1 pool
    # (docs/M43-plan.md B1: "grim_d1–d3, topgrim, wall_d1–d3, ogerpon mirror")
    # names it, but it was never added here. `model-pz` matches the live
    # ogerpon ship config (sub 55265105), same on-distribution rationale as
    # DEFAULT_ARM's `model-c-pkgz`.
    "oger_mirror": "model-pz:checkpoints/m41_ogerpon.pt:decks/ogerpon.csv",
    # composites — budget is per-spec since M40 S3. Phase 3 measured the wrap
    # INERT against our pilot (comp ≈ plain on the grim grid), so composites
    # earn no place in the pool; kept only as specs for instrument work.
    "comp_grim": "solved:checkpoints/m39_bc_grim.pt:grim_live:800:400",
    "comp_wall": "solved:checkpoints/m38_bc_wall.pt:greattusk_wall:800:400",
}
# `model-c-pkgz`, not `model-c-pkg`: S6 adopted `planzero`, so the next net
# ship serves a ZERO plan vector. Two reasons this has to match here.
# (1) The corpus should be on-distribution for the agent we actually ship —
#     collecting under a pilot that serves a non-zero plan would manufacture
#     the same train/serve mismatch S6 spent the milestone removing.
# (2) The sampler below reads logits at plan=0. With a `model-c-pkg` arm the
#     BASE action would come from a non-zero-plan ranking while the DEVIATION
#     came from a zero-plan one, so exploration would be measured against a
#     distribution the pilot never used.
DEFAULT_ARM = "model-c-pkgz:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
DECK_IDX = 9000
# The plan vector the SAMPLER reads logits at. The startup parity guard
# asserts the ARM serves exactly this at every prompt, so the two policies
# the collector compares (base action vs deviation) are the same policy.
# Single source: if the sampler ever changes, the guard follows automatically.
SAMPLER_PLAN = np.zeros(27, dtype=np.float32)
ENC_VER = 3  # encode_state_v3 below; bump WITH the encoder calls, G-14 6b reads it
# Cheap, always-on-disk opponent for the parity probe — independent of --beds
# so a composite-only pool does not make the probe pay solver latency.
PROBE_BED = "model:checkpoints/m28_winners.pt:alakazam_v2_h4"
COLUMNS = ("states", "state_ids", "options", "option_ids", "n_options",
           "labels", "game_ids", "results", "deck_idx", "teacher_score",
           "seat_won", "episode_ids")
MANIFEST = "s2_manifest.json"


# --------------------------------------------------------------------------
# manifest — the resume contract
# --------------------------------------------------------------------------
def load_manifest(out: Path) -> dict:
    p = out / MANIFEST
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"config": None, "chunks": {}}


def save_manifest(out: Path, man: dict) -> None:
    """Atomic: a crash mid-write must not leave an unreadable manifest, or the
    resume contract is worse than no resume at all."""
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmp, out / MANIFEST)


def config_fingerprint(a) -> dict:
    """Everything that changes what a row MEANS. Resuming across a change to
    any of these would silently blend two different corpora into one dir, which
    is exactly the kind of quiet contamination this milestone keeps finding."""
    return {"arm": a.arm, "deck": a.deck, "tau": a.tau, "eps": a.eps,
            "top_k": a.top_k, "games_per_chunk": a.chunk_games}


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------
def make_sampler(a, rng):
    """Return pick(logits, base_action) -> (action, explored).

    Two designs, both operating on the POLICY's own ranking rather than on the
    uniform distribution over legal options:

      --tau T      softmax(logits / T), the principled knob; sweep T until
                   agreement lands materially below 0.90.
      --eps E      with probability E, take a uniform draw from the TOP-K
      --top-k K    options. Cheaper to reason about and bounded: an explored
                   action is always one the policy ranked plausibly, so the
                   corpus stays on-distribution.

    Both are OFF by default, which yields pure greedy — useful for measuring
    the agreement floor of an unexplored corpus before choosing a rate.
    """
    def pick(logits, base):
        if a.tau > 0:
            z = np.asarray(logits, dtype=np.float64) / a.tau
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            idx = int(rng.choices(range(len(p)), weights=p, k=1)[0])
            return idx, idx != base
        if a.eps > 0 and rng.random() < a.eps:
            k = min(a.top_k, len(logits))
            top = np.argsort(np.asarray(logits))[::-1][:k]
            idx = int(top[rng.randrange(k)])
            return idx, idx != base
        return base, False
    return pick


# --------------------------------------------------------------------------
# collection-time parity guard
# --------------------------------------------------------------------------
def parity_verdict(rec: dict) -> tuple[bool, str]:
    """Decide the guard from a probe census. Pure so the decision boundary is
    testable without engine games.

    Fails on a plan mismatch, and fails on a probe that observed nothing —
    a guard that can pass vacuously is the S6-mechanism-probe mistake again.
    """
    if rec.get("prompts", 0) == 0:
        return False, ("probe observed 0 prompts — harness defect, not a "
                       "pass (the instrumented pilot never acted)")
    if rec.get("plan_mismatch_prompts", 0):
        return False, (f"arm served a plan != SAMPLER_PLAN on "
                       f"{rec['plan_mismatch_prompts']}/{rec['prompts']} "
                       f"prompts — the base action and the sampled deviation "
                       f"would come from two different policies")
    return True, "arm serves SAMPLER_PLAN at every prompt"


def verify_plan_parity(a, n_games: int = 2) -> dict:
    """Collection-time G-14. This collector's first outing manufactured the
    exact train/serve mismatch S6 had removed that morning (diary 2026-08-02):
    the default arm served a NON-ZERO plan while the sampler read logits at
    plan=0. G-14 catches that class at ship time; nothing caught it at
    collection time — this does, before a single chunk is spent.

    Behavioural, not static: plays n_games of the ARM vs a cheap fixed bed
    with the arm's OptionScorerV3 subclassed to record the literal plan vector
    it is fed (the m40_planzero_probe pattern — patch around ONE make_pilot
    call so the bed side stays pristine).
    """
    import rl.plan as rp
    import rl.policy as rl_policy

    rec = {"prompts": 0, "plan_mismatch_prompts": 0, "plan_enumerations": 0,
           "games": n_games}
    real_enum = rp.enumerate_plans
    real_cls = rl_policy.OptionScorerV3

    def counting_enum(obs):
        rec["plan_enumerations"] += 1
        return real_enum(obs)

    class _Recording(real_cls):
        def act(self, state_ctx, plan, state_ids, options, option_ids, k,
                greedy=True):
            rec["prompts"] += 1
            served = np.asarray(plan, dtype=np.float32).ravel()
            if not np.array_equal(served, SAMPLER_PLAN):
                rec["plan_mismatch_prompts"] += 1
            return super().act(state_ctx, plan, state_ids, options,
                               option_ids, k, greedy=greedy)

    rp.enumerate_plans = counting_enum
    rl_policy.OptionScorerV3 = _Recording
    try:
        fn_a, deck_a = mr.make_pilot(mr.parse_spec(a.arm), instance="parity_a")
    finally:
        rp.enumerate_plans = real_enum
        rl_policy.OptionScorerV3 = real_cls
    fn_b, deck_b = mr.make_pilot(mr.parse_spec(PROBE_BED), instance="parity_b")

    for g in range(n_games):
        fns = (fn_a, fn_b) if g % 2 == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if g % 2 == 0 else (deck_b, deck_a)
        mr._engine_game(fns[0], fns[1], decks[0], decks[1])
    return rec


# --------------------------------------------------------------------------
# one chunk
# --------------------------------------------------------------------------
def run_chunk(a, bed: str, seed: int, n_games: int, game_id0: int,
              out: Path, tag: str) -> dict:
    """Play n_games of the arm vs `bed`, write this chunk's shards, return its
    manifest entry. Writes NOTHING outside this chunk's own files, so a crash
    can only ever lose the chunk in flight."""
    import torch
    from rl.policy import OptionScorerV3, option_dim_of

    rng = random.Random((seed * 7919) ^ hash(tag) & 0xFFFFFFFF)
    sampler = make_sampler(a, rng)
    deck_ids = mr.resolve_deck(a.deck)

    # A private copy of the arm's net, so we can read LOGITS. The pilot itself
    # only returns picks, and we need the distribution to sample from it.
    ckpt = a.arm.split(":")[1]
    sd = torch.load(ROOT / ckpt if not Path(ckpt).is_absolute() else ckpt,
                    map_location="cpu")
    net = OptionScorerV3(n_state_ids=20, option_dim=option_dim_of(sd))
    net.load_state_dict(sd)
    net.eval()

    stats = {"games": 0, "wins": 0, "rows": 0, "prompts": 0, "explored": 0,
             "agree": 0}
    shard = {k: [] for k in COLUMNS}
    pending: list[tuple] = []
    orig_make = mr.make_pilot

    def recording_make(spec, instance):
        fn, ids = orig_make(spec, instance)
        if not instance.endswith("_a"):
            return fn, ids

        def wrapped(od):
            obs = to_observation_class(od)
            if obs.select is None:
                return fn(od)
            base_picks = [int(i) for i in fn(od)]      # the ARM's real action,
            base = base_picks[0]                        # rules included
            state_num, state_ids = encode_state_v3(obs.current, deck_ids)
            state_ctx = np.concatenate(
                [state_num, encode_context(obs.select.context)]
            ).astype(np.float32)
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([x for x, _ in pairs]).astype(np.float32)
            oids = np.stack([i for _, i in pairs])
            action = base
            explorable = (obs.select.maxCount == 1
                          and len(obs.select.option) >= 2
                          and obs.select.context == SelectContext.MAIN)
            if explorable:
                with torch.no_grad():
                    logits = net(
                        torch.from_numpy(state_ctx).unsqueeze(0),
                        torch.from_numpy(SAMPLER_PLAN).unsqueeze(0),
                        torch.from_numpy(state_ids.astype(np.int64)).unsqueeze(0),
                        torch.from_numpy(opts).unsqueeze(0),
                        torch.from_numpy(oids.astype(np.int64)).unsqueeze(0),
                    )[0].squeeze(0).numpy()
                action, did = sampler(logits, base)
                stats["explored"] += int(did)
            stats["prompts"] += 1
            stats["agree"] += int(action == base)
            pending.append((state_ctx, state_ids, opts, oids, action))
            return [action] + base_picks[1:]
        return wrapped, ids

    def on_game(g, result, seat_stats):
        won = result == 0
        stats["games"] += 1
        stats["wins"] += int(won)
        res = 0.0 if result == 2 else (1.0 if won else -1.0)
        for sc, sids, opts, oids, label in pending:
            shard["states"].append(sc)
            shard["state_ids"].append(sids)
            shard["options"].append(opts)
            shard["option_ids"].append(oids)
            shard["n_options"].append(len(opts))
            shard["labels"].append(label)
            shard["game_ids"].append(game_id0 + stats["games"] - 1)
            shard["results"].append(res)
            shard["deck_idx"].append(DECK_IDX)
            shard["teacher_score"].append(0.0)
            shard["seat_won"].append(1.0 if won else 0.0)
            shard["episode_ids"].append(-1)
        stats["rows"] += len(pending)
        pending.clear()

    mr.make_pilot = recording_make
    try:
        mr.play_series(mr.parse_spec(a.arm), mr.parse_spec(BEDS[bed]),
                       n_games, seed=seed, on_game=on_game)
    finally:
        mr.make_pilot = orig_make

    files = []
    if shard["labels"]:
        f = out / f"shard_{tag}.npz"
        np.savez_compressed(
            f,
            states=np.stack(shard["states"]).astype(np.float32),
            state_ids=np.stack(shard["state_ids"]).astype(np.int32),
            options=np.concatenate(shard["options"]).astype(np.float32),
            option_ids=np.concatenate(shard["option_ids"]).astype(np.int32),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            labels=np.array(shard["labels"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
            teacher_score=np.array(shard["teacher_score"], dtype=np.float32),
            seat_won=np.array(shard["seat_won"], dtype=np.float32),
            episode_ids=np.array(shard["episode_ids"], dtype=np.int64),
        )
        files.append(f.name)
    stats["shards"] = files
    stats["bed"] = bed
    stats["seed"] = seed
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beds", nargs="+", default=["wall_d1", "grim_d1", "arch_d1"],
                    choices=sorted(BEDS))
    ap.add_argument("--games", type=int, default=400, help="games PER BED")
    ap.add_argument("--chunk-games", type=int, default=50,
                    help="resume granularity; a crash loses at most this many")
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--deck", default="alakazam_v2_h4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--kill-above", type=float, default=0.90,
                    help="PRE-REGISTERED: abort if agreement exceeds this once "
                         "--kill-after prompts are in. Resume after changing "
                         "the sampler into a FRESH --out.")
    ap.add_argument("--kill-after", type=int, default=20000)
    ap.add_argument("--status", action="store_true",
                    help="print progress and exit without collecting")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(a.out)
    fp = config_fingerprint(a)
    if man["config"] is None:
        man["config"] = fp
        save_manifest(a.out, man)
    elif man["config"] != fp and not a.status:
        print("REFUSING TO RESUME: this dir was collected under a different "
              "config.\n  on disk: " + json.dumps(man["config"])
              + "\n  now    : " + json.dumps(fp)
              + "\nMixing them would blend two corpora in one dir. Use a fresh "
                "--out.")
        return 2

    plan = [(bed, s, min(a.chunk_games, a.games - s))
            for bed in a.beds for s in range(0, a.games, a.chunk_games)]
    done = man["chunks"]

    def totals():
        t = {"games": 0, "rows": 0, "prompts": 0, "agree": 0, "explored": 0,
             "wins": 0}
        for e in done.values():
            for k in t:
                t[k] += e.get(k, 0)
        return t

    if a.status:
        t = totals()
        print(f"{a.out}: {len(done)}/{len(plan)} chunks, {t['games']} games, "
              f"{t['rows']} rows")
        if t["prompts"]:
            print(f"  agreement {t['agree']/t['prompts']:.4f}  "
                  f"explored {t['explored']/t['prompts']:.4f}  "
                  f"WR {t['wins']/max(t['games'],1):.3f}")
        for bed in a.beds:
            n = sum(1 for k in done if k.startswith(bed + ":"))
            print(f"  {bed:<12} {n}/{len([1 for b,_,_ in plan if b==bed])} chunks")
        return 0

    todo = [(b, s, n) for b, s, n in plan if f"{b}:{a.seed}:{s}" not in done]
    print(f"{a.out}: {len(done)}/{len(plan)} chunks done, {len(todo)} to run "
          f"(chunk={a.chunk_games} games)")
    if not todo:
        print("nothing to do — collection complete")
        return 0

    # Collection-time G-14: verified on EVERY invocation that will collect —
    # the arm's serve path can change between resumes, and the probe is a few
    # seconds against hours of collection. Informational record goes to the
    # manifest but NOT into config_fingerprint (that would refuse to resume
    # every pre-guard dir for a check that changes no row's meaning).
    rec = verify_plan_parity(a)
    ok, reason = parity_verdict(rec)
    man["parity"] = {**rec, "enc_ver": ENC_VER, "ok": ok,
                     "checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_manifest(a.out, man)
    print(f"[parity] {reason}  "
          f"(prompts {rec['prompts']}, plan_enum {rec['plan_enumerations']})")
    if not ok:
        print("PARITY GUARD: refusing to collect — fix the arm spec (or the "
              "sampler) so both sides read the same plan distribution.")
        return 4

    for bed, start, n in todo:
        tag = f"{bed}_{a.seed}_{start:05d}"
        key = f"{bed}:{a.seed}:{start}"
        t0 = time.time()
        # game_id must be globally unique across chunks or BCDatasetV3's
        # by-game val split will straddle chunks and leak.
        game_id0 = 1_000_000 + abs(hash(key)) % 900_000
        print(f"[s2] {key}  {n} games  {time.strftime('%H:%M:%S')}", flush=True)
        entry = run_chunk(a, bed, a.seed, n, game_id0, a.out, tag)
        entry["secs"] = round(time.time() - t0, 1)
        done[key] = entry
        save_manifest(a.out, man)          # committed only after the shard
        t = totals()
        agree = t["agree"] / max(t["prompts"], 1)
        print(f"     rows {entry['rows']:>6}  WR {entry['wins']/max(entry['games'],1):.3f}"
              f"  chunk-agree {entry['agree']/max(entry['prompts'],1):.4f}"
              f"  CUM-agree {agree:.4f}  ({entry['secs']}s)", flush=True)
        if t["prompts"] >= a.kill_after and agree > a.kill_above:
            print(f"\nPRE-REGISTERED KILL: agreement {agree:.4f} > "
                  f"{a.kill_above} after {t['prompts']} prompts.")
            print("The exploration design failed again (M39's bestresp read "
                  "0.935). The corpus is dead HERE, before training —")
            print("raise --tau or --eps and collect into a FRESH --out. "
                  "Progress so far is committed and inspectable.")
            return 3

    t = totals()
    print(f"\ndone: {t['games']} games, {t['rows']} rows, "
          f"agreement {t['agree']/max(t['prompts'],1):.4f}, "
          f"explored {t['explored']/max(t['prompts'],1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
