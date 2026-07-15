# M9 plan: beat solver:lucario 0.55 — parallel pilot-fix + learning ladder

Status: **planned 2026-07-14, not started.** Diary: `docs/M9.md` (create on start, one dated
entry per leg including kills, status checklist at top). Bar unchanged from M8: **any agent
≥0.55 vs `solver:lucario`, n≥800.**

## Why this ladder

M8 closed at 0.359 (`ppo_m83_legB_it5.pt`); BC base `osv2_bc2.pt` 0.345. The 07-14 pipeline
rerun (`ppo_run.log`/`ppo_run2.log`) reproduced the known pathology — PPO peaks iter 4
(34.5%) then decays below the 28.5% baseline. Two levers are **measured dead ends** and are
not in this plan:

- More PPO iterations/epochs on the current loop (M8.3: gains within noise even with entropy
  cured; reproduced 07-14).
- Inference-time MCTS on the current value head (M8.4: sims ladder flat 0.515/0.490/0.495 at
  16/32/64 — the head is a win/loss classifier that can't *rank* sibling actions). M8.1
  dev-tier solver also stays killed (pooled 0.492 n=1600).

Live levers: (1) the BC teacher (`generic`) is strictly weaker than the eval target
(generic + turn_solver); (2) plain BC never sees the student's own states (no DAgger in the
codebase); (3) the target's own pilot has two documented blind spots (M8.0 taxonomy: hand-
discard burned t1 vs passing 38/100, Boss's Orders hoarded 21/100) — and the bar accepts any
agent; (4) value-as-ranker + two gated RL-core probes attack the documented failure
mechanisms directly.

**Key framing:** a perfect clone of the target mirrors it at 0.50 — imitation legs gate on
**progress (+3pp)**; only the pilot-fix and beyond-teacher legs gate on 0.55.

**Teacher choice rationale (asked and settled 2026-07-14):** the Lucario rule pilot
(`sample-agent`) was the original BC teacher and is measurably worse — 55% fidelity ceiling
because its edge is hidden `AttackPlan` lookahead not present in the observation (clone won
38% vs the 90%-fidelity generic clone; adding combat-lookahead features moved fidelity only
to 56.4% and *lowered* win rate — DECISIONS.md 2026-07-08 ×2). Rule: pick the strongest
teacher whose reasoning is **observable** — the solver (Legs 2–3), then pilot v2 if Leg 1
promotes. The rule-pilot teacher is strictly dominated and stays retired.

## Measurement protocol (all legs)

- Canonical run: `python -m rl.matchrunner play --a <candidate> --b solver:lucario -n 800
  --workers 12 --seed <S> --checkpoint runs/m9_<leg>.jsonl`
- Screen n=400 one seed → past go-gate → confirm n=800 fresh seed. **Promotion:** pooled
  n≥1200 wr ≥0.55 AND each seed >0.52. Floors on promotion: `random:kyogre` n=200 × 2 seeds
  ≥0.90; `rule:lucario` n=400 no regression vs 0.362; latency p99 sane.
- Pinned baselines: osv2_bc2 0.345 · ppo_m83_legB_it5 0.359 · solver mirror 0.500.
- Hygiene: **no edits to `rl/` modules while any collection/eval is live** (workers re-import
  from disk); one dated `docs/M9.md` entry per leg including kills, with exact command/n/seed.
- Checkpoint names pinned per leg: `osv2_dagger1.pt`, `osv2_soldis1.pt`, `bc_v1_rank.pt`,
  `ppo_m9_kl.pt`.

## Session 1 — parallel: Leg 1 (pilot v2) + Leg 2 collection (DAgger)

### Leg 1: pilot v2 — hand-discard + Boss's Orders fixes (~2h, can clear the bar outright)

**Freeze the eval target.** `solver:lucario` is built at runtime from `rl/generic_pilot.py`
+ `rl/turn_solver.py` (`rl/matchrunner.py:94-97`); editing `score_play` in place improves
the opponent too and breaks the byte-parity pin vs `tcg/pilot.py`. All fixes go behind a
default-off flag:

1. Thread `fixes: frozenset = frozenset()` through `make_generic_pilot` (`rl/generic_pilot.py:131`)
   and `make_solver_pilot` (`rl/turn_solver.py:348`). Default-off keeps existing specs and
   the test suite byte-identical.
2. **Fix A (hand-discard):** in `score_play` (`rl/generic_pilot.py:284-285`), when
   `"handdiscard" in fixes` and `_hand_holds_keepers(me)`, return **-100** instead of 150 —
   below END, so Carmine is never burned t1 against passing.
3. **Fix B (gust):** add `_GUST_TRAINER_NAMES = ("Boss's Orders",)`; in `score_play`, when
   the card is a gust trainer and the opp bench has a target my board KOs faster than their
   active (reuse `_turns_to_first_ko`/`_best_damage` from `rl/combat`), score ~2450 (above
   trainer taper, below KO tier). Target pick already works via `EFFECT_TARGET` scoring.
4. New specs `generic2:`/`solver2:` in `parse_spec`/`make_pilot` (`rl/matchrunner.py:65,90-101`)
   passing `fixes={"handdiscard","gust"}`. Keep the two fixes separable flags for attribution.
5. Mirror into the twins `tcg/pilot.py`/`tcg/constants.py` only after measurement.

**Gates:** smoke n=20 zero-crash → `solver2:lucario` vs `solver:lucario` n=400 (go ≥0.53) →
n=800 fresh seed. **Kill** pooled <0.52 (try Fix A alone before killing). On promotion:
pilot v2 becomes the teacher for Leg 3 / a rerun of Leg 2.

### Leg 2: DAgger — student states, solver labels (collection ~40-60 min, runs alongside Leg 1 evals)

1. **Collect:** new `collect_dagger(n_games, checkpoint, decks_file, out_dir=data/bc_dagger,
   seed)` in `rl/bc.py`, cloned from `collect_games_v2` (`bc.py:145`) with one change —
   **picks from the student, labels from the teacher.** Student per seat via
   `matchrunner.make_pilot(("model", checkpoint, deck_ids), instance)` (`matchrunner.py:138-155`);
   teacher via existing `_teacher_pilot("solver", ...)` (`bc.py:131-142`). Per decision:
   label = teacher's first pick on the same obs; the *student's* (temperature-sampled, not
   argmax) pick advances the game. CLI: `python -m rl.bc collect --teacher dagger
   --checkpoint checkpoints/osv2_bc2.pt --games 1000 --out data/bc_dagger`.
2. **Train:** extend `BCDatasetV2.__init__` (`bc.py:330`) to accept a list of shard dirs;
   fine-tune on `[data/bc_v2b, data/bc_dagger]` (~3:1) with new `--init
   checkpoints/osv2_bc2.pt` warm-start in `train_v2` (`bc.py:390`), lr 1e-4, 4-6 epochs →
   `osv2_dagger1.pt`.
3. **Gate on win rate, never fidelity:** n=400 vs solver, go ≥0.38 → n=800 confirm.
   **Kill** <0.375 pooled; a second DAgger round only if round 1 moved ≥+3pp.

## Session 2 — Leg 3: disagreement-weighted solver distillation + matched-size re-probe (~2h)

1. Collect 2,200 more `--teacher solver` games appended to `data/bc_v2_solver` (shard
   numbering auto-resumes, `bc.py:171`; split into 2 seeded runs if a session threatens 2h)
   → matches the generic leg's 3,000.
2. **Mark disagreements at collection:** when `teacher=="solver"`, also run the plain generic
   pilot per decision and record `solver_diff = int(solver_pick != generic_pick)` as a new
   (optional-for-old-shards) shard column. ~1-2 rows/game.
3. **Weighted CE** in `train_v2`: `F.cross_entropy(..., reduction="none") *
   (1 + (W-1)*solver_diff)`, W=10, no sweep → `osv2_soldis1.pt` — trains the solver's
   1%-of-prompts signal instead of drowning it in the 99% where the teachers agree.
4. Gate as Leg 2 (go ≥0.375 at n=400 vs 0.345). This also settles the deferred matched-size
   solver-teacher re-probe with the right metric. **Kill** → teacher-gap track closed. Use
   the pilot-v2 teacher if Leg 1 promoted.

## Session 3 — Leg 4: value-as-ranker probe (offline-gated; ~1-2h only if the offline gate passes)

1. Add read-only `score_siblings(obs, deck)` to `rl/turn_solver.py`: run
   `_candidate_actions` at the root, step each one ply, `_dfs` remainder, return per-sibling
   `score_leaf` scores (`turn_solver.py:256`) — pure recombination of existing machinery.
2. New `collect-rank` mode in `rl/value_train.py` (beside `collect`, `value_train.py:75`),
   v1 stack (the MCTS path is v1-only): record (state_ctx, option encodings, sibling scores)
   at learner decisions.
3. Train a **pairwise logistic ranking head** (action-conditioned Q/ranker on the
   OptionScorer option pathway, fresh head) over sibling pairs whose score gap clears a
   margin (skip aliased ties) → `bc_v1_rank.pt`.
4. **Gate 1 (offline, minutes, zero games):** held-out pairwise accuracy ≥0.70 on
   margin-filtered pairs; kill below. **Gate 2:** re-run sims ladder
   `mcts:<ckpt>:lucario:{16,32}` n=200 each; flat again = three strikes, search track closed
   for the campaign.

## Session 4 — Leg 5 (REQUIRES Piotr's sign-off: touches rl/ppo.py, reverses the M8.5 no-go)

- **5a — KL-anchored PPO:** add `β·KL(π_θ‖π_BC)` (β≈0.02) inside `ppo_update`
  (`rl/ppo.py:149`), warm-start from the best base produced by Legs 1–3. One bounded
  attempt: 5 iters, eval-every 1, ~2h → `ppo_m9_kl.pt`. Directly targets the reproduced
  peak-then-decay drift off the BC manifold.
- **5b — AZ-lite visit-count probe:** `mcts_search` returns `root.N` (`rl/mcts.py:160-203`)
  and nothing consumes it. One bounded probe: ~200 games of 16-sim visit distributions, one
  KL-to-visits BC epoch, eval n=200. Kill on flat.
- Both are single-shot: no iteration budget beyond the bounded run regardless of near-misses.

## Sequencing

Leg 1 ∥ Leg 2-collection → Leg 2 train/gate → Leg 3 (only if Leg 2 moved ≥+3pp or its
collector is already built) → Leg 4 offline gate whenever a free hour exists → Leg 5 only on
sign-off. If Leg 1 promotes, immediately re-run Legs 2/3 with the pilot-v2 teacher — the
neural track's ceiling rises with the teacher.

## Files to modify

- `rl/generic_pilot.py`, `rl/turn_solver.py`, `rl/matchrunner.py` (Leg 1 `fixes` param +
  `solver2:`/`generic2:` specs; Leg 4 `score_siblings`)
- `rl/bc.py` (Leg 2 `collect_dagger`, multi-dir `BCDatasetV2`, `--init` warm-start; Leg 3
  `solver_diff` column, weighted CE)
- `rl/value_train.py` (Leg 4 `collect-rank`, ranking trainer)
- `rl/ppo.py` (Leg 5a only, after sign-off)
- Twins `tcg/pilot.py`/`tcg/constants.py` mirrored post-measurement; `docs/M9.md` diary.

## Verification

- Each leg has an explicit go/kill gate; nothing promotes without pooled n≥1200, two seeds,
  and the floor battery.
- `pytest tests/` (incl. `test_parity.py`) stays green after Leg 1's default-off flag —
  byte-parity of the untouched `solver:` path is the regression check.
- End-to-end on any promoted candidate: `python -m tcg.shipping export` + `gate` league
  gates, then `./build_submission.sh --checkpoint <ckpt> --deck lucario` (neural) or the
  rules-bundle path (pilot v2).
