# M13 plan: outcome-grounded setup value — games → deck understanding

Status: **SPEC'D 2026-07-17, not started.** Diary: [M13.md](M13.md) (created).
Bar unchanged: any agent ≥0.55 vs `solver:lucario` pooled n≥1200/2 seeds + meta
co-gate; **ship at ≥0.53 pooled** (standing user policy). This spec is kept
current with dated notes on any mid-flight change.

## Context — why this and only this is left

The user's goal: the model should **learn to play this particular deck** —
combos, card synergies, energy routing — from playing many games. Twelve
milestones of measurement say game volume only converts into skill through a
signal that (a) understands card effects and (b) can value *setups*, not just
prize lines:

- Imitation caps at teacher strength × fidelity; no observable teacher
  understands the deck (M1–M3, M9, M10, M11-R0).
- Win/loss PPO: credit too sparse, measured dead twice (M2/M8.3).
- Everything built on `score_leaf`'s non-lethal scoring is dead — M8.1 (0.314),
  M11 EI (0.314/0.357), M12 Gate 2 (0.383) triangulate ONE bottleneck: the
  search can execute combos exactly (the engine plays every effect) but cannot
  tell whether a *setup* matters.
- M12 Gate 1 (pairwise 0.873) proved the fix for "value can't rank" is the
  RANKING LOSS. What was missing is outcome-grounded labels instead of
  heuristic-score labels (the M12 circularity).

**M13 thesis:** self-play at real volume → a setup-value head trained on GAME
OUTCOMES with a pairwise ranking loss over matched pairs → that value replaces
the heuristic leaf inside the (exact, effect-executing) turn search → the
search becomes an improvement operator that fires meaningfully on EVERY turn →
M11's expert-iteration ladder finally has something to climb.

Live-observation fixes fold in as Rung 0 groundwork (both are data-quality
issues for the same pipeline): the Solrock/Lunatone conditional-attack card
fact, and ATTACH_FROM prompt oversampling.

## Known dead ends this must not repeat

- Raw win/loss classifier value (M5, M8.4) — pairwise ranking loss ONLY.
- Value labels derived from `score_leaf` (M12 circularity) — labels come from
  game OUTCOMES only.
- Inference-time patches over the net (live-obs round 1: 0.359/0.372/0.385) —
  all fixes land in data/tables/training, never as agent()-time overrides.
- Unbounded/high-noise exploration during collection (M11 EI round 1).

## Rung 0 — groundwork + data + the offline value gate (kill-fast)

**0a. Card-fact patch (conditional attacks).** New fact table in `rl/combat.py`
(mirrored in `tcg/combat.py` — parity-tested):
`CONDITIONAL_ATTACKS = {980: ("requires_board_name", "Lunatone")}` — attack 980
(Solrock) deals 0 unless a Lunatone is on own board. Honored inside
`_best_damage`/`_charged_best` via an optional `board_names` argument threaded
from callers that have the board (generic_pilot attach values, plan
enumeration, solver facts). Default-empty behavior byte-identical for all other
cards; extend `tests/test_parity.py`/`test_combat.py`. This stops teacher,
plans, AND labels overvaluing Solrock at the source (live-obs finding).

**0b. Self-play state collection at volume.** New `rl/setup_value.py`
`collect` (template: `rl/rank.py` collector — multiproc, solver pilots at LIVE
budgets so games are fast): log every own decision's `(state_ctx, state_ids,
turn, my_prizes, opp_prizes, deck_idx, game_id, result)` — states only, no
solves ⇒ ~5s/game. **Batch 1: 2,000 games (~35 min, 12 workers) — fits the 2h
rule.** Scale-up batches to 10k+ overnight REQUIRE Piotr's sign-off (flag to
him regardless: value learning is RL-core-adjacent and his interest area; no
`rl/ppo.py` edits in any rung). ATTACH_FROM decisions logged with a marker
column for later oversampling.

**0c. Matched-pair construction + training.** Pairs drawn from DIFFERENT games
with MATCHED context — same turn bucket (bins of 4), same (my_prizes,
opp_prizes), same matchup (mirror-dominant) — one state from an eventually-won
game vs one from an eventually-lost game; label: won > lost. No same-game
temporal pairs in v1 (confound risk). Exclude turns ≤2 (openers aliased).
Train the `OptionScorerV3` trunk's value pathway (plan=zeros) with pairwise
logistic loss (reuse `rl/rank.py::_pair_stats`) → `osv3_setupval1.pt`.

**Gate V0 (offline, zero games): held-out matched-pair ranking accuracy
≥0.62** (chance 0.5 by construction; report per-turn-bucket accuracy — the
late-game buckets should be easiest, the turn-3–8 SETUP buckets are the ones
that matter). Kill <0.58 after one iteration (pairing/feature fix allowed).

## Rung 1 — value-guided search probe (the broad improvement operator)

New leaf mode in `rl/turn_solver.py`: `score_leaf` keeps its EXACT terminal +
prize terms as anchors (W_WIN/W_PRIZE untouched — the M12 lesson: never let a
learned signal override a certain prize), and replaces the sub-prize heuristic
tail (threat/counter/race/dev-bonus) with `LAMBDA * V(state)`, LAMBDA scaled so
the learned term stays strictly below one prize (the dev-bonus precedent).
Torch at data-gen; numpy replay for any shipped variant (same npz mechanism as
v3).

- **Gate S1:** `vsolver:<ckpt>:<deck>` matchrunner spec (solver pilot with the
  learned leaf active on all turns via the dev-tier trigger path) vs
  `solver:lucario`, screen n=400 seed 1, confirm n=800 seed 2. Null effect =
  0.500 mirror. **Go ≥0.53 pooled; kill ≤0.50 pooled.** (M8.1's heuristic
  dev-tier scored 0.314 here — clearing 0.50 is already history-making.)
- Measure latency in the same runs: one value forward per DFS leaf; if p99
  breaks the 50ms move budget, the SHIP config drops MAX_NODES / caps
  value-leaf count — measured, not assumed. A ship-eligible vsolver (≥0.53 +
  battery) ships per standing policy without waiting for Rung 2.

## Rung 2 — expert iteration on the broad operator (only if S1 ≥ 0.52)

M11's `rl/plan_iter.py` expert mode re-run with the value-guided solver as
teacher: the bar-honoring rule generalizes — a line is committed/labeled when
it clears MIN_OVERRIDE_SCORE **or** improves the learned leaf by a margin
calibrated on V0's held-out distribution (the operator now fires meaningfully
on most turns, which is exactly what M11's EI lacked). ATTACH_FROM rows
oversampled (weight 3×, `_WeightedView` mechanism). Collect 600 → train
`osv3_plan1.pt` (init osv3_plan0c) → **gate ≥0.445** (plan0c 0.415 + 3pp);
then EI rounds r=1.. with τ=0.3 plan-level-only exploration (the M11 amended
protocol), round gate +3pp, kill on 2 flat rounds. Value head refreshed from
each round's new games (the full loop closes).

## Rung 3 — battery + ship

Best candidate (vsolver or osv3_plan1/ei): confirm n=800 seed 2 → pooled
n≥1200 **ship ≥0.53** (campaign ≥0.55, each seed >0.52) → meta co-gate
(pinned: ship-rules 0.448 · osv3_plan0c 0.406) → league G1–G6 → `tcg.shipping`
export/gate → `build_submission.sh --message` → add the submission id to
`notebooks/model_monitor.ipynb` MODELS and watch live.

## Budgets & hygiene

- Rung 0 end-to-end ≈ 1.5h attended (batch-1 volume). Rung 1 ≈ 1h. Rung 2
  rounds ≤2h each. Anything >2h (10k-game batches) = Piotr sign-off first.
- All edits batched before runs; `rl/ppo.py` untouched; every run logged in
  [M13.md](M13.md) with command + `runs/*.jsonl` + verdict + Observations
  (per-bucket V0 accuracy, operator fire-rate, latency, plan histograms);
  kills documented like promotions; MILESTONES.md row updated on close.

## Files

New: `rl/setup_value.py`, `tests/test_setup_value.py`, `docs/M13.md`.
Edits: `rl/combat.py` + `tcg/combat.py` (+parity/combat tests) [0a],
`rl/turn_solver.py` (learned-leaf mode + `vsolver` support), `rl/matchrunner.py`
(`vsolver:` spec), `rl/plan_iter.py` (teacher hook + oversampling) [Rung 2].
Checkpoints: `osv3_setupval1.pt`, `osv3_plan1.pt`, `osv3_ei_v{r}.pt`.
