# M11 plan: learned turn-planning — plan-conditioned policy via expert iteration

Status: **approved and started 2026-07-16 (eve), branch `feature/m11`.** Diary:
[M11.md](M11.md). This spec is kept current — mid-flight design changes get dated
notes here.

## Context

M9 measured that buddy (external rule agent) beats our solver 0.615 through **plan
coherence** — a turn-scoped AttackPlan that every prompt serves — but imitating it is a
measured dead end (osv2_dagger2 0.198: hidden-plan labels are contradictory w.r.t. obs).
User directive: buddy-like performance **without manual heuristics**, learned from many
simulations with exploration annealed high→low. M11 does this by (1) making the plan an
explicit observable input, (2) using the existing turn solver at widened offline budgets
as the expert-iteration improvement operator (supervised training only — no `rl/ppo.py`,
which is Piotr-owned and a measured dead end), (3) exploring at the plan level with
temperature annealed over rounds. Bar: ≥0.55 vs `solver:lucario` pooled n≥1200/2 seeds +
meta co-gate; **ship at ≥0.53 pooled** (user policy).

**Documentation discipline (user-requested):** `docs/M11-plan.md` is written FIRST
(this plan, in the M9-plan.md format) and kept current — any mid-flight design change
(dims, gates, schedules) is edited into the spec with a dated note, so the spec always
reflects what was actually built. `docs/M11.md` is the running diary, updated at every
step (not just end-of-session), one dated entry per rung/round containing: exact
command + `runs/*.jsonl` path + verdict, AND an **Observations** subsection with the
qualitative notes future iterations will need — plan coverage %, derive-plan ambiguity
rate, plan-head val accuracy, which plan types the head favors (null-plan rate,
gust rate, needs_attach rate — log a candidate-choice histogram per collection run),
where the student derails from solver lines, τ's visible effect on diversity, any
surprises/bugs/hypotheses. Kills get the same treatment as promotions — the notes are
the input to the next round's design.

**The load-bearing correctness rule (anti-aliasing invariant):** every training row's
plan features come from the *same solve that produced that row's label*. This is what
separates M11 from the buddy-DAgger 0.198 disaster.

## Design (settled)

- **PLAN_DIM=27 feature block** (all-zeros ⇒ "no plan"): [0] has_plan · [1:7] attacker
  slot one-hot (active+5 bench) · [7:13] target slot one-hot · [13:17] attack index
  one-hot (position in `_CARD[id][3]`, cap 4) · [17] needs_attach · [18] needs_gust ·
  [19] damage/340 · [20] target prize/3 · [21] lethal · [22] wins_game ·
  **[23:27] risk/trade features (user-requested, buddy's prize-trade logic as
  learnable features not rules):** [23] attacker prize value/3 (what I expose) ·
  [24] return_ko_exposed (opp active's `_best_damage(op_active, attacker,
  extra_energy=1)` ≥ attacker hp after my attack — they KO my planned attacker next
  turn) · [25] concedes_game (return_ko_exposed AND attacker prize ≥ opp prizes left —
  losing this attacker loses the game) · [26] opp best turns-to-KO my attacker /4
  (capped race view). All combat scalars from deck-agnostic `rl/combat` math computed
  vs the opponent's ACTUAL current board (active + bench are fully observable in obs).
  The opponent's board also reaches the plan scorer through the state trunk
  (encode_state_v2 numeric + BOTH boards' card-id embeddings in state_ids), so
  matchup-specific play ("this card vs that opponent") is learnable; the explicit
  [23:27] interaction features make the prize-trade case cheap to learn, and the
  null-plan candidate is the learnable "don't sacrifice the Mega ex" action.
- **Plan rides as a separate tensor/shard column** — `encode_state_v2` and all five v2
  state_ctx construction sites untouched; old shards (bc_v2b etc.) mix in with zeros
  default (semantically exact "no plan"); no regeneration.
- **`OptionScorerV3`**: forward(state_ctx, plan, state_ids, options, option_ids);
  state_enc in-dim 1580 (1297+64+27+192); plus `plan_enc` + `plan_head` for
  **plans-as-options scoring**: enumerate feasible plan candidates from obs (cheap
  combat math), encode each as its PLAN_DIM vector, score jointly against the trunk —
  the codebase's pointer-net idiom; chosen candidate's vector is byte-identical to what
  conditions the policy (zero train/serve mismatch). Detection: `"plan_enc.0.weight"`
  in state_dict (+ assert state_enc.0 width 1580).
- **Warm-start invariant**: `load_v2_into_v3(v2_sd)` copies v2 weights, zero-inits the
  27 plan columns ⇒ V3(plan=0) ≡ V2 (unit-tested).
- **Plan candidates** (`rl/plan.py`, bundle-pure: numpy+cg.api+rl.combat): null plan
  first; attackers = board slots with `_best_damage>0` (or with +1 energy when a basic
  energy in hand and `not energyAttached` ⇒ needs_attach); bench targets only when a
  gust trainer in hand (`GUST_IDS={1182}` — card fact, not heuristic); cap 48.
- **Inference protocol**: recompute plan at every own MAIN prompt
  (`obs.select.context == SelectContext.MAIN` — NOT `select.type`), hold for submenus
  keyed on `(obs.current.turn, yourIndex)`; τ-sampling only during collection, argmax
  at eval. Reset state on `obs.select is None` (kaggle) AND turn-counter drop
  (direct loop — matchrunner pilots persist across a whole series, `matchrunner.py:289`).
- **Exploration**: τ over rounds 1→4: 1.0, 0.7, 0.4, 0.2; Dirichlet(α=0.5, ε=0.25)
  round 1 only. Sampled plan shapes execution only — never a training input.

## Phase E — all edits batched before any run

1. **`rl/turn_solver.py`**: `_dfs` returns `(score, line, obs_list)` (obs at which each
   line[i] was taken — already in scope); depth/node caps read from the `budget` dict
   (no module-global mutation). New `solve_turn_line(obs, deck, deadline_s=…, dev=False,
   fixes=frozenset(), max_depth=MAX_DEPTH, max_nodes=MAX_NODES) -> (score, line,
   per_step_obs)` — no truncation, no override gate (per-step obs are determinized
   snapshots: plan derivation ONLY, never training states). `solve_turn` becomes a thin
   wrapper, byte-identical behavior (existing tests pin it).
2. **`rl/plan.py`** (new): `PLAN_DIM`, `Plan` namedtuple, `encode_plan`,
   `enumerate_plans(obs)`, `derive_plan(line, per_step_obs, root_obs)` (ATTACK step →
   attackId + attacker slot mapped to root coords via promote/retreat picks;
   EFFECT_TARGET pick → target slot; ATTACH-before-ATTACK → needs_attach; no ATTACK ⇒
   null plan; return None + log when enumeration misses the derived plan).
3. **`rl/policy.py` + `tcg/network.py` twins**: `OptionScorerV3` (+`plan_logits`,
   `act`, `act_plan(tau, dirichlet_eps, rng)`).
4. **`rl/plan_iter.py`** (new): everything else —
   - `collect(mode=expert|ei, n_games, decks_file, out_dir, checkpoint=None, tau=1.0,
     dirichlet=0.0, deadline=2.0, label_deadline=0.5, workers=12, shard_size=200,
     seed=0)`. **Multiprocess** (spawn Pool over game chunks, matchrunner `_make_jobs`
     idiom; per-worker shard names; single-process DAgger's 5 games/min was the
     bottleneck). Per-seat per-game plan state machine keyed on (turn, yourIndex).
     *expert mode*: widened `solve_turn_line` at each own turn's first MAIN; execute
     the line in the real game with derail guard (validate indices vs live menu; on
     derail re-solve at label_deadline, override gate bypassed, greedy fallback on
     stand-pat). *ei mode*: student (checkpoint) samples plan at τ + Gumbel-sampled
     actions conditioned on it; teacher labels per the anti-aliasing invariant.
     Shard columns = collect_games_v2's nine + `plans(D,27)`, `plan_cands(sum_M,27)`,
     `n_plan_cands(D,)`, `plan_labels(D,)` (−1 = skip plan loss). Game-25 wall-clock
     projection print; shard auto-resume.
   - `BCDatasetV3`/`collate_v3` (loader lives HERE — **`rl/bc.py` gets zero edits**;
     it's parity-pinned and shared with Piotr). Old plan-less shards get shaped
     defaults (plans=zeros, plan_labels=−1). Reuse `_WeightedView`/`load_population`
     from bc.py.
   - `train(data_dirs, name, init=None, init_v2=None, epochs, lr, plan_weight=1.0)`:
     loss = CE(policy) + 0.5·value + plan_weight·CE(plan_logits) over rows with
     plan_labels≥0; reports val top-1, per-deck, plan-head top-1, coverage.
     `load_v2_into_v3` lives here.
   - CLI `python -m rl.plan_iter collect|train`.
5. **`rl/matchrunner.py`**: v3 branch in the existing `model` kind (before the v2
   check): closure `fn3` with plan state (reset on select-None AND turn drop), replan
   at MAIN via `enumerate_plans` + `plan_logits` argmax → league gates and meta-eval
   work for free.
6. **Ship path** (only at Rung 2): `submission/main.py` v3 branch (`score_options_v3` +
   `score_plans` numpy twins, module-global plan state), `tcg/shipping.py` export
   bundles `rl/plan.py` + parity_check v3 branch (forward AND plan_logits, ≤1e-4).
7. **Tests** (all before runs; 411 existing + new green): solve_turn_line vs solve_turn
   first-action identity + obs_list length; plan encode/enumerate/derive on synthetic
   lines; stubbed-engine collect wiring for both modes (executed-vs-label separation);
   **V3(plan=0) ≡ V2 warm-start invariant**; BCDatasetV3 old+new shard mixing; rl/tcg
   V3 twin parity; fn3 plan-state reset.

## Phase R — rungs (each round ≤2h; >2h needs Piotr sign-off; no rl/ edits while live)

**Rung 0 — infra + plan-BC v0** (after pytest green + smoke n=20):
```
python -m rl.plan_iter collect --mode expert --games 600 --decks data/league/population.json \
    --out data/plan_ei0 --deadline 2.0 --label-deadline 0.5 --workers 12 --seed 0
python -m rl.plan_iter train --data data/plan_ei0 data/bc_v2b --init-v2 checkpoints/osv2_bc2.pt \
    --epochs 8 --lr 3e-4 --name osv3_plan0
python -m rl.matchrunner play --a model:checkpoints/osv3_plan0.pt:lucario --b solver:lucario \
    -n 400 --workers 12 --seed 1 --checkpoint runs/m11_r0_s1.jsonl
```
Gates: plan coverage ≥0.85 at collection; **wr ≥0.38** (base osv2_bc2 0.345).
Kill: <0.35 after one fix iteration.

**Rung 1 — EI rounds r=1..4** (τ = 1.0/0.7/0.4/0.2; round 1 adds `--dirichlet 0.25`;
optionally `data/league/population_m11.json` with lucario ×4 for mirror share):
```
python -m rl.plan_iter collect --mode ei --checkpoint checkpoints/osv3_ei{r-1}.pt --games 600 \
    --decks <pop> --out data/plan_ei{r} --tau {τ} --deadline 2.0 --label-deadline 0.5 \
    --workers 12 --seed {r}
python -m rl.plan_iter train --data data/plan_ei0 … data/plan_ei{r} data/bc_v2b \
    --init checkpoints/osv3_ei{r-1}.pt --epochs 5 --lr 1e-4 --name osv3_ei{r}
python -m rl.matchrunner play --a model:checkpoints/osv3_ei{r}.pt:lucario --b solver:lucario \
    -n 400 --workers 12 --seed 1 --checkpoint runs/m11_ei{r}_s1.jsonl
```
Round gate: ≥ previous +3pp. Kill: 2 consecutive flat rounds. Early exit: screen ≥0.55.

**Rung 2 — battery + ship** (best checkpoint R): confirm n=800 seed 2
(`runs/m11_confirm_s2.jsonl`); pooled n=1200 **ship ≥0.53** (campaign ≥0.55, each seed
>0.52); `replay_bc meta-eval` ≥0.448; league G1–G6 (random ≥0.90 n200, generic ≥0.55
n400, rule:lucario ≥0.35 n400, latency mean <50ms); then Phase-E step 6 ship edits →
`tcg.shipping export --checkpoint osv3_eiR.pt --deck lucario` → `gate` →
`./build_submission.sh --checkpoint … --message "M11 plan-EI"`.

## Verification

- `pytest tests/` green after Phase E; `solve_turn` byte-identical (existing pins).
- V3(plan=0)≡V2 invariant test; shard-mixing test; parity (torch↔numpy, rl↔tcg twins).
- Smoke: `plan_iter collect --mode expert --games 5` shard loads in BCDatasetV3;
  matchrunner v3 pilot n=20 zero-crash.
- Collection logs: coverage ≥0.85, derive ambiguity rate, plan-choice histogram
  (null/gust/attach rates), game-25 projection.
- Diary discipline: docs/M11.md updated at every step — command + jsonl + verdict +
  Observations subsection per entry (see Context); docs/M11-plan.md kept current with
  dated notes on any design change.

## Files
New: `rl/plan.py`, `rl/plan_iter.py`, `tests/test_plan.py`, `tests/test_plan_iter.py`,
`docs/M11-plan.md`, `docs/M11.md`. Edits: `rl/turn_solver.py`, `rl/policy.py`,
`tcg/network.py`, `rl/matchrunner.py`; Rung-2 only: `submission/main.py`,
`tcg/shipping.py`. **Untouched: `rl/ppo.py`, `rl/collector.py`, `rl/bc.py`,
`rl/replay_bc.py`, `rl/encoders.py`, all five v2 state_ctx sites.**
