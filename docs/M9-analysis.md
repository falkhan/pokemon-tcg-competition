# M9 input — why the neural agent is stuck, and where the headroom actually is

**Date:** 2026-07-14 · **Branch context:** `feature/m8` @ `5a3ee4e` ("first
pipeline — fail to beat the current best") · Written as input to the M9 plan.

## TL;DR

The neural agent's win rate is a **product of three factors, and the first
one is the binding constraint**:

```
strength ≈ teacher strength × imitation fidelity × (1 + RL uplift)
             ~0.467              ~0.74–0.77            ~1.00
```

- **The BC teacher is the WEAKEST of the three rule brains** (the user's
  hypothesis — confirmed, and it's a documented house law: "agent strength ≈
  teacher strength × imitation fidelity", DECISIONS.md 2026-07-08). A
  *perfect* clone of the generic pilot would still sit ~8pp below the 0.55
  campaign bar.
- The clone then loses another ~11pp to imitation drift, and 44.6% of its
  training decisions come from **losing** seats (measured below).
- PPO uplift is ~zero under every configuration tried — re-confirmed by the
  new pipeline run, which (see §3) re-ran a pre-M8.3 recipe and reproduced
  the already-falsified result.

The two avenues with real headroom both attack the teacher term: **raise the
teacher and re-clone** (taxonomy pilot fixes; expert distillation re-test
under materially changed premises), and **clone only the teacher's winning
play** (win-filtered BC — runnable today from the shards committed in
`5a3ee4e`). The M8.6 inference stack (solver override + guard) then adds its
independent points on top of whatever clone exists.

## 1. The strength ladder (all vs `solver:lucario`, documented sources)

| Agent | WR vs solver | Source |
|---|---|---|
| Expert (`rule:lucario`, AttackPlan) | **0.638** | M8.0 leg 1: solver vs expert 0.362 (n=400) |
| Solver ship agent (mirror) | 0.500 | definition |
| Generic pilot — **the BC teacher** | ~0.467 | M8.0: solver A/B vs generic 0.533 (n=800) |
| osv2_bc2 / ppo_m83_legB_it5 (best clones) | 0.345 / 0.359 | M8.2 / M8.3 (n=400/800) |
| osv2_bc (bugged-teacher clone) | 0.237 | M8.0 |

Campaign bar: **0.55**. Ceiling of a perfect generic-pilot clone: **~0.467**.
The teacher choice was rational at the time (observable scoring, no hidden
state to alias) — but it hard-caps the whole neural track below the bar.

## 2. The imitation gap, quantified from the committed shards

`data/bc_v2b` (3000 games, fixed generic teacher, now in-repo):

- **200,847 decisions**; **55.4% from winning seats, 44.6% from losing seats**
  — nearly half the dataset teaches the teacher's *losing* patterns with the
  same cross-entropy weight as its wins.
- Menus: median 4 options, p90 15, max 147; 15.2% forced (single option).
- Fidelity of osv2_bc2: 0.887 val top-1 → the ~11pp strength gap to the
  teacher is compounding drift (BC is trained on teacher states, plays its
  own) plus the losing-seat supervision above.

## 3. What the new pipeline run actually tested (and why it couldn't work)

`ppo_run.log` / `ppo_run2.log` + `checkpoints/ppo_m75_it49.pt` (v2 arch):

- Both runs used the **M7.5 attempt-2 recipe** (fixed-solver promotion;
  run 2: `bc_v1_value` critic warm-start, baseline "current best vs_solver
  28.5%") — i.e. the stack from *before* the M8.2 clean re-clone and the
  M8.3 stabilization (dev-potential shaping + lr 3e-5).
- Run 1 (50 iters): vs_teacher flat 28–38%, vs_best oscillating 38–50%, and
  **entropy creeping 0.575 → 0.72** — the diffusion signature M8.3 leg B
  cured is back, exactly as expected without the shaping/lr fix.
- Run 2 (19 iters): promoted once at 34.5% (n=200) then decayed 29.5 → 26 —
  the classic rise-then-decay.
- **Verdict: no new information.** M8.3 already established that even the
  *stabilized* loop is neutral (0.359 @ n=800, +1.4pp = noise); this run
  reproduced the pre-fix pathology. Its only value is confirming that
  re-running PPO variants is not where the next point comes from.

## 4. Why PPO stays neutral (mechanics, for the record)

- The collector pool is already majority rule-based with the ship agent
  heaviest (M7.5 rebalance) — curriculum diversity is not the gap.
- At ~0.30 WR vs the strong pool members, positive-advantage trajectories
  are rare and the win/loss-only return gives ~1 bit per ~90-decision game;
  the gradient mostly polishes the BC prior (ratio ≈ 1.0 throughout both
  logs). Dev-potential shaping fixed the *stability*, not the *signal*.
- KL-anchor leg C remains untried, but the marginal-point evidence says it
  buys stability PPO already has, not signal it lacks.

## 5. Avenues, ranked by expected headroom

### A. Raise the teacher, then re-clone (attacks the binding constraint)

- **A1 — Taxonomy pilot fixes (cheap, compounding).** The M8.0 taxonomy's
  two open pilot-side holes: keepers burned to Carmine vs END (fix: keepers
  score BELOW END), and Boss's Orders hoarded 8+ turns (gust never used).
  Every +1pp on the generic pilot is ≈ +0.75pp on a 0.887-fidelity clone,
  and improves the ship agent for free. Gate: generic-vs-solver A/B before
  re-cloning.
- **A2 — Expert distillation RE-TEST (flagged: re-testing a falsified
  result under changed premises).** "Clone the expert" was refuted
  2026-07-08 (55% fidelity wall, weaker clone than bc_v1 — hidden AttackPlan
  state). Three premises have since changed: (i) M7.2b proved AttackPlan is
  a **single-turn commitment reproducible from observable state** (that's
  where the pilot's race math came from); (ii) encoders v2 (id embeddings,
  deck context, combat/race features) replaced the v1 encoding the wall was
  measured on; (iii) DAgger now exists (`--driver`), so expert labels can be
  collected on *student-visited* states instead of pure expert trajectories.
  Probe: 500-game expert-teacher collection → fidelity read. **Kill if val
  top-1 < 0.75** (the old wall was 0.55; generic reference 0.887). If it
  clears, the teacher term jumps 0.467 → 0.638 and everything downstream
  re-rates.

### B. Squeeze the imitation term (no new ceiling, but cheap points)

- **B1 — Win-filtered / advantage-weighted BC (NEW, runnable today).** Train
  osv2 on winning-seat decisions only (111,355 remain — 56% of the data
  that produced 0.887 fidelity) or weight decisions by `results` (already a
  shard column). Clones "the teacher on its good days"; the only BC variant
  that can *exceed* average teacher strength. Cost: one `train_v2` variant
  + one n=400 eval. Needs a ~10-line mask/weight hook in
  `BCDatasetV2`/`train_v2` (both twins).
- **B2 — The built M8.6 stack** (model-solver, guard, DAgger legs — plan
  §M8.6): closes toward the solver-mirror ~0.50 on whatever clone is best.
- **B3 — Data scale:** 3000 → 10k teacher games (teacher self-play is
  embarrassingly parallel; fidelity is likely data-limited before
  capacity-limited).

### C. PPO — park it

Keep the stable loop for distillation duty (A2/B1 warm starts). No more
recipe iterations without a materially changed premise; the last three runs
(M8.3 A, M8.3 B, this pipeline) all landed within noise of their warm start.

### D. Measure before choosing: clone-loss taxonomy

The M8.0 taxonomy profiled the *solver's* losses; nobody has batch-
postmortemed the *clone's* losses (plan §M8.6 forensic leg). If they're
lethal-missed-dominated → B2's solver override covers it; if
setup-dominated → prioritize A1's pilot fixes. ~100 games + existing
`postmortem --batch`.

## 6. Expected-value map (honest)

| Stack | Expected vs solver | Notes |
|---|---|---|
| Best clone today + B2 wrappers | 0.40–0.50 | M8.6 legs, no retrain |
| + B1 win-filter + B3 scale | 0.44–0.52 | still generic-teacher capped |
| + A1 teacher fixes, re-clone | 0.48–0.55 | scales with taxonomy fix size |
| A2 success (expert clone ≥0.75 fid) + B2 | **0.52–0.60** | the only path with clear bar-plus headroom |

The bar most likely falls to **A2-if-it-works, else A1+B stack**; B alone
probably tops out just under it.
