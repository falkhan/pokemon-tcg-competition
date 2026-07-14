# M9 plan — Exceed the teacher (options to test, post-M8)

**Charter:** unchanged from M8 — any agent **≥ 0.55 vs `solver:lucario` at
n ≥ 800** (±3.5pp), zero G1 crashes, latency within G6, then the ship gates.
Best artifact today: `ppo_m83_legB_it5.pt` at **0.359** (osv2_bc2 0.345).

**Why these options.** M8 falsified the two classic routes: policy-gradient
PPO (stable now, but the sparse-outcome signal can't move a BC-anchored
policy — M8.3) and value-guided search (sims ladder flat with both L4 fixes
in — M8.4). What survives is everything that improves a policy **without**
either mechanism:

1. **BC's remaining loss is distribution shift, not label quality.** Val
   fidelity is 0.887 *on teacher-visited states*; the student plays itself
   into states the teacher never showed it, and errors compound. DAgger
   (student rollouts, teacher labels) is the textbook repair and has never
   been tried here.
2. **Vanilla BC cannot exceed its teacher, and the teacher loses to the
   bar.** The generic teacher is ≈ 0.467 vs the solver (solver A/B 0.533).
   Closing the imitation gap tops out below 0.5 — crossing 0.55 needs a
   mechanism that *deviates upward*. Outcome-weighted cloning (AWR-lite:
   supervised, like everything that has worked in this project — see the
   value-head lesson, DECISIONS 2026-07-08) is the cheapest such mechanism:
   clone winners' decisions harder than losers'.
3. **The solver's lethal edge is a wrapper, not a property of the rule
   pilot.** `solve_turn`/`should_solve` are policy-agnostic; the neural
   checkpoint can inherit the within-turn lethal search at inference for
   free. M8.0's taxonomy says lethal conversion is where search pays
   ("search only pays where greedy is structurally blind").
4. **One recorded open item:** the M8.2 solver-teacher fidelity probe ran at
   800 games vs leg A's 3000 — "if a solver-teacher warm start ever looks
   necessary, re-probe at matched size first" (M8.md 2026-07-13). A solver
   teacher raises the imitation ceiling from ≈0.467 to 0.5 by definition.

No single lever clears the bar; the plan stacks them with a gate per lever.

**Constraints & conventions:** as M8 (legs ≤ 1–2 h, n=400 minimum / n=800
for ship calls, fresh recorded seed, record the solver version, deck-blind
math only, diary entry per leg including kills — diary: `docs/M9.md`).
Instruments landed on this branch, all offline-tested: `solver-model:`
matchrunner spec, `rl.bc collect --student` (DAgger + agreement rate),
`rl.bc train --weight-outcome BETA`, multi-dir `--data`.

---

## M9.0 — Measurement first (2 cheap probes; BLOCKS training legs)

**(a) Loss taxonomy on the NEURAL agent (~45 min).** M8.0's taxonomy audited
the solver pilot; the neural agent's loss modes vs the solver were never
tabulated. ~100 kaggle-env games best-checkpoint-vs-solver with
`json_prefix`, `postmortem --batch` on the losses. Routing: if
missed-lethal / race flags dominate → expect M9.1 to pay; if setup flags
dominate → weight M9.2/M9.3.

**(b) On-student-distribution fidelity (~20 min, the DAgger premise probe —
validate before paying for a run, the M2 meta-lesson):**
```
uv run python -m rl.bc collect --teacher generic --student checkpoints/osv2_bc2.pt \
    --games 100 --out data/bc_dagger_probe --shard-size 100
```
The printed student/teacher agreement is fidelity *where the student actually
plays*. **Hypothesis: agreement ≤ 0.80 (vs 0.887 on-teacher).** If agreement
≈ val fidelity instead, distribution shift is NOT the problem → kill M9.2,
reweight toward M9.3/M9.4.

## M9.1 — solver-model hybrid A/B (zero training; cheapest lever first)

The lethal turn solver (T1–T4, proven +edge on the rule pilot) layered on the
neural checkpoint at inference:
```
uv run python -m rl.matchrunner play --a solver-model:checkpoints/ppo_m83_legB_it5.pt:lucario \
    --b solver:lucario -n 800 --workers 14 --seed 91 --checkpoint runs/m91.jsonl
uv run python -m rl.matchrunner play --a solver-model:checkpoints/ppo_m83_legB_it5.pt:lucario \
    --b model:checkpoints/ppo_m83_legB_it5.pt:lucario -n 400 --workers 14 --seed 92
```
**Gates:** vs its own bare checkpoint ≥ 0.53 (the solver must add something);
vs `solver:lucario` ≥ 0.40 (beats 0.359 + noise). Latency within G6 (solver
adds ~50 ms/move on the rule pilot). **Kill:** hybrid ≤ bare checkpoint —
would mean the checkpoint already converts lethal, contradicting M9.0(a);
reconcile before training legs. If it passes, EVERY later checkpoint gets
measured both bare and hybrid — the wrapper is free win-rate.

## M9.2 — DAgger (distribution-shift repair; 1 round, then reassess)

Round 1 (~2 h collect + ~1 h train, gated on M9.0(b) showing a real gap):
```
uv run python -m rl.bc collect --teacher generic --student checkpoints/osv2_bc2.pt \
    --games 1500 --out data/bc_dagger_r1 --seed 90
uv run python -m rl.bc train --arch v2 --epochs 10 --name osv2_dag1 \
    --data data/bc_v2b,data/bc_dagger_r1
```
**Gates:** vs `solver:lucario` n=400 ≥ 0.38 (osv2_bc2's 0.345 + noise);
G2 vs random ≥ 0.845 non-regression; agreement re-probe with `osv2_dag1`
must RISE (the loop is converging). **Continue** to round 2 while each round
gains ≥ 3pp; **ceiling check:** this lever alone tops out near the teacher's
≈ 0.467 — it feeds M9.3/M9.5, it does not clear the bar by itself.

## M9.3 — Outcome-weighted cloning (the exceed-the-teacher lever)

AWR-lite on existing data first (no collection; ~1 h/leg):
```
uv run python -m rl.bc train --arch v2 --epochs 10 --name osv2_awr_b05 \
    --data data/bc_v2b --weight-outcome 0.5
```
Sweep β ∈ {0.5, 1.0, 2.0}. The epoch line's win/loss-decision accuracy split
is the mechanism check: win-acc holds, loss-acc drops. **Gates:** best β vs
`solver:lucario` n=400 > osv2_bc2's 0.345; G2 non-regression (β too high =
cloning lucky noise — the known AWR failure mode; the per-batch
normalization keeps lr comparable across β). **Kill:** all β ≤ baseline →
outcome credit at 1-game granularity is too noisy; note for the diary and
fall back to plain DAgger data. **Stack:** rerun best β on
`data/bc_v2b,data/bc_dagger_r1` once M9.2 lands — weighted student-state
labels are the two levers composed.

## M9.4 — Solver-teacher at matched size (the recorded M8.2 open item)

Only if M9.1 shows the solver edge transfers OR M9.2/M9.3 stall below the
teacher ceiling: 3000-game `--teacher solver` collection (matched to leg A),
re-measure the fidelity drop (bar: ≤ 3pp, as M8.2). Pass → the solver
replaces the generic teacher in the M9.2/M9.3 recipes (ceiling 0.467 → 0.5+).

## M9.5 — Stack + ship path

Best checkpoint from M9.2/M9.3(/M9.4), wrapped `solver-model:` (per M9.1),
n=800 vs `solver:lucario` for the campaign read, then the league gates and
`./build_submission.sh --checkpoint ... --deck lucario`. Note the ship bundle
is numpy-only: shipping the hybrid needs the turn solver's engine calls,
which already ship in the rules bundle — wiring is the M7.5 two-line flip
plus the checkpoint pilot as `inner`.

## Deferred / rejected (with reasons)

- **KL-anchored PPO (leg C):** available, Piotr's call, but M8.3 says spend
  elsewhere. Revisit only as distillation of a stronger artifact.
- **Ranking-loss distillation of solver leaf scores** (attack the "value
  can't rank siblings" wall with a ranker teacher, not a classifier):
  real candidate, bigger build (search-tree logging + a margin loss). Open
  it only if M9.1–M9.3 all stall.
- **More MCTS / sims:** falsified twice with instruments fixed (M8.4).
- **Per-deck BC / deck pipeline:** the team's parallel lever, not this plan.

## Decision tree

```
M9.0(a) taxonomy ∥ M9.0(b) agreement probe ∥ M9.1 hybrid A/B   (independent)
M9.0(b) gap real → M9.2 round 1 → rounds while +3pp/round
M9.3 β sweep on bc_v2b (parallel to M9.2 collect; CPU serializes)
M9.1 pass → all later evals bare + hybrid | fail → reconcile with M9.0(a)
M9.2/M9.3 stall at teacher ceiling → M9.4 solver teacher (matched-size probe)
best artifact + hybrid ≥ 0.55 @ n=800 → league gates → ship call (Piotr)
```

## Piotr's calls, explicitly

1. Any run > 2 h (the M9.2 collect at 1500 games flirts with it). 2. Opening
the ranking-distillation build. 3. Every ship flip. 4. PPO leg C, if ever.
