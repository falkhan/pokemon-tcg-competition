# M23 — converge against strong opponents, then decide the architecture

**Status:** STUB (2026-07-21). Phase 1 is cheap and decides whether the next big investment is
training-side or architecture-side. Refine after Phase 1 runs.

## Context — what M22 established

1. **Every pilot we build sits at 0.17–0.22 vs `rule:dragapult`** (neural no-search, neural+search,
   rule+search), while the Pokémon Company sample agent sits at **0.50 on the same deck**. The
   deficit is not neural-specific and not about within-turn search.
2. **Our training loop is anchored to weak, endogenous opponents.** `solver:lucario` — 45% of B3's
   training and the mirror gate — is beaten **0.67** by `rule:lucario` (the sample agent) on the
   same deck. The agent learned to beat a weak sparring partner, which is why its 0.51 mirror
   collapses to 0.17 out-of-loop.
3. **The 0.22→0.50 gap is exactly two capabilities, both with inputs already in the encoder:**
   - **Gap A — prize-race / Mega-exposure denial** (policy gap): don't over-commit the 3-prize Mega
     when the opponent can cash it.
   - **Gap B — coordinated within-turn line** (sequencing gap): gust→attach→attack bound to one
     target. We *have* a plan head, but it never gets gradient toward coordinated lines because no
     teacher we have used produces them.
4. **Three levers tried in M22, none moved out-of-loop — but only two were real tests:**
   - C1 within-turn search: +2.25pp, unresolved, **no headroom** (6.7× compute changed 0 actions).
     Cleanly null.
   - C2 whole-board threat feature: falsified, p=0.548. Cleanly null (weak-test caveat noted).
   - **C2c teacher upgrade (one leg): +1.68pp, inside 5.48pp MDE — NOT a real test.** The agent
     trained 50% against `rule:lucario` and only reached 0.365 vs it; in-loop win-rate vs the
     teacher was **still climbing** (15% → 18% between iters 4 and 9). It had not converged.
5. **No validated offline→live predictor.** Mirror is contaminated (45% of training) and endogenous;
   `rule:dragapult` is the clean out-of-loop floor (never in any pool — keep it that way).

The M22c-RL leg (sub 54864089) is the direct antecedent: it under-powered the teacher test twice
(one slice, one short leg). M23 tests the hypothesis properly.

## Phase 1 — converge the strong-teacher recipe (PRIMARY, cheap, ~half day)

Continue from `ppo_current_m22cRL` (already 10 iters into the recipe) for ~20 more iters (≈30
total), and strengthen the *rest* of the pool — we are now trying to WIN, not isolate a variable,
so a broadly stronger opponent distribution is the right call:

- **Up:** `rule:lucario` (the 0.50 sample agent), `rule:iono` (sample agent, the other deck we own),
  `mirror` (self-strengthens as the agent improves — a free curriculum).
- **Down/out:** the weak `solver:` meta-deck slices, `solver:lucario` (the 0.22 pilot).
- **Hold:** `random:kyogre` floor slice; `past` self-play tail.
- **Never:** `rule:dragapult` — the clean out-of-loop gate; adding it spends the instrument.

Continue KL at 0 (pure PPO — that is what we are testing). Watch for drift over the longer run.

**The convergence signal is free and diagnostic** — log win-rate vs the teacher (`rule:lucario`)
each iteration:

| observation | reading |
|---|---|
| vs-teacher climbs toward ~0.45+ AND `rule:dragapult` moves >5.5pp | teacher/training lever VALIDATED — this is the recipe, iterate |
| vs-teacher plateaus well below the teacher AND dragapult flat | the plan head cannot learn to beat this opponent → the bottleneck is ARCHITECTURE → Phase 2 |

**Gate:** `rule:dragapult` n=800 2-seed vs the current 0.172/0.189. Mirror reported, not a ship
criterion. Ship-or-not by standing policy (technical gates enforced, win-rate informational).

## Phase 2 — the planning-head redesign (CONTINGENT, milestone-sized)

Only if Phase 1 plateaus. Make the plan head actually drive coordinated multi-step execution — the
gust→attach→attack-on-one-target line the sample agent achieves via a persistent plan structure
(`sample-agent/main.py`), trained against an opponent that punishes incoherence. Encoder-v5-scale
cost (width change → `migrate_v4_to_v5` warm start → PPO leg → re-pin). Scope only when earned.

## Strategic caveat — the teacher ceiling (flag, do not act on yet)

`rule:lucario` at 0.50 is a mediocre ceiling: perfectly matching it caps us near 0.50 vs dragapult,
while the leaderboard leaders are RL agents at ~800 score. The strongest *observable* teacher is
the **leaderboard replays** we already harvest (`rl/replay_bc.py`, the M10 "clone the leaderboard"
idea). Learning from them (BC warm-start + PPO) is the long-horizon answer if the sample-agent
teacher tops out — but it carries known failure modes (M21 BC-un-learns-PPO dead-end; the 0.4294
BC ceiling) and must not jump ahead of the cheap Phase 1 test.

## What NOT to do

Jump straight to Phase 2. Of M22's three null levers, two (search, threat-feature) were cleanly
tested; the teacher was never tested to convergence. Spending an encoder-v5 milestone before one
more cheap leg would be the "commit expensive before testing cheap" error M22 spent all day
avoiding.

## Live monitoring (parallel, no compute)

Three live arms now accruing: B2 (54846434), B3 (54849475), M22c-RL (54864089). Re-run
`notebooks/m22_ab_monitor.ipynb` for the powered read; the refusal gate holds any verdict until a
gap clears its MDE. Offline `rule:dragapult` remains the better-powered signal for weeks —
live is the slow exogenous confirmation, not the primary read.
