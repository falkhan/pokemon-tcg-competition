# M8 campaign report — "Beat the ship agent" (setup play)

**Branch:** `feature/m8` · **Period:** 2026-07-12 → 07-13 (overnight) ·
**Diary:** [M8.md](M8.md) · **Spec:** [M8-plan.md](M8-plan.md)

## TL;DR

- **Campaign bar:** any agent ≥ 0.55 vs `solver:lucario` (n ≥ 800). **Best
  artifact: 0.359** (`ppo_m83_legB_it5.pt`) — the gap is real but moved ~12
  points in one day (osv2_bc was at 0.237).
- **The single biggest finding:** both failed M7 PPO attempts trained from a
  warm start that had CLONED the play-tier bug ("can't use trainers" was
  literally imitated). Re-cloning the fixed teacher (`osv2_bc2`) jumped
  **+10.8pp vs the solver with zero RL**.
- **PPO's pathology is cured, its promise is falsified:** with dev-potential
  shaping + lr 3e-5 the entropy diffusion and rise-then-decay are GONE — but
  8-iter probes climb nothing durable (0.359 @ n=800 vs 0.345 baseline). The
  policy-gradient signal on sparse TCG outcomes is too weak to move a
  BC-anchored policy. Recommendation: stop buying PPO iterations.
- **Both L4 (AlphaZero-lite) instruments now exist and pass their gates:**
  the L3 archetype determinizer infers the opponent's exact list by turn 2
  (top-1 = 1.000), and the value head retrained on search-visited states goes
  0.57 → 0.85 sign-accuracy (the measured MCTS-killing OOD failure, fixed).
  The sims-ladder reading is the final input — see below.
- **Kaggle:** submission **54621283** (solver-on rules agent, all five pilot
  fixes) went live tonight (commit `e3caadd`). Nothing produced overnight
  beats it, so no duplicate submission was made.
- **Search-at-inference for SETUP was killed honestly:** the M8.1 dev tier
  lost its A/B twice (pooled 0.492, n=1600). Post-fix greedy already develops
  well; search only pays where greedy is structurally blind (multi-step
  lethal).

## The instrument bugs this campaign caught

Measurement came first (M8.0), and it paid: three separate instruments were
broken and are now fixed —

1. **Post-mortem select/action off-by-one.** In every replay shape, the
   action recorded at step *i* answers the select at step *i−1*; the tool
   paired same-index, reading every per-step flag one prompt late. All
   M8-era taxonomy numbers use the fixed pairing.
2. **`--seed` was dropped on the parallel path** (and never drove game
   randomness anyway — the engine RNG does). Seeds now thread through;
   recorded seeds identify runs, not game streams.
3. **`value_train`/`mcts` lacked the pre-M3 dimension shim** for bc_v1-era
   checkpoints; both are dimension-aware now.

Plus the process lesson (memory + diary): spawn workers re-import from disk —
never edit a module a live run imports (cost: leg A's iters 6–7).

## Results by milestone

| Milestone | Verdict | Key numbers |
|---|---|---|
| M8.0 measurement reset | ✅ | Baselines vs today's pilots: osv2_bc 0.237/0.297 (solver/generic); taxonomy: attach-off-racer 55, hand-discard 38, trainer-hoarded 21 per 100 games |
| M8.1 setup-mode solver | ❌ killed (per plan) | A/B vs lethal-only 0.526 → 0.458 (pooled 0.492, n=1600); expert regression both passes; terms migrated to M8.3 shaping |
| M8.2 BC refresh | ✅ pass | `osv2_bc2`: 0.345 vs solver (+10.8pp), 0.350 vs generic (ref 0.297), G2 0.845; solver-teacher probe: fidelity −5.6pp → generic teacher stays |
| M8.3 PPO probe | ⚖️ neutral (closed) | Leg A (clean start): 35→33→27.5→33, no climb. Leg B (+dev shaping, lr 3e-5): entropy FLAT first time; peak eval 40% (n=200) verifies to **0.359 @ n=800** (+1.4pp = noise) |
| M8.4(a) L3 determinizer | ✅ gate pass | Top-1 archetype 1.000 from turn 2 (bar ≥0.9 by t4); containment-first matching (cosine mis-ranks sparse reveals — measured) |
| M8.4(b) value-on-search-states | ✅ gate pass | Sign-acc 0.57 → **0.85** on search states (0.57 = the quantified OOD failure) |
| M8.4(b) sims ladder | ❌ flat → **L4 no-go** | 0.515 / 0.490 / 0.495 at 16/32/64 sims (n=200 ea) — the fixes removed "more sims = worse", but search adds nothing over the policy head |

## The M8.5 decision (yours): L4 AlphaZero-lite go/no-go

Everything the review needs is now measured:

**For (go):** both documented causes of the M2 MCTS failure are fixed and
verified (determinization: 1.000 by t2; value OOD: 0.85). The PPO alternative
is falsified under best-known conditions. The stable training loop +
`osv2_bc2` base + L3 determinizer are exactly the AZ-lite ingredients, and
Kaggle's latency budget (600 s/game vs our ~3 s) leaves room for
search-at-inference while distillation catches up.

**Against (no-go), and this is how it came out:** the sims ladder — the
direct test — is FLAT with both fixes in (0.515/0.490/0.495). A 0.85
sign-accuracy win/loss classifier still can't rank sibling actions whose
margins sit below its noise floor; the M2 diagnosis survives both repairs.
**Recommendation: NO-GO** on the full AZ-lite loop for this hardware budget.
The salvage is real, though: the stable PPO loop, the L3 determinizer
(useful for ANY future search or opponent-modeling work), and the
search-state value collection are proven, tested components on the shelf.

**Fallback if no-go:** the rules agent keeps improving through post-mortem
engineering (five fixes shipped this week, every gate a series best), and the
team's deck-analysis pipeline (per-deck BC + gates) is the next lever — the
"any agent that wins" bar doesn't care that the winner is rule-based.

## Ship state

- Live: **54621283** — solver-on rules bundle, all five M7.5 pilot fixes
  (play-tier, ATTACH_FROM, Carmine, fetch, empty-bench). Shipped by Piotr
  tonight; expected to climb from 487.9-era scores.
- Neural ship path (v2 numpy bundle) is built, gated, and one
  `./build_submission.sh --checkpoint <ckpt> --deck lucario` away whenever a
  checkpoint clears the league gates.

## Reproduce any number

Every result in this report has a diary entry in [M8.md](M8.md) with the
exact command, n, and seed. Long measurements support `--checkpoint FILE`
resume. `./scripts/m8_watch.sh` shows live run progress.
