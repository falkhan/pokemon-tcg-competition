# M17 Post-Mortem

**Milestone:** M17 (value-first 10k policy collection → retrain → measure)
**Outcome:** 🔴 NO-SHIP — regressed vs M16 on every axis. `checkpoints/osv3o_plan2.pt` kept for review.
**Champion remains:** M16 `osv3o_plan1` (Kaggle submission 54801291), mirror 0.424 / meta 0.534.
**Branch:** feature/m11

---

## 1. The core paradox

M17 trained *dramatically better offline* yet played *worse live* — the textbook distillation / distribution-shift signature. Third recurrence of "more/better data → worse live" (M14 → M15 → M17).

| Metric | M16 (`osv3o_plan1`) | M17 (`osv3o_plan2`) | Delta |
|---|---|---|---|
| Train val_acc | 0.769 | **0.899** | +13pp (offline) |
| Train plan_acc | 0.845 | **0.968** | +12pp (offline) |
| Live mirror (pooled n=800) | 0.424 | **0.380** | −4.4pp |
| Live meta co-gate | 0.534 | **0.404 / 0.396** | −13pp |
| Head-to-head vs plan1 (n=400) | — | **0.400** (160W/240L) | plan1 wins 60/40 |

The offline gain is entirely memorization of the 10k teacher's quirks; none of it transfers.

---

## 2. Confirmation — regression is real, not eval noise

Three independent measurements agree on direction:

| Test | plan2 result | Verdict |
|---|---|---|
| Mirror vs solver (pooled n=800) | 0.380 | ❌ vs 0.424 |
| Meta co-gate (re-eval, n=60/deck) | 0.396 | ❌ vs 0.534 |
| **Head-to-head vs plan1 (n=400, seed1)** | **0.400** | ❌ plan1 wins 60/40 |

When the two nets play *each other* on identical seeds, plan2 wins only 40% — `osv3o_plan2` is objectively the weaker agent. The dominant `mega_lucario_ex+solrock` archetype (90% of meta weight) collapsed to **0.367** (vs M16's 0.517, −15pp), which single-handedly tanks the meta score.

---

## 3. Smoking gun #1 — SETUP plans collapsed to 0

| | SETUP plans committed | games |
|---|---|---|
| M16 collection | **2,420** | 800 |
| M17b collection | **0** | 10,000 |

The new value net (`osv3o_setupval1`, acc 0.800) was supposed to *drive* setup-plan commits (margin 200, turn < 32). It committed **none** across 10k games. The entire "value feeds the planner" hypothesis (M14 → M17) delivered **zero setup labels** this run. We paid the 10k collection cost and got plain expert imitation — plus dilution of M16's cleaner signal.

## 4. Smoking gun #2 — the new value net is mis-calibrated for its one job

- M16 used `osv3_setupval2` (old, 12-id) → 2,420 SETUP commits → shipped 0.424/0.534
- M17 used `osv3o_setupval1` (new, 20-id, 0.800 acc) → 0 SETUP commits

The 0.800 pairwise accuracy is measured on a *different distribution* than the live commit threshold. The 20-id state change pushed the value landscape past the margin-200 gate so it never fires. **The new teacher is worse at its only job.**

> Root-cause statement: `osv3o_setupval1` failed to commit any setup plans, so the 10k collection was plain expert imitation at high volume — overfitting the teacher (train plan_acc 0.968) and diluting M16's signal, yielding a strictly worse agent.

---

## 5. Strengths (proven — keep)

- **Option-identity encoding (M16)** is real and durable — 0.424 is still the best honest neural mirror; the 452-test-green encoder fix holds.
- **Value-retrain pipeline** works mechanically (safari → train → 0.800 acc).
- **Mixed-opponent + runaway-cap collection** is robust: 10k games, no crashes, ~3.5h at 8 workers.
- **The ship-gate worked** — it correctly blocked a −4.4pp regression from reaching Kaggle.

## 6. Weaknesses (broken — fix next)

1. **Teacher labels don't survive distillation** (3rd recurrence). Fidelity law in action.
2. **Value→planner coupling is uncalibrated** (SETUP=0) — the "value teaches setup" loop delivers no labels.
3. **No self-play signal transfers** — expert / EI / DAgger all regress or tie (M9 Leg-2 killed, M11 EI killed, M17 expert regressed).
4. **Meta / cross-deck play is the weakest axis** (0.367 vs the dominant archetype) — and it's the axis the leaderboard rewards most.

---

## 7. Your two curiosities

### A. Better teaching experience (alternatives to solver self-play)

Current teacher = solver pilot self-play, labels = solver's single pick. Limits: ceiling bounded by a frozen heuristic; single-pick labels (no credit assignment); train/serve distribution shift.

Ranked alternatives (data-driven):
1. **Disagreement-weighted DAgger** (M9 Leg-3 pattern; `bc.collect_dagger` infra exists) — label with solver but upweight the states where solver and student disagree (W=10). Targets the states the student gets wrong, not the 95% it already nails. Highest ROI, no new collection.
2. **Value-as-ranker live override** (`rank.py`, M12 — proven pairwise 0.873 offline) — override the student only on confident sibling-ordering errors. Directly fixes the documented "classifier can't rank siblings" root failure. Untried as a live teacher.
3. **Buddy-distillation** (`ext:buddy` spec + `data/external/buddy`) — buddy's 0.615 is hypothesized to come from trainer/energy discipline, the exact measured gap. Distill buddy's sequencing, not our solver's.
4. **Self-imitation from the student's own wins** — `plan_m17b` has 71,277 plan rows, ~50% self-win rate. BC on the student's winning trajectories. Cheap, reuses paid data.
5. **Frontier / GO-exploration teacher** — solver explores with temperature, labels highest-value lines (not just greedy). More diverse labels.

### B. Better self-learning

Current self-learning = none that transfers. The student trains on its own states but is labeled by a frozen teacher → converges to teacher; if teacher ≠ real opponent, it's worse.
1. **Iterative dataset-repair loop** — after each ship, forensic the replays (`forensic_m16.py`), find top-3 defect classes, inject targeted collection oversampling those states (force energy-waste / wrong-attacker positions). Closes the M14 loop.
2. **Generative replay / balancing** — mix `plan_m17b` + `plan_m16` + a slice of `bc_v2b` so the net doesn't forget the M16 distribution (10k volume likely drowned plan_m16's cleaner signal).
3. **PPO re-test from `osv3o_plan1`** — PPO was killed on the OLD encoder with a multiplicative-loss bug later found (M2/M8); that result is confounded. Re-test warm from the option-identity net.
4. **ELO / openskill self-play pairing** — M4 tried, rolled back on noise; re-introduce with meta_v2 co-gate as fitness instead of mirror win-rate (mirror overfits — M4 lesson).

---

## 8. Rated next milestones

Rated on (expected live gain) × (data support) × (1/cost):

| # | Milestone | Rating | Why |
|---|---|---|---|
| **M18a** | DAgger + disagreement-weighting (reuse plan_m17b) | ⭐⭐⭐⭐⭐ | Attacks the proven distillation gap; infra exists; no new collection |
| **M18b** | Re-train `osv3o_plan2` with OLD value net (`osv3_setupval2`) | ⭐⭐⭐⭐ | Isolates the SETUP=0 hypothesis; 1 train run, no collection |
| M18c | Value-as-ranker live override teacher (`rank.py`) | ⭐⭐⭐⭐ | Fixes "can't rank siblings"; 0.873 offline; untested live |
| M18i | Forensic-driven targeted collection (energy-waste / wrong-attacker oversample) | ⭐⭐⭐⭐ | Closes M14 loop; existing tooling |
| M18e | Buddy-distillation (`ext:buddy`) | ⭐⭐⭐ | Targets energy-waste gap directly; buddy 0.615 is real signal |
| M18d | Self-imitation from student's winning trajectories | ⭐⭐⭐ | Uses paid data; low cost |
| M18g | Dataset balancing (plan_m17b + plan_m16 + bc_v2b), retrain | ⭐⭐⭐ | Cheap; may recover M16 signal |
| M18f | PPO from `osv3o_plan1` (re-test, bug confounded) | ⭐⭐⭐ | High upside if bug was culprit |
| M18h | Re-collect with recalibrated value net (fix SETUP=0) | ⭐⭐ | Highest cost (10k collect); depends on M18b first |

---

## 9. Recommended next step — M18a + M18b (run in parallel)

Both reuse the banked 10k games — neither needs new collection.

### M18b — value-net isolation (fastest diagnostic, do first / in parallel)
**Hypothesis:** the new value net (`osv3o_setupval1`) caused the regression by committing 0 setup plans.
**Test:** retrain `osv3o_plan2` variant on the *same* data but re-collect / relabel using the OLD value net `osv3_setupval2` (which produced 2,420 SETUP commits in M16), OR simply retrain on existing data and re-measure to establish the volume-only baseline.
**Decision rule:**
- If SETUP commits return and the regression reverses → the new value net is the sole culprit → recalibrate it (proceed to M18h with a fixed net).
- If not → the regression is pure volume/dilution → prioritize M18a + M18g.
**Cost:** 1 collection + 1 train run (or 1 train run if reusing data). No shipping.
**Gate:** ship only if mirror > 0.424 AND meta > 0.534.

### M18a — disagreement-weighted DAgger (highest ROI)
**Hypothesis:** high-volume expert imitation overfits the 95% of easy states; the student's live losses come from the ~5% of states where it disagrees with the solver.
**Method:** use `bc.collect_dagger` (M9 Leg-2 infra) — student advances the game, solver labels, and **upweight (W=10) the decisions where solver and student disagree**. Train warm from `osv3o_plan1`.
**Why it should work:** targets exactly the distillation gap instead of re-teaching what the net already knows. Reuses banked games where possible.
**Cost:** relabel + train (no fresh 10k collection required for the first pass).
**Gate:** ship only if mirror > 0.424 AND meta > 0.534.

**Sequencing:** launch M18b first (or in parallel) as the cheap diagnostic; M18a as the primary improvement attempt. If M18b shows the value net is the culprit, pivot to M18h (recalibrated net). If not, M18a + M18g carry the milestone.

---

## Appendix — artifacts

- Train logs: `runs/m16_train_10k.log` (plan2), `runs/m16_train.log` (plan1)
- Collection logs: `runs/m16_collect_10kb.log` (plan_m17b, SETUP=0), `runs/m16_collect.log` (plan_m16, SETUP=2420)
- Screens: `runs/m17_screen_s1.jsonl` (0.370), `runs/m17_screen_s2.jsonl` (0.390); M16 `runs/m16_screen_s{1,2}.jsonl` (0.425 / 0.420)
- Meta: `runs/m17_meta_plan2.log` (0.396 re-eval), M16 `runs/m16_meta_s1.log` (0.534)
- Head-to-head: `runs/m17_h2h.jsonl` (plan2 0.400 vs plan1, n=400 seed1)
- Checkpoints: `osv3o_plan1.pt` (M16 champ), `osv3o_plan2.pt` (M17 no-ship), `osv3o_setupval1.pt` (new value net), `osv3_setupval2.pt` (old value net)
- Data: `data/plan_m17` (32 shards, 6.4k games), `data/plan_m17b` (50 shards, 10k games), `data/plan_m16` (800 games)
