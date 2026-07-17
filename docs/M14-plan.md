# M14 plan: setup plans — the value feeds the PLANNER, not an override

Status: **started 2026-07-18 (eve), full pipeline in one session; branch
`feature/m11`.** Diary: [M14.md](M14.md).

## Context

M13 proved the setup value (osv3_setupval2, 0.768 matched-pair / 0.80 in the
setup phase) and killed its override consumer (the override law: 5
measurements — no signal beats greedy at the margin when consumed as live
overrides). M14 gives the value its correct consumer: **plan-head training
targets**. Today the plan head commits only kill plans (~7% of turns; 93%
null). M14's teacher additionally commits a SETUP PLAN whenever the
value-guided search (calibrated margin 200, turn <32 where the value is
validated) finds a line that improves the learned setup value — so the plan
head learns "what to BUILD", and the policy learns to execute builds, on the
turns that were previously plan-blind. The tempo tax the override law
punishes applies only inside data-gen games (accepted: labels are the
product, not the collection win rate); the SHIPPED agent is a pure
plan-conditioned policy forward pass — no live overrides.

**User directives (2026-07-18):** run the full pipeline; play against
MULTIPLE opponents in collection; SHIP the result regardless of the win-rate
gate (live replays are the evaluation — same rationale as the 54779834
observation run); technical integrity gates (parity/isolation/gate-game)
still enforced.

## Pipeline (one pass)

1. **Collector extension** (`rl/plan_iter.py` expert mode):
   - `--opponents` list (specs, rotated per game with the seat-fair pattern):
     `solver:` mirror, `ext:` buddy (0.615 — the pressure opponent),
     `rule:lucario` (the original expert). Recorded seat(s): teacher only
     when the opponent is external (no imitation of third parties — M10 ban);
     both seats in the solver mirror (as today).
   - `--value-ckpt`: per-worker leaf_value (osv3_setupval2, LAMBDA 3000,
     perspective-negated) — at a fresh turn's first MAIN where the KILL bar
     does not clear and turn < 32: value-guided solve (0.5s, margin 200);
     if it clears, commit line + derived plan as a SETUP plan (execute it).
     Stats: kill/setup/null plan counts, per-opponent W/L.
2. **Train `osv3_plan2.pt`**: plan_iter train on [new mixed data + plan_ei0b
   + bc_v2b], init osv3_plan0c, lr 1e-4, 5 epochs (policy CE + plan CE now
   fed by setup plans + value huber).
3. **Ship** (user-directed, win-rate gate waived): `tcg.shipping export
   --checkpoint osv3_plan2.pt --deck lucario` → `gate --agent neural`
   (technical only) → `build_submission.sh --message "M14 setup-plans"` →
   add submission id to the monitor MODELS.
4. **Measure for the record (non-blocking, after ship)**: n=400 vs
   `solver:lucario` + meta co-gate; numbers go to the diary as context for
   the replay-observation round, not as ship criteria.

## Files

`rl/plan_iter.py` (opponents + value-teacher in expert mode), tests,
`docs/M14.md`, monitor MODELS row. Checkpoint `osv3_plan2.pt`.
Collection `data/plan_m14` (~800 games, 12 workers, nice 5 — work day over).
