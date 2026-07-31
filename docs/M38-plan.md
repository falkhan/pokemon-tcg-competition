# M38 plan — corrected-teacher retrain

Status: **refined v1**, 2026-07-31. Supersedes the stub. Sources:
[m37-post-mortem.md](m37-post-mortem.md), [m37-code-audit.md](m37-code-audit.md),
critique pass 2026-07-31. Piotr's refinement decisions, recorded:
(1) `value_solve` disabled for the gen-1 clean collect; (2) the G1 bar A/B is
folded into the Phase 0 battery; (3) minimal-diff ship, deferred items tracked
in [BACKLOG.md](BACKLOG.md); (4) fine-tune AND scratch as parallel retrain arms.

## Context, in one paragraph

The M37 audit found that the prize-array inversion M36 flagged in `rl/plan.py`
was systemic (4 sites) and that the damaging one was in
`rl/turn_solver.py::score_leaf`: the BC teacher scored every knockout it could
take at −150,000/prize, so for ~30 milestones the expert labels have said
*"don't take prizes unless it ends the game."* That is a precise mechanism for
the two loss modes the post-mortem calls chronic — 30% swept, 30% contested
races lost — and for the agent's characteristic dig/deck-out style. The leaf
fix is in (post-fix: 0/832 prize leaves negative, override 7/18 vs 0/18), but
`MIN_OVERRIDE_SCORE` is still mis-calibrated (P0b): only 1 of the 7 post-fix
overrides is a prize line, the other 6 are outright wins — the corrected
solver is still mostly gagged. `plan_iter._cleared` (rl/plan_iter.py:163)
gates kill-plan labels on that same bar, so the corpus is not clean until the
bar is.

## Success criterion (the number the milestone is judged by)

The ceiling is **817** (M30 and M35, twice), not M37's ~740 equilibrium.

| live outcome | reading |
|---|---|
| implied ELO > 817 | **ceiling breached** — campaign best, hypothesis confirmed at full strength |
| ~750–817 | hypothesis directionally confirmed, ceiling intact — iterate, don't pivot |
| ≤ ~740 | null result — teacher fix did not move live strength; fall back to post-mortem ranked list |

Band mapping: the teacher-fix loss mass (sweeps + contested races) is
concentrated exactly in the 700+ mix (lucario, archaludon, dragapult, rocket
— M37 went 18-19 at 700–799 and 1-3 at 800+). The wall/mirror deck-race mass
is the G4 lane. **Both lanes probably have to land to breach 817** — walls
alone went 3-6 against 697–836 opponents; escaping the band did not escape
the walls.

## Hypotheses

**H1 (main).** The bulk of the "policy quality" loss mass (sweeps + contested
races, 60% of M37 losses) is a *teacher* defect, not a capacity defect. A
corrected and recalibrated solver teacher, plus a regenerated corpus and a
retrain, yields a policy that closes prize races.

H1 makes *behavioral* predictions, pre-registered here and measured with the
post-mortem loss-anatomy classifier (`scripts/pm_extra.py`) on offline gate
replays — not just WR:

- sweep-loss share: down from 30%
- contested-race conversion (me≤4/opp≤2 losses): down from 30%
- prizes taken per game: up
- attack-when-lethal-available rate: up

If WR improves but the loss anatomy does not move, the improvement is not the
hypothesized mechanism and probably won't transfer live.

**H2 (secondary, carried from the post-mortem, audit-independent).** The
wall/mirror deck-race losses are an economy problem that rule-level work
(racemode4, margin raceconserve) can flip; measurable only against a wall BC
clone, not `solver:wall` (strawman-invalid twice over).

## Goals

1. **G1 — recalibrate the override bar**, decided inside the Phase 0 battery
   (see below). Candidate design: a **semantic gate** — track "this line takes
   ≥1 prize or wins" as an explicit boolean through `_dfs` and gate the
   override on that, making the semantics independent of the W-vector forever
   (any constant bar silently re-breaks every time a W-term is retuned).
   Control arm: lowered constant. The pinning test
   (`test_score_leaf_prize_direction_is_not_inverted`) forces the change to be
   deliberate. Note the bar's dual use: `_cleared` gates corpus labels on it,
   so the winner changes label *density* too — tracked per arm in E0c.
2. **G2 — re-pin every solver-backed baseline**, strictly AFTER the G1 winner
   lands (pins made against the wrong bar go stale immediately). All bed pins
   (grim 0.650 / rocket 0.610, `solver:lucario` campaign bar 0.55) are stale;
   fresh controls everywhere (drift law) before any arm is judged.
3. **G3 — regenerate the corpus and retrain.**
   - Gen-1 clean collect with **`value_solve` OFF** (decision 1): the M14
     setup-plan tier scores lines with an old-lineage value net whose notion
     of "good setup" was learned from prize-phobic play — with it on, the
     corpus is not clean. Kill-plans + greedy fallback only. Re-enabling
     setup plans with a retrained value net is a gen-2 question →
     [BACKLOG.md](BACKLOG.md).
   - Fresh `--out` dirs (collect has NO resume), 8 workers max, deck mix from
     the live 700–800 distribution (Q2, resolved below).
   - **Two retrain arms in parallel (decision 4):** fine-tune from
     m28_winners (pre-registered expected winner — the poisoned prior is
     localized to attack/KO decisions and the clean corpus overwrites it,
     while 30 milestones of setup/economy skill is preserved) vs scratch
     (needs the clean corpus alone to re-teach everything). E0c's label-shift
     profile is the tiebreak prior: a shift concentrated on attack prompts
     favors fine-tune.
   - Then the greenlit AWR arm on the same clean corpus (audit sequencing
     rule: AWR on the old corpus fits poisoned labels).
   - Ride-along landing with the retrain (never on the frozen net): the
     `rl/plan.py` wins/concedes fix (P2, ships via v[22]/v[25]) and its
     `submission/rl/plan.py` twin.
4. **G4 — parallel lane: wall BC clone bed** (post-mortem P0, independent of
   the audit) and the rule work it unblocks: racemode4 (wider demote:
   Enriching Energy, Poké Pad, Sacred Ash timing) and the parked M36 universal
   margin raceconserve (mirror bed measurable today, no clone needed).
5. **G5 — re-validate the gacfr3 rule stack against the retrained policy.**
   Include a **no-rules arm** (the raw retrained policy — the correct baseline
   for what the stack still buys), then rebuild additively from gacfr3.
   Ship-relevant outcome is narrow (decision 3): strip a rule only if
   **actively harmful**; redundant-but-harmless rules stay one more milestone
   (stripping is a free follow-up ship → [BACKLOG.md](BACKLOG.md)).

## Phase 0 — merged teacher battery (E0a + E0b + E0c + G1 in one design)

Decision 2: the stub's Phase 0 → Phase 1 split had a sequencing flaw — E0a
would have kill-gated the hypothesis using a solver still known half-broken
(P0b unfixed), a false-kill risk. Merged battery instead. **Arms:**

| arm | leaf | bar |
|---|---|---|
| A0 (control) | old inverted `score_leaf`, vendored behind an env flag | current |
| A1 | corrected | current (`W_PRIZE − 1`) |
| A2 | corrected | lowered constant |
| A3 (candidate) | corrected | semantic gate (prize-or-win boolean) |

Implementation note: vendor the old `score_leaf` behind an env flag
(`M38_OLD_LEAF=1`), NOT a git-pinned old ref — the old ref differs in more
than the leaf (T4 trigger, docstrings) and would confound the comparison. The
env-flag idiom is already proven in this codebase (`M22_WHOLE_BOARD`,
turn_solver.py:87 — spawn workers inherit env but re-import modules).

- **E0a — strength.** Each arm vs **non-solver opponents only**: sample
  agents (tuned/iono/dragapult), rocket/grim BC clones, m28 lineage, mirror.
  Solver-backed beds are confounded here — correcting the solver also changes
  the bed opponent, and the campaign has two documented bed-fidelity failures
  (garchomp, wall) already. n≥800/cell, `--workers 8`. Pre-registered MDE:
  n=800 resolves ~±5pp at 95% — adequate for the large effects hypothesized;
  do not over-read "no difference" on subtle ones.
- **E0b — over-greed probe.** Return-KO walk-in rate per arm. The W-vector
  was tuned while the prize term was inverted; W_COUNTER (−1k×prize) is ~2%
  of W_PRIZE (+100k) and has never had to restrain a live prize reward.
  Outcome feeds a W_COUNTER decision: no walk-in excess → no retune (keeps
  the ship diff minimal); excess → retune lands with the retrain, and the
  principled 2-ply prize-exchange term goes to [BACKLOG.md](BACKLOG.md).
- **E0c — label shift (the primary evidence).** Small collect per arm; diff
  label distributions. Metrics: kill-plan commit rate (`stats["kill_plans"]`),
  labels/game density (guard vs the M8.1 dev-tier trap — a permissive bar
  flooding the corpus with marginal lines), attack-when-available rate,
  supporter-play rate (the chronic under-play line: 10.2%/prompt). Teacher
  game-strength is a weak proxy for label quality in both directions — the
  student distills per-decision labels, not the teacher's winrate under its
  own compute budget — so E0c outranks E0a as evidence for H1.

**Kill gate (joint):** H1 is dead only if **no bar arm beats A0 in E0a AND
E0c shows negligible label shift**. Either one alone is not a kill. Fallback
on kill: the post-mortem's ranked list (wall clone bed, racemode4, margin
raceconserve, band gate refresh).

## Phase 1 — G2 re-pins (with the winning bar), then G3

Collect (fresh dirs, `value_solve` off, 8 workers) → BC arms (fine-tune ∥
scratch) → gate battery: **primary opponents are non-solver** (m28 champion
lineage, sample agents, BC clones incl. the G4 wall clone when ready, mirror
bed) — a student judged mainly by its own teacher inherits teacher-student
correlation; solver beds are secondary/diagnostic. Loss-anatomy deltas
(H1 predictions above) reported alongside WR. Then the AWR arm on the same
corpus. AWR re-smoke (Q3): verify `--outcome-weight` on the new corpus format
AND check the weight histograms — a non-phobic teacher shifts the outcome
distribution, so the effective weighting changes meaning even if the code
runs clean.

## Phase 2 — parallel lane + rules

G4 lane runs from day 1 (clone harvest is independent of everything above).
G5 rules matrix once a retrained candidate exists.

## Ship decision

Minimal-diff ship (decision 3): **retrained net + the `rl/plan.py`
wins/concedes fix, gacfr3 stack unchanged** unless G5 shows a rule actively
harmful. One-variable-per-ship discipline (the racemode3 precedent) — a
bundled ship that lands mid-pack teaches nothing. Everything else discovered
along the way goes to [BACKLOG.md](BACKLOG.md).

Standard pre-ship battery: `scripts/qc_battery.py` (3 games vs
tuned/iono/dragapult) + prev-ship tarball mirror leg + human replay review +
explicit go. `scripts/prize_semantics_probe.py` runs as a standing pre-ship
regression instrument. Deck choice: **ask Piotr** (default assumption
alakazam_v2_h4, but confirm). Monitor row on submit (standing rule).

## Changes required

- `rl/turn_solver.py`: G1 winner (semantic gate candidate: prize-or-win
  boolean through `_dfs`) + deliberate pinning-test update; env-flag vendored
  old leaf for A0; possible W_COUNTER retune per E0b.
- `rl/plan_iter.py`: collect flag to disable `value_solve` if not already
  CLI-reachable; `_cleared` follows the G1 winner.
- `rl/plan.py` + `submission/rl/plan.py`: wins/concedes fix — lands with the
  retrain only.
- New/updated scripts: merged-battery runner (arms A0–A3 over non-solver
  beds); E0c label-shift differ.
- Bed/gate config: re-pinned baselines + campaign bar restated against the
  corrected solver. MILESTONES.md header gets an **epoch marker**: "all
  solver-bed numbers before 2026-07-31 are pre-fix era, not comparable" (R3,
  machine-visible instead of remembered).
- Tests/CLAUDE.md (R4, both halves): add the "no fake-only tests for engine
  semantics" rule to CLAUDE.md, and a small recorded-fixture layer — capture
  real-engine transition traces (the audit's 58/58 probe is essentially this)
  and pin semantic invariants against them in CI.
- Corpus + checkpoints: fresh collect dirs, new BC/AWR ckpts, monitor row on
  any submit.

## Resolved questions (were Q1–Q3)

- **Q1 → both arms** (decision 4). Fine-tune m28_winners pre-registered as
  expected winner; scratch arm keeps the comparison honest. Rationale: the
  poisoned prior is localized (bad prize-closing), not uniform (setup/economy
  play is genuinely good). Relabeling the old corpus was considered and
  rejected even if shards allowed it: the old states were visited under a
  prize-phobic policy, so the corpus under-represents exactly the states a
  race-closing policy must handle (the compounding-error/DAgger argument for
  fresh collection — fixing the state distribution, not just the labels).
- **Q2 → mix from the live 700–800 distribution** (post-mortem n=63 table:
  lucario 13, wall 9, mirror 8, archaludon 8, grim 8, dragapult 4, rocket 4),
  with a meaningful mirror/self-play share (mirror is a live loss mode, 3-5).
  Budget: clean corpus ≥ the size behind the current champion, or the scratch
  arm is handicapped by construction.
- **Q3 → re-smoke, nearly free**, including weight histograms (see Phase 1).

## Risks

- **R1**: corrected teacher may not be stronger *as a pilot* even if labels
  improve. Mitigated by the merged battery + joint kill gate (E0a alone can
  no longer false-kill; E0c is the primary evidence).
- **R2**: a race-closing retrain may regress the deck-out *wins* (4 in M36)
  and the wall matchup shape — G5's no-rules arm and the QC battery cover
  this; net effect is judged by WR + loss anatomy, not by preserving any
  single win mode.
- **R3**: historical cross-milestone comparability is gone for solver beds —
  epoch marker in MILESTONES.md (see changes); never quote pre-fix numbers
  next to post-fix ones in the diary.
- **R4**: `tests/builders.py` fakes can express engine-impossible transitions
  (how the inversion survived 30 green milestones). Adopting BOTH halves:
  CLAUDE.md rule + recorded live-engine fixture layer (see changes).
- **R5 (new)**: gen-1 collect without setup plans may lose label coverage on
  setup-heavy early turns (the greedy fallback labels those prompts). Watch
  the E0c label profile; if early-game quality craters, the gen-2
  value-net-retrain path in BACKLOG.md is the answer, not re-enabling the
  poisoned net.
