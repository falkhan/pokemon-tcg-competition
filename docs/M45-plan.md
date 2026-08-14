# M45 — improve the lucario pilot (wide retrain + fix probes + iono-pool PPO leg)

## Context

M44's L-seat kill exposed that the campaign had no lucario pilot; the same
evening's scouting reversed it: 651 seats ≥800 (479 ≥1100) of
`mega_lucario_ex+solrock` (`ed7c14ba…`) were harvested and a BC clone panel
trained — best draw `m45_bc_lucario_d3.pt` (md5 2514a4f3, 8ep) floors
**0.585 vs tuned / 0.372 vs iono**. Piotr's directive: improve it, Track
A+B, ship-oriented (comp ends 2026-08-16; two Kaggle slots available
tomorrow). Both tracks end in the full ship ritual + Piotr's replay review
+ explicit go + deck confirmation — nothing submits without it.

**The three evidence pins this plan is built on (docs/M44.md, exploration
2026-08-13):**
1. `data/bc_m45_lucario` is ALREADY width-143 (40,451 decisions / 579
   episodes — 1.75× the corpus that yielded m41b's +2.78pp width gain; 4×
   the ogerpon corpus where width FAILED at 10k decisions). The d3 clone
   was trained `--init m28_winners` (width 100) and silently truncated all
   43 wide columns (`rl/policy.py:218-219`), including the
   matchup/energy-ceiling blocks a scaling-damage opponent needs.
2. Loss anatomy from the floor jsonls (M44 3g telemetry): vs iono 92% of
   losses are prize-race (out-damaged — Voltaic Chain scaling is invisible
   to `rl/combat.py`'s damage model); vs tuned 14% bench-out losses, 60%
   of wins by opponent deck-out. The live fix default
   (`conserve,racemode2,racemode4,planzero`) is INERT on this deck except
   planzero + a Poké-Pad sliver of racemode4. Generic fixes that CAN act:
   `benchfloor` (targets the bench-outs), `gustveto` (the M36 anti-scaling
   blanket demote — built for exactly the iono failure), plus
   backstop/deadenergy/gustsnipe/tempo.
3. The only documented mover of the iono cell is training WITH iono in the
   pool (K's 0.603 vs everyone else's ≤0.42; "you become what you train
   against", `rl/collector.py:63-66`). And the M44 K-lesson: in-leg
   vs_teacher promotion is a screen that can anti-correlate with the broad
   field — ship decisions come from the 25-bed gate diagnostic, never from
   vs_teacher.

## Milestone hygiene (first actions)

- Branch `feature/m45` off `feature/m44` (M45 reuses M44's league/ritual
  scripts; PR stacks on #31 or lands after it merges).
- Write `docs/M45-plan.md` (this plan, r1) + start `docs/M45.md` diary —
  incremental diary entries at every result, Hermes at stage transitions,
  BLAS pinning env on every matchrunner/collector run.
- Adopt the ship deck file: copy `data/kaggle/lucario_ed7c14ba_deck.csv` →
  `decks/lucario_solrock.csv` (tracked; hash-verify roundtrip
  `ed7c14ba…`). All subsequent runs reference the decks/ path.

## Track A — wide retrain + fix probes → ship slot 1 (morning, ~2.5h)

**A1. Wide panel (G-13: 3 draws), ~35 min GPU.** For (epochs, seed) in
(14, 3), (11, 1), (8, 2):
```
uv run python -m rl.plan_iter train --data data/bc_m45_lucario \
  --name m45_lucario_wide_e<EP>s<S> --init-wide checkpoints/m45_bc_lucario_d3.pt \
  --epochs <EP> --seed <S> --lr 3e-4 --outcome-weight 0.25
```
`--init-wide` = `widen_option_dim` warm-start (`rl/plan_iter.py:1393-1398`,
:869 — appended columns zero-init, epoch-0 ≡ d3 exactly). `--outcome-weight
0.25` because the corpus is NOT winners-only (m41b/m43 convention,
docs/M43.md:232-238). Diary val_acc per draw immediately (checkpoints store
no metadata).

**A2. Floors on all 3 wide draws** (reuse the m44_step4 pattern): each vs
`rule:tuned:lucario` AND `rule:iono`, n=400, seed 1, `model-pz:<ckpt>:decks/lucario_solrock.csv`,
resumable jsonls `runs/m45_wide_<name>_vs_<anchor>.jsonl`. **Selection: best
pooled (tuned+iono) wr; kill any draw that regresses BOTH anchors vs d3**
(d3 baseline: 0.585/0.372). If ALL wide draws regress → wide is dead for
this lineage (diary the kill; Track A continues with d3).

**A3. Fix probes on the A2 winner** (single-variable, ~25 min): arms
`model-pz-bf` (planzero+benchfloor) and `model-pz` + gustveto… fix-kind
tokens: check `_MODEL_FIX_KINDS` for existing planzero+benchfloor /
planzero+gustveto kinds; if absent, add two entries (2-line additive change
to `rl/matchrunner.py`, mirroring `model-pz-snipe`). Each arm vs both
anchors, n=400. **Adopt a fix only if pooled Δ ≥ 0 AND neither anchor
regresses by more than one CI (±4.9pp)** — at n=400 these are
behavior-shaping reads, not strength claims; the QC replays are the real
review surface. Default ship fixes = `planzero` alone if both probes are
neutral-negative.

**A4. Gate diagnostic** (advisory, informs Piotr): spec
`docs/specs/m45_A.json` — arm = winner config on `decks/lucario_solrock.csv`,
control = `model-c-pkgz:checkpoints/ppo_best_m43a_base.pt:alakazam_v2_h4`
(the m43a incumbent — same cross-pair instrument O passed at +5.51pp), beds
verbatim from `m43_laneA_base.json`, n=400, seed 1, bars {pass 0.02, kill
0.0} advisory. Generate via the `scripts/m44_make_gate_spec.py` pattern
(point it at an m45 roster entry or inline spec). ~15 min. A kill →
escalate with numbers (Piotr's menu), never silent.

**A5. Ship ritual** (reuse `scripts/m44_ship_ritual.sh`, add an `L45` case):
export `--deck lucario_solrock --fixes <adopted>` → tarball deck md5-check →
`ship_verify --checkpoint <winner> --deck lucario_solrock --corpus
data/bc_m45_lucario --gate-arm <arm token>` (corpus is w143 + zero plans →
6a/6b/6c pass cleanly, no exception needed) → `qc_battery --prefix
m45_qc_A --prev dist/submission_neural_20260813_145134.tar.gz` (the M44
ogerpon ship as mirror leg) + bespoke leg vs the tuned sample agent's
sister-list matchup is already the battery's `tuned` leg; add bespoke vs
the m44 alakazam tarball. → **STOP: Piotr replay review + explicit go +
deck confirmation (lucario_solrock) → submit slot 1.** MODELS + DECK_META
(`lucario_solrock`: ace Mega Lucario ex, new emoji) in the ship commit.

## Track B — iono-weighted PPO leg → opportunistic ship slot 2 (afternoon, ~2.5h)

**B1. Leg** on the Track-A winner net (wide if it won, else d3), 15
iterations, tag `m45_L_r1`, via `scripts/m44_run_leg.sh` machinery
(heartbeat + watchdog + pause/resume — battle-tested 4/4 today). Pool
(sums 1.00, iono-heavy per the documented mechanism):
```
--opponents "model-c-pkgz:checkpoints/ppo_best_m44_K_r1.pt:alakazam_v2_h4=0.10" \
            "model-pz:checkpoints/ppo_best_m44_O_r1.pt:ogerpon=0.10" \
            "model-pz:checkpoints/ppo_best_m44_G_r1.pt:grim_live=0.10" \
            "mirror=0.25" "past=0.10" \
            "rule:tuned:lucario=0.15" "rule:iono=0.20"
```
`--start <winner>.pt --learn-deck decks/lucario_solrock.csv --eval-deck
decks/lucario_solrock.csv --kl-coef 0.1 --eval-every 5 --eval-games 200
--games-per-iter 400 --workers 8 --device auto`. Dragapult stays held out
(collector hard-refuses). Operator inner kills per M44 (entropy not
rising, KL bounded ≤~0.15, no loss cause rising while wr improves — watch
`loss_cause/prizes` vs iono-heavy pool).
- Note: `scripts/m44_leg.py` is roster-driven; either add an m45 roster or
  launch rl.ppo directly with the flags above (simpler — the leg is a
  one-off; keep m44_run_leg.sh's wrapper by parameterizing the command).

**B2. Read**: in-leg promotion (vs_teacher) is a SCREEN ONLY. The decision
instrument is the same gate diagnostic as A4 with arm = `ppo_best_m45_L_r1`
(spec `docs/specs/m45_B.json`) **plus the two floor anchors n=400** (did
iono actually move? — the entire point of the leg). Ship slot 2 only if:
gate ≥ Track A's number − 1 SE AND iono floor improved. A leg that
regresses the gate (the K/G pattern, 2-of-3 prior) → diary the kill, no
second ship, slot stays banked.

**B3. If shipped**: full A5 ritual again (`m45_qc_B`), STOP for Piotr, its
own MODELS entry.

## Kill criteria / stop rules

1. All wide draws regress both anchors → wide dead; continue on d3.
2. Gate kill (≤0.0 vs incumbent) on any ship candidate → Kill-criterion-4
   style escalation to Piotr with numbers (options: ship anyway / hold);
   never silent.
3. Leg inner kills → keep the Track-A net; Track B produces no candidate.
4. Standing law: offline = screening, live decides. Max 2 ships. QC +
   Piotr review never waived.
5. External wrapper kills (4 today): resume via `--resume` / cell-file
   deletion per the established procedures — not failures.

## Files touched (all reuse, minimal new code)

- NEW: `docs/M45-plan.md`, `docs/M45.md`, `decks/lucario_solrock.csv`,
  `docs/specs/m45_A.json`, `docs/specs/m45_B.json` (+ maybe
  `docs/specs/m45_roster.json` if the leg goes roster-driven)
- EDIT (small, additive): `rl/matchrunner.py` `_MODEL_FIX_KINDS` — up to 2
  new kind tokens for the fix probes (mirror `model-pz-snipe`, one line
  each + a parse test assert if the existing tests pin the dict)
- EDIT: `scripts/m44_ship_ritual.sh` — an `L45` case (ckpt/deck/fixes/
  corpus per A5); possibly `scripts/m44_make_gate_spec.py` roster-path arg
- REUSE UNCHANGED: `rl/plan_iter.py --init-wide`, `rl/matchrunner play`,
  `scripts/gate_spec.py`, `scripts/qc_battery.py`, `scripts/ship_verify.py`,
  `scripts/m44_run_leg.sh`/`m44_heartbeat.sh`, `rl/ppo.py --resume`
  machinery, `build_submission.sh`

## Verification

- A1: each draw's epoch-0 val reproduces d3 (the widen invariant prints in
  plan_iter); diary val_acc per draw.
- A2/A3: floor jsonls decoded by the measure-agent law (0=a-win); compare
  against d3's 0.585/0.372 with ±4.9pp CIs in mind.
- Fix-kind edits: `uv run pytest tests/test_matchrunner.py -q` (parse
  tests), plus a one-off `parse_spec` roundtrip of the new tokens.
- A4/B2: `gate_spec.py hash → run → decode` (pre-registered, resumable);
  full suite `uv run pytest tests/ -q` before any ship commit.
- A5/B3: `ship_verify` ALL PASS required (no registered exceptions this
  time — the corpus matches the serve width and has zero plans); QC
  battery zero crashes; sweeps investigated before the go.
- Timeline check: A ships late morning 08-14; B decision ~16:00; comp ends
  08-16 — both ships accrue ≥1.5 days of games.
