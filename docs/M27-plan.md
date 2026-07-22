# M27 plan (STUB — to be refined after 54903635's live read)

**Thesis (from the M26 root-cause analysis, [M26.md](M26.md)):** the deck-out
defect was one instance of a general disease — the clone spends its turn-ending
action (ATTACK/END) while free actions sit unplayed. The per-class report shows
it across TWO classes: ATTACH (fixed by O1 for Telepath) and PLAY, where the
hoarded supporters are the worst cells (Boss's Orders 2/31 = 0.065, Hilda 0.219,
Xerosic 0.250 — top confusion "chose ATTACK instead" ×16). Boss's Orders is the
exact tool that breaks the wall/stall matchups we still bleed to. M27 =
generalize "set up BEFORE you attack" beyond energy.

## Levers, cheapest first

1. **Supporter backstop (pilot arm, O2-family):** chosen action ends the turn
   AND no supporter played this turn AND a supporter the model itself scores
   above threshold is legal → play it first. Riskier than the energy backstop
   (WHICH supporter and WHEN genuinely matter — Boss's Orders vs Hilda are not
   interchangeable), so: per-card sub-arms if the blanket arm fails
   (generic2a/2b precedent), model-scored choice among supporters, own battery.
2. **Evolve backstop:** same shape, targets the chronic `evolve-left` flag
   (teacher-level per M25 flag audit — expect smaller gains; cheap to measure).
3. **Temptation-state weighting (training-side complement):** extend
   `--class-weight` to weight exactly the rows where the teacher chose a FREE
   action while a turn-ending option was on the menu — teaches the ordering,
   not the class frequency. More efficient per reweighted row (M19/M21 dose
   law). Base recipe on `m26_bc_attach5` or fresh; gate vs the current pins.
4. **Turn-phase masking (structural, only if 1–3 underdeliver):** mask
   turn-ending options while unused free value remains, lethal checks pierce
   the mask (plan head already computes `lethal`/`wins`).
5. **Low-deck draw guard (parked, higher risk):** don't take the Dudunsparce
   draw at deck ≤ N unless the plan needs it — deviates FROM the teacher
   (both take it ~60% at deck ≤6; teacher survives by converting faster), so
   override-law risk is higher; needs stall opponents to evaluate.

## Battery hardening

- Extract a real **wall/stall deck** (kangaskhan/crustle family) from replays
  (the grimmsnarl extraction pattern) — the grim clone is advisory-only and the
  wall bleed (teacher-level, 0.31/n=39) has NO offline instrument at all.
- Keep: grim mill opponent (advisory 0.6175), `class_report.py` post-train,
  `offline_behavior.py` A/B probes, control-retrain rule for any per-class claim.

## 🔴 NEW HARD RULE — pre-ship human QC (Piotr, 2026-07-22)

**Before ANY Kaggle submit: play 3 games with the ACTUAL candidate bundle and
save them for the visualizer, then STOP for Piotr's manual review.** No submit
without his explicit go after watching them. Mechanism (exists, zero new code):

```python
from tcg.evaluation import play_games
play_games("submission/main.py", <opponent>, 3,
           replay_prefix="m27_qc_<candidate>")
# -> replays/m27_qc_<candidate>_000..002.html (embedded visualize JSON,
#    one-click open in ptcgvis.heroz.jp) + replays/index.html regenerated
```

Notes: run AFTER `build_submission.sh` export so `submission/` is the exact
ship state (it runs main.py, not the matchrunner twin — this QCs what actually
ships, including flag defaults); pick an opponent that exercises the changed
behavior (for setup-discipline arms: the grim mill clone or a stall deck, not
random); name the prefix after the candidate arm. Rule also added to CLAUDE.md
— it applies to every future ship, not just M27. Rationale: burnt slots are
unrecoverable and offline gates measure win rates, not intent — human eyes on
3 games catch "technically passes, behaves absurdly" defects no gate covers.

## Measurement notes

- **Phase 0 = live read of 54903635 first** (M26 post-ship watch): Telepath
  contested share (~86% expected), deck-out share <50% (from 55%), attach rate
  46.4%→?, grimmsnarl-family record (no change expected — weakness untouched).
  n≥20 before locking conclusions. If the live read surprises (e.g. forced
  Telepath hurts live), M27's thesis needs re-aiming before spending arms.
- Pins to beat: whatever 54903635's battery established — lucario 0.671 n=800 /
  dragapult 0.3375 n=400 / grim 0.6875 n=400 (re-pin after Phase 0).
- Standing: workers ≤8 · one ship per milestone · deck md5 ritual · MDE
  discipline · control retrain before per-class fidelity claims · diary
  incrementally · Hermes at phase transitions.

## Open questions for refinement

- Blanket supporter backstop vs per-card sub-arms from the start?
- Temptation-state weighting on top of attach5's recipe or fresh base?
- Wall-deck extraction: which deck_hash (need the current field re-read)?
- Does the O2 energy backstop get revisited once composed with a supporter
  backstop (the M26 composition lesson says measure, don't assume)?
