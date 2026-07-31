# Backlog — deferred items for future milestones

Standing doc (created 2026-07-31, M38 refinement pass, Piotr's decision 3:
minimal-diff ships, everything else tracked here). One line of origin per
item so the context survives; strike or move items into a milestone plan when
they're picked up.

## Deferred from M38 by design

- **Gen-2 collect with a retrained value net.** M38 gen-1 runs with
  `value_solve` OFF because the M14 setup-plan tier scores lines with an
  old-lineage value net trained on prize-phobic play. Once a clean-corpus BC
  exists: retrain the value head on the clean corpus, then decide (with a
  measurement) whether the setup-plan tier still earns its keep in a gen-2
  collect. Watch item feeding this: M38 R5 (early-game label quality under
  greedy fallback).
- **Rule-stack strip.** G5 strips only actively-harmful rules from gacfr3;
  redundant-but-harmless ones ride one more milestone. A pure-strip submit is
  a cheap single-variable follow-up ship once the retrained net's live
  behavior is known.
- **W_COUNTER principled retune / 2-ply prize-exchange term.** If E0b shows
  return-KO walk-ins, the minimal fix (constant retune) ships with the
  retrain; the principled version — score net prize exchange (my prizes taken
  − expected return-KO prize value) instead of a flat counter penalty — is a
  separate solver milestone.
- **`setup_plans_late` (M33) re-evaluation.** Late-turn (≥32) setup commits
  are exactly the deck-out-adjacent behavior the old objective rewarded;
  re-justify the tier against a race-closing policy before it returns in
  gen 2.

## From the M37 audit (logged, not scheduled)

- **Silent fix-name ignore**: an unrecognized name in `_ATTACH_FIXES` is
  silently dropped — nothing would catch a typo in a ship config. Minor
  hardening: fail loudly on unknown names.
- **Kangaskhan-only wall blindspot**: racemode3/4 trigger needs a wall
  Pokémon visible on board (`_RACEMODE_WALL_IDS`); a Kanga-only variant is
  invisible to it. Revisit when a live sample shows one.
- **`rl/` ↔ `tcg/` duplication** (~200 KB, seven pairs, manual parity tests)
  — the unfinished M6 migration. Documented in the audit; needs a dedicated
  housekeeping milestone, not a ride-along.

## From the M37 post-mortem (ranked list, below the M38 cut)

- **Band gate refresh**: re-freeze `band_decode.py` weights on the n=63
  700–800 sample (the 600-band weights are one milestone stale). Cheap;
  candidate ride-along for any M38/M39 ship commit.
- **Stop-investing list** (do not resurrect without new evidence): grim
  deck-tech (6-2 live, self-resolved), gustveto (2 residual plays), hop beds
  (absent at altitude), brick residual (2 losses, unactionable), starmie
  (solved).
