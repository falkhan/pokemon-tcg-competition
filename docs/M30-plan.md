# M30 plan — deterministic tempo/deck-economy rules (O-family) + single-teacher data arm

## Context

M29 (sub 54914673, pooled same-deck winners) resolved ~65 ELO **worse** live than
54903635 at equal game count (665 vs 735 @ n=44), despite a real, replicated
offline dragapult gain. The post-mortem (`docs/m29-post-mortem.md`) localized the
regression to item tempo and deck economy: END-with-playable-item 12.0%→26.2%,
telepath attach 84.9%→67.7%, deck-out losses 19%→30%, and a case-study loss
holding 36 cards at deck-out. The one intervention class with a live-validated
win is the deterministic override rule (**O1 telepath — the M26 ship**: 54903635
= `m25_bc_alakazam_v3h` + O1, identical weights to 54897966, rule the only
variable, and it won live). BC cannot fix the item defect — the teacher itself
declines these items (M27 instrument), so imitation is capped.

Labeling correction to carry through: 54903635 is the **M26** ship
(docs/M26.md:202, DECISIONS.md:51); the post-mortem's "M28-O1" label is wrong —
fix `docs/m29-post-mortem.md` in P0.

**M30 goal:** extend the O-family with three deck-economy rules, run them on BOTH
candidate bases (Piotr's call: battery decides between `m25_bc_alakazam_v3h` and
`m28_winners`), plus a parallel single-teacher data arm from 54773249's 334
cached wins (Piotr approved; confirmed cached, no refresh needed).
Pre-registered bars, no judgment-call ships — M29's process failure does not
repeat.

**Key structural fact:** arm 1 (B1 + O1 + surviving rules) is a strict superset
of the live-best config. If every new rule dies in P3, arm 1 degenerates into an
exact 54903635 re-ship — so the post-mortem's open rollback decision is
implicitly resolved INSIDE the battery, not a separate track.

Milestone bookkeeping: first commit the uncommitted post-mortem artifacts
(docs/m29-post-mortem.md + docs/M29.md entry + memory-driven episode-cache
changes stay untracked as data) on `feature/m29`, then branch `feature/m30`
from it (merging M29 to main via PR is separate housekeeping for Piotr). Copy
this plan to `docs/M30-plan.md`; diary incrementally in `docs/M30.md`; Hermes
updates at phase transitions; workers ≤8 everywhere; 0=WIN decode.

## The three new rules (fix names)

All follow the O1 pattern: deterministic, card-fact/id-pinned, reorder the
model's ranked option list, no value-signal consumption (outside the killed
M8.1/M12/M13 override class). Card ids verified against `decks/clone54618168.csv`:
Buddy-Buddy Poffin **1086** ×4, Poké Pad **1152** ×4, Sacred Ash **1129**,
Dudunsparce **66** ×2 (no ex variants in the deck).

- **`tempo` (O3):** at MAIN, when the model's top choice is END and a PLAY
  option for Poffin/Poké Pad is legal → force the highest-ranked such PLAY.
  END-only first (targets the measured 26.2% defect exactly); an END+ATTACK
  variant only if END-only doesn't move the metric. Honest precedent note:
  O2 backstop — the ATTACH-class "force before turn-end" analog — was parked
  non-inferior-no-gain in M26; O3 is its PLAY-class cousin. Not disqualifying
  (the item defect is measured, the attach one was already fixed by O1), but
  it is the closest prior and the null outcome is live.
- **`deckguard` (O4):** when own `deckCount <= 6` and the top choice is the
  Dudunsparce ABILITY → demote it below the best non-Dudunsparce option.
  Threshold matches `attach_probe.py`'s existing low-deck (≤6) instrument.
- **`ash` (O5):** when own `deckCount <= 10` and a PLAY option for Sacred Ash
  exists → force it (engine legality implies Pokémon in discard; it shuffles
  Pokémon back into the deck — the anti-deck-out tech already in the 60).

P0 preconditions (do not trust printed-card memory or untested resolution):
1. Verify engine semantics of the Dudunsparce ability and Sacred Ash from
   cached replays (deck-count delta on use).
2. Verify **ability-option card-id resolution**: `_hand_card_id`
   (rl/plan.py:267) only resolves HAND-area options; ABILITY options carry
   board `area`/`index` (ACTIVE/BENCH). `deckguard` needs the board-resolution
   pattern (as in `sub_behavior.py`'s ability attribution) — confirm ability
   options' fields on real observations before writing the predicate.

## Phases

### P0 — instruments + semantics first (no rule code until these exist)

1. Promote the post-mortem probe to `scripts/loss_forensics.py` — copy it out
   of the session scratchpad on /tmp FIRST (it does not survive reboots).
   Kaggle-replay side instrument.
2. Extend `scripts/offline_behavior.py` (engine-game side) with the
   **END-with-playable-item rate** counter and a **per-side deck-out split**
   (it already computes deck-out losses and telepath attach rate — only these
   two additions are new).
3. **Calibrate offline baselines:** run the extended `offline_behavior.py` on
   the exact live config (`modelt:m25_bc_alakazam_v3h:clone54618168`) vs grim
   clone and `rule:lucario`, n=60 each. The live-replay numbers (12.0% / 84.9%)
   may not transfer to engine games, so ALL behavior pins in P3/P4 are
   RELATIVE to this measured O1-only offline baseline, not to live-replay
   absolutes.
4. Replay semantics checks (Dudunsparce/Sacred Ash deltas, ability-option
   field resolution). Diary findings.
5. Fix the "M28-O1" → "M26 ship" label in `docs/m29-post-mortem.md`.

### P1 — rule implementation

- `rl/plan.py`: add id constants (`POFFIN_ID`, `POKE_PAD_ID`, `SACRED_ASH_ID`,
  `DUDUNSPARCE_IDS = {66}`) and a new `apply_play_overrides(obs, ranked,
  fixes)` beside `apply_attach_overrides` (rl/plan.py:279). Reuse
  `_hand_card_id` (rl/plan.py:267) for PLAY options; add a board-area resolver
  for ABILITY options per the P0 finding. Runs AFTER attach overrides; its
  promote-rules fire only when `ranked[0]` is turn-ending, so an O1-promoted
  ATTACH naturally takes precedence; `deckguard` is a demotion and composes
  with anything.
- Wire both call sites (single-source-of-truth pattern,
  `tests/test_attach_override.py` docstring):
  `submission/main.py:218-221` (after argsort, before maxCount truncation —
  same `_ATTACH_FIXES` env set, vocabulary just grows) and
  `rl/matchrunner.py` closures `fn4` (~:428) + `fn3` (~:496).
- `rl/matchrunner.py:63-68` `_MODEL_FIX_KINDS`: add per-arm spec kinds —
  `modelt-tempo` (O1+tempo), `modelt-guard`, `modelt-ash`, `modelt-eco`
  (O1 + all three). Hyphenated kinds parse fine through the colon-split.
- Sync the byte-identical twin `submission/rl/plan.py` (pinned by
  `test_bundle_twins`).
- Tests in `tests/test_attach_override.py` style (SimpleNamespace obs): one
  per rule firing + one per guard (END-only, deck thresholds, non-MAIN no-op,
  composition with O1), extend the spec-kind parse test and twin-bytes test.
  Full `uv run pytest` green before any games.

### P2 — data arm + rocket clone (parallel with P1, pure compute)

- **Arm D corpus:** `uv run python -m rl.replay_bc build --only-subs 54773249
  --deck-hash 9294d9d8 --winners-only --min-score 600 --hand-aware
  --out data/bc_m30_54773249_winners` (334 wins already cached).
  Train with the exact m28_winners recipe (`scripts/m28_fg_train.sh` pattern):
  `plan_iter train --data ... --name m30_54773249_winners --init-v3m
  checkpoints/m27_both.pt --epochs 10 --lr 1e-4 --card-kind-weight SUPPORTER:5
  --card-kind-weight STADIUM:5`. Corpus is the only variable vs m28_winners.
- **Team-rocket clone (validation-gated, NOT a gate):** corpus exists
  untrained — `data/bc_m24_54834745` (teacher score 1156, deck-hash prefix
  `3394cd30`, ~15k decisions). One command: `plan_iter train --data
  data/bc_m24_54834745 --name m30_bc_rocket_54834745 --epochs 10 --lr 1e-4`.
  Export its deck by **prefix-filtering** `opp_decks.parquet` on
  `deck_hash.startswith("3394cd30")` + `_write_deck_csv`
  (rl/kaggle_ingest.py:88,97) → `data/kaggle/rocket_3394cd30_deck.csv`.
  **M28 strawman law applies:** before it may serve even as an advisory, it
  must reproduce the live loss mode — vs the live-pin config it should produce
  long games and push our deck-out share up (we are 0–3 live vs this family).
  If it doesn't, diary the kill and use it only as a long-game behavior test
  bed for the rules, never as a pin. Either way the kangaskhan/crustle family
  stays unmeasured in M30 — the coverage hole is narrowed, not closed; say so
  in the diary.

### P3 — cheap behavior screens (kill rules that don't move their target)

For each single-rule arm on base B1, vs the grim mill clone
(`model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv`)
and `rule:lucario`, n=60 each via the extended `offline_behavior.py`, compared
against the P0 offline baseline of the same base+O1:

- `tempo` must cut the offline END-with-playable-item rate materially vs
  baseline.
- `deckguard` + `ash` must cut deck-out-loss share in the grim matchup vs
  baseline.
- Any rule that does not move its target metric → dropped, diaried (kills are
  results). These are behavior counters only — strength is decided in P4.

### P4 — full battery, pre-registered split bars

Arms (clone54618168 deck everywhere; new `scripts/m30_p4_battery.sh` /
`m30_p4_decide.py` from the m29 patterns — with the battery ORDER inverted per
post-mortem rec #5: dragapult + kyogre floors and grim first, the n=800 mirror
LAST and only for arms still alive):

1. **B1 + O1 + all survivors** (`modelt-eco:...m25_bc_alakazam_v3h...`)
2. **B1 + O1 + tempo only** — pre-registered fallback against the M26
   composition trap ("one precise intervention beats three stacked ones"): if
   the composed arm underperforms, this attributes whether composition dragged
   it, without burning another milestone. Runs even if all rules survive P3.
3. **B2 + O1 + survivors** (m28_winners base)
4. **D + O1 only** (`modelt:...m30_54773249_winners...` — isolates the corpus)

Each: `rule:dragapult` 2×200, `random:kyogre` 200 (+ second seed if edge-read),
grim clone 2×200 (advisory), then `rule:lucario` 4×200 (n=800). Pins = live
54903635: luc 0.671 / drag 0.3375 / grim 0.6875 / kyo ≥0.95.

**Pre-registered ship bars — SPLIT by intervention class (no goalpost moves,
no judgment-call override):**

- **Rule arms (1–3): non-inferiority bar** — mirror z > −1.96 vs 0.671 AND
  dragapult z > −1.96 vs 0.3375 AND kyogre ≥0.95. This relaxation vs
  m29_p4_decide.py's resolved-better rule is deliberate and valid ONLY here:
  same-weights rule ships have a clean live A/B and the claimed gain is
  behavioral (the M26 O1 precedent). Plus the behavioral claim itself, offline
  and relative: END-with-playable-item strictly below the same base's O1-only
  baseline AND deck-out-loss share strictly below it AND telepath attach not
  degraded.
- **Arm D (weights change): the M29 original bar, un-relaxed** — mirror
  RESOLVED-better (z > +1.96 vs 0.671) AND dragapult non-inferior AND kyogre
  ≥0.95. A weights change is exactly the class that produced the M29
  regression, and non-inferiority at n=800 (MDE ~7pp) structurally cannot
  detect a 65-ELO-class live drop. If arm D fails this bar it goes to Piotr as
  evidence only, never a ship candidate.
- **Tie-break: a qualifying rule arm beats arm D**, always (weights changes
  need live evidence rule arms don't). Among rule arms: best deck-out-share
  improvement, then grim. Dragapult is explicitly NOT the tie-breaker — it was
  M29's trap axis.
- If NO arm clears: evidence to Piotr, no ship, no exceptions — this is the
  M29 lesson, in writing, before the numbers land.

### P5 — QC + STOP (hard rule)

Review `verify-before-consequential-actions.md` first. Then:
`./build_submission.sh --checkpoint <winner>.pt --deck clone54618168` + md5
ritual (deck.csv md5 8e8cf124; assert the fix-name default engaged in the
bundled main.py by direct import BEFORE build). QC = 3 games of the actual
bundle via `tcg.evaluation.play_games("submission/main.py", <grim mill clone —
the opponent that exercises deck economy>, 3, replay_prefix="m30_qc_<arm>")`.
**STOP for Piotr's manual replay review and explicit go. No submit without it.**
(If the ship default env string changes — e.g. `"telepath,tempo,ash"` — that is
part of the sign-off, per the main.py:100 comment convention.)

## Files to touch

- `rl/plan.py` + `submission/rl/plan.py` (twin) — new constants +
  `apply_play_overrides` + ability-option resolver
- `submission/main.py` (~:218) — second override call
- `rl/matchrunner.py` (:63-68, ~:428, ~:496) — spec kinds + closure wiring
- `tests/test_attach_override.py` — new rule/guard/composition tests
- `scripts/offline_behavior.py` — END-item counter + per-side deck-out split
- `scripts/loss_forensics.py` (new, rescued from /tmp scratchpad),
  `scripts/m30_p4_battery.sh`, `scripts/m30_p4_decide.py` (new, from m29
  patterns, floors-first ordering, split bars)
- `docs/m29-post-mortem.md` (label fix), `docs/M30-plan.md`, `docs/M30.md`,
  `docs/MILESTONES.md`

## Verification

1. `uv run pytest tests/test_attach_override.py -v` then full suite — green
   before any game runs.
2. P0 semantics + resolution checks diaried with replay evidence; offline
   behavior baseline of the live config measured and pinned as the relative
   reference.
3. P3 screens show each surviving rule moves its target counter vs the P0
   baseline; kills diaried.
4. P4 battery decoded by the pre-registered `m30_p4_decide.py` (0=WIN, z/MDE
   quoted per axis, split bars encoded in the script) — the script prints the
   ship verdict, not me.
5. P5 QC replays in `replays/m30_qc_*` for Piotr's review; ship only on his go.

## Rough sequence & cost

Housekeeping commit on feature/m29 → branch feature/m30 → P0+P1 (code+tests,
no games) → P2 trains (~1h) → P3 screens (~1–2h at n=60, workers 8) → P4
battery (~overnight, 4 arms, floors-first so dead arms skip the n=800 mirror)
→ P5 QC + STOP. Hermes notification at each transition and on any
kill/failure.
