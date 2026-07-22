# M25 Plan — get PAST the frontier BC converges to

**Status: REFINED** (2026-07-21 ~22:30 session, Piotr + first live read of 54885173).
Supersedes the stub written at ship time. Antecedents: `docs/M24.md` (ship diary),
`docs/M23.md` (signal-audit verdict). Branch: `feature/m25` (M22/M23/M24 merged to `main`
as d2b8743 this session — the "commit first" open question is resolved).

## Framing (unchanged from the 07-21 discussion)

BC converges to ~teacher × fidelity — it reaches the frontier, never crosses it. The route
past the top decks, in order of cost: exhaust the remaining BC headroom → PPO fine-tune on
top of the BC base (the M23 audit says PPO starved for signal, not that it is weak —
self-play BETWEEN strong BC pilots should finally feed it) → deck engineering as a
deck+pilot CO-ADAPTATION loop (zero-shot transfer is proven dead, so the pilot must be
re-fitted with every deck change).

## Refinement-session findings (2026-07-21 ~22:30)

1. **First live read of 54885173** (49 games to 22:17Z, ~2h15m post-ship): 25W–24L,
   score 600 → **723.0** — ~145 points above all four our-deck arms (531–579). Early and
   unsettled; the A/B power gate (`notebooks/m22_ab_monitor.ipynb`, TARGET_N=200,
   MDE 10pp) remains the ONLY sanctioned comparator. One loss was a self-match vs our own
   arm. The notebook has 54885173 wired but is unexecuted — first morning action is
   re-running it to regenerate `docs/M22-live-state.md`.
   **Matchup profile by TRUE archetype** (card-id grouping, labeler bypassed; all n's
   tiny — directional hypotheses only): Mega Lucario field 8W–6L (the 0.605 offline
   shape holds); Dragapult 4W–1L (live pilots, softer than `rule:dragapult`); Alakazam
   mirror+variants 5W–5L; **Marnie's Grimmsnarl control 1W–4L and Team Rocket 1W–3L —
   the two bleeds**. The Grimmsnarl bleed is the same archetype as the 1268 cluster —
   Phase 1's teacher candidate is also counter-training for our worst live matchup.
2. 🔴 **The archetype labeler is systemically broken, and the M24 Alakazam mislabel was
   not an isolated case.** Root cause read from code: `_assign_archetypes`
   (`rl/kaggle_ingest.py:645`) greedily merges any deck whose pooled Pokémon
   stat-feature vector has cosine ≥ `ARCHETYPE_COS = 0.95` (eyeball-tuned) with an
   existing cluster centroid, most-played first — distinct decks collapse into the
   dominant cluster and inherit its name. Confirmed casualty: the **1268-score cluster
   deck `3121746f`** (subs 54861775/54863653/54861685, 689 opp_decks rows — more than
   the ~400 estimated) is labeled `mega_lucario_ex+solrock` but its card ids are
   **3× Marnie's Grimmsnarl ex + Froslass/Snorunt mill + 4× Munkidori — a Darkness
   control/mill deck**. Consequences:
   - The Phase-1 "1268-cluster teacher" is a CONTROL archetype, not a lucario variant —
     as a second arm it buys archetype diversity, and its playstyle (mill/stall) is the
     style our pilots have never expressed.
   - Every historical per-archetype claim is suspect: "the field is 76% mirror", "all
     four arms bleed vs mega_lucario_ex+solrock" (that bucket contains at least
     Grimmsnarl-mill and Alakazam) — the field composition must be re-read after the fix.
   - The M24-ship follow-up (audit labels for `3394cd30`, `f2b4039a`, `475928aa`) is
     hereby widened to: fix the labeler, relabel, re-read the field.

## Phase 0 — live validation (overnight, no compute)

- 54885173 settles from the μ=600 prior; A/B monitor power gate decides when the
  M24-vs-our-deck-arms comparison is real. NO directional reads before it clears MDE.
- Morning checklist: re-run `m22_ab_monitor.ipynb` (regenerates `M22-live-state.md` with
  the M24 arm), settled score, `kaggle_ingest refresh --subs 54885173` +
  `forensics --sub 54885173` (per-archetype rows only meaningful AFTER the labeler fix),
  agent-error count, and whether the live matchup profile matches the offline battery
  shape (0.605 lucario-field / 0.290 dragapult-line).

## Phase 1 — instrument fix + cheap BC headroom (days, not weeks)

0. **Labeler fix FIRST** (new, blocking all archetype reads): label each deck_hash from
   its OWN top-2 Pokémon (`_archetype_name` already does this correctly — the defect is
   only the cosine cluster-join), or replace the stat-feature cosine with card-identity
   overlap (e.g. Jaccard over Pokémon ids). Re-harvest, verify the known decks land
   right (clone→alakazam, `3121746f`→grimmsnarl, our 60→lucario), then re-read field
   composition + re-run forensics for all live arms. Small, testable, high leverage.
1. **1268-cluster (Grimmsnarl-mill) teacher**: corpus from snowball-2 data
   (`--only-subs 54861775 54863653 54861685`, deck `3121746f`, 689 rows); train fresh
   V3-as-BC; identical battery. Candidate second arm / champion. Note: single-teacher
   beat pooled in M24 — prefer the highest-score sub's seats if volume allows.
2. **Fidelity push on the shipped teacher**: more epochs, winner/score weighting
   (`bc.py --weighting`), hand-aware ids (untried on replay corpora) — each behind the
   0.660/0.290 pins, MDE discipline.
3. Periodic re-snowball — the ladder improves; our teacher pool should track it.

## Phase 2 — the S2-flip test, then PPO fine-tune (CONTINGENT on the flip)

- **Pre-registered discriminator**: collect a small self-play batch BETWEEN BC pilots
  (clone vs clone), re-run `scripts/m23_signal_audit.py s2`. Supporter↔win correlation
  turns materially POSITIVE → PPO has food → fine-tune from the BC base (clone opponents
  in the pool, S4 adv-by-type logging live, kill-gate = any regression vs the BC base on
  `rule:lucario`). Correlation still ~zero → PPO stays shelved; diary why.
- Endogeneity guardrail: clones may enter TRAINING or EVALUATION, never both sides of a
  claim; `rule:dragapult` + live accrual remain the only exogenous instruments.
- With a Grimmsnarl-mill BC pilot from Phase 1, the self-play pool can be
  cross-archetype (alakazam-BC vs grimmsnarl-BC) — closer to the real ladder mix than a
  pure clone mirror.

## Phase 3 — deck+pilot co-adaptation loop (scope only if Phases 1–2 deliver a pilot)

- Local search (1–3 card swaps) around the proven lists (clone 60, `3121746f` 60),
  evaluated by the strongest pilot at n clearing the ~7pp MDE; every accepted mutation
  re-fits the pilot (nearest-deck BC + short PPO leg). Matchup-aware tuning against the
  TRUE field distribution (post-labeler-fix) is the stretch goal.

## Open questions for Piotr

- Ship cadence: second arm (1268/Grimmsnarl candidate) alongside 54885173, or wait for
  its live read? (Slots appear free; one-ship-per-milestone rule stands.)
- Phase 2 collect config: pure clone-mirror vs mixed cross-archetype pool; games budget.
- Deck-loop acceptance bar: games per mutation given slot/throughput limits.
- ~~M24 commit lands first?~~ DONE — merged to `main` (d2b8743), branch `feature/m25`.

## Standing constraints (unchanged)

Workers ≤8 · dragapult never in a pool · one ship per milestone after the full battery ·
deck md5 ritual (`ship-deck-verify`) · MDE refusal discipline · diary incrementally.
