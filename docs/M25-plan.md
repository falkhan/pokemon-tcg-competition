# M25 Plan — STUB (to be refined next session)

**Status: STUB** (2026-07-21 late, Piotr + session discussion). Written while M24's ship
(sub 54885173, replay-BC pilot + clone deck) accrues overnight. Review and refine before
executing anything beyond Phase 0.

## Framing (from the 07-21 discussion)

BC converges to ~teacher × fidelity — it reaches the frontier, never crosses it. The route
past the top decks is, in order of cost: exhaust the remaining BC headroom → PPO fine-tune
on top of the BC base (the M23 audit says PPO starved for signal, not that it is weak —
self-play BETWEEN strong BC pilots should finally feed it) → deck engineering as a
deck+pilot CO-ADAPTATION loop (feasible for the first time: a 0.6-class pilot can express
deck differences that M4's weak pilot could not; zero-shot transfer is proven dead, so the
pilot must be re-fitted with every deck change).

## Phase 0 — live validation (overnight, no compute)

- 54885173 settles from the μ=600 prior; A/B monitor (`notebooks/m22_ab_monitor.ipynb`,
  4 arms wired) is the only sanctioned reader — its power gate decides when the
  M24-vs-our-deck-arms comparison is real. NO directional reads before it clears MDE.
- Morning checklist: settled score, per-archetype forensics of 54885173 (harvest first),
  agent-error count, and whether the clone deck's live matchup profile matches the offline
  battery (0.605 lucario-field / 0.290 dragapult-line shape).

## Phase 1 — cheap BC headroom (days, not weeks)

1. **1268-cluster teacher**: build corpus from snowball-round-2 data
   (`--only-subs 54861775 54863653 54861685`, same-deck cluster `3121746f`, ~150 seats);
   train fresh V3-as-BC; identical battery. Candidate second arm / champion.
2. **Fidelity push on the shipped teacher**: more epochs, winner/score weighting
   (`bc.py --weighting`), hand-aware ids (untried on replay corpora) — each behind the
   0.660/0.290 pins, MDE discipline.
3. Periodic re-snowball — the ladder improves; our teacher pool should track it.

## Phase 2 — the S2-flip test, then PPO fine-tune (CONTINGENT on the flip)

- **Pre-registered discriminator**: collect a small self-play batch BETWEEN BC pilots
  (clone vs clone), re-run `scripts/m23_signal_audit.py s2`. Supporter↔win correlation
  turns materially POSITIVE → PPO has food → Phase 2 fine-tune from the BC base (clone
  opponents in the pool, S4 adv-by-type logging live, kill-gate = any regression vs the
  BC base on `rule:lucario`). Correlation still ~zero → PPO stays shelved; diary why.
- Endogeneity guardrail: clones may enter TRAINING or EVALUATION, never both sides of a
  claim; `rule:dragapult` + live accrual remain the only exogenous instruments.

## Phase 3 — deck+pilot co-adaptation loop (scope only if Phases 1–2 deliver a pilot)

- Local search (1–3 card swaps) around the proven lists (clone 60, 1268 list), evaluated
  by the strongest pilot at n clearing the ~7pp MDE; every accepted mutation re-fits the
  pilot (nearest-deck BC + short PPO leg). Matchup-aware tuning against the field
  distribution (parquet holds every deck + score) is the stretch goal.

## Added 07-21 ~20:30 (Piotr's correction)

The shipped clone deck is an **ALAKAZAM** deck (verified from card ids), not a lucario
list — the harvest archetype label for 54618168 was wrong. Phase 1 therefore adds:
**audit the archetype labeler against decklist card ids** for every corpus teacher
(3394cd30, f2b4039a, 475928aa lists unchecked), and read the 1268-cluster deck's actual
cards before any archetype claim. Morning forensics of 54885173 should read its
per-archetype rows knowing OUR side is Alakazam (mostly single-prize board — watch the
prize-trade pattern vs Mega decks). Grab an alakazam sprite for the monitor when
convenient (`notebooks/sprites/alakazam.png`).

## Open questions for the refinement session

- Ship cadence: second arm (1268 candidate) alongside 54885173, or wait for its live read?
- Phase 2 collect config: pure clone-mirror vs mixed clone pool; games budget.
- Deck-loop acceptance bar: how many games per mutation given slot/throughput limits?
- Whether the M24 commit should land first (still uncommitted — Piotr's call).

## Standing constraints (unchanged)

Workers ≤8 · dragapult never in a pool · one ship per milestone after the full battery ·
deck md5 ritual (`ship-deck-verify`) · MDE refusal discipline · diary incrementally.
