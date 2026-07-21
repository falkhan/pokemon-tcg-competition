# M24 Plan — replay-BC as the primary signal (the pipeline inversion)

## Context

The M23 signal audit (docs/m23-signal-audit-plan.md, verdicts in docs/M23.md 18:27) closed
every "fix our side" branch with pre-registered thresholds: architecture EXONERATED (E1:
fresh OptionScorerV3 BC-trained on the 1251-target's replays → fidelity 0.657 ≈ V2 clone
0.664, strength 0.5625 pooled n=400 vs `rule:lucario`, bar 0.54, both seeds clear), plan
head NOT decorative (S6: 0.90 attack / 0.68 attach consistency — do not delete), exploration
/ critic / entropy all healthy (S3/S5/S7). The one confirmed defect is the SIGNAL SOURCE:
S2 showed supporter↔win correlation in our own self-play is ~zero/negative — the RL loop
starves because its data contains no signal for the behaviors that win.

**M24 therefore inverts the pipeline: leaderboard-replay BC is the primary trainer;
self-play PPO is demoted to an optional, gated fine-tuner.** Piotr's decisions
(2026-07-21 evening): replay-BC on other teams' public replays is an APPROVED ship path;
deck question resolved as **both-decks A/B**; corpus **expand now**; **no early clone
ship** — one ship per milestone, after gates.

## Phase 0 — corpus expansion (launch immediately)

1. `kaggle_ingest snowball` the top ~10 subs by leaderboard score (current `targets` head:
   54618168 1251.4, 54827443 1191.3, 54834745 1190.0, 54829486 1180.2, 54837973 1174.1,
   54773249, 54826859, 54840044, 54703548, 54833827 — all ≥1140), then harvest. NOTE the
   board moved since morning — several new ≥1140 subs exist; coverage per sub decides the
   trainable set, not the wishlist.
2. Rebuild corpora with `replay_bc build`:
   - per-teacher corpus for every sub that lands ≥~150 episodes with obs (M10 law: single
     consistent policies clone better than pooled unobservable mixtures — pooled corpus is
     the fallback, score-weighted, not the default);
   - the 54618168 corpus refreshed (was 348 seats);
   - **our-deck corpus**: filter opponent seats whose deck_hash == `20dcd313…` (our
     lucario.csv is the 0.899-weight meta deck — strong pilots play OUR exact 60 on-ladder).
     `build` filters by `--only-subs` today; add a `--deck-hash` predicate next to it
     (metadata already in episodes.parquet; small change + test beside the only-subs test).
3. Record per-corpus stats in docs/M24.md as produced (episodes, decisions, deck hashes).

## Phase 1 — the A/B candidates

- **C-clone (clone-deck arm):** V3-as-BC on the biggest single-teacher corpus
  (54618168 refreshed; consider `--epochs 10`, the E1 recipe — `rl.plan_iter train`, fresh,
  width-driven). [Corrected 07-21: this deck is an ALAKAZAM list, not a lucario mirror —
  see docs/M24.md 20:30 entry; the harvest archetype label was wrong.] Deck = `data/kaggle/clone_54618168_deck.csv` (hash `9294d9d8…`; pin its
  md5 in the ship ritual before any ship talk).
- **C-ours (our-deck arm):** V3-as-BC trained/fine-tuned toward piloting `decks/lucario.csv`
  (`20dcd313…`, md5 `aef8da62…`): primary source = the our-deck-hash corpus (strong pilots
  on our exact 60); fallback if volume is thin = the mirror-archetype pooled corpus, gated
  zero-shot on our deck (the M18.1 lesson says zero-shot deck transfer is real but must be
  measured, never assumed).
- **Gate battery, identical for both** (measure-agent skill; 0=WIN decode; 2-seed):
  `rule:dragapult` n=400×2 PRIMARY (baselines: M22c-RL 0.1888, B3 0.172, clone-line
  0.20–0.25); `rule:lucario` n=200×2 (E1 pin 0.5625); `random:kyogre` n=200×2 comparative
  floor; mirror `solver:lucario` informational only; `--latency` before any ship talk.
  MDE discipline: no sub-MDE claims; dragapult stays OUT of every training pool (hard
  guard in parse_pool).
- **Ship**: Piotr's explicit call only, after gates; `build_submission.sh --deck <deck>`
  + md5 verification per `verify-before-consequential-actions.md` + ship-deck-verify memory.
  Both-arm ship (A/B) is the pre-approved shape if both gate well.

## Phase 2 — CONTINGENT: PPO fine-tune on top of the BC base

Only if Phase 1 gates well and Piotr wants the extra leg. Recipe deltas: start from the
Phase 1 winner, opponent pool anchored on the CLONES (`model:bc_clone_*` specs) + sample
agents; S4 advantage-by-type logging (already live in rl/ppo.py) watches whether credit
reaches card-economy actions; kill-gate = any regression vs the BC base on
`rule:lucario` beyond MDE (the M21 Gate-A lesson runs both directions).

## Verification

- Suite green before/after every code change (`uv run pytest tests/ -q`, 607 at scoping).
- Corpus builds validated by the existing `replay_bc` align/audit gates (≥0.98 agreement).
- Gate results diaried in docs/M24.md at the moment produced; hermes updates at stage
  transitions; heartbeat on runs >30 min; workers ≤8 hard cap.

## Out of scope / do-not-do

- No dragapult in any training pool (evaluation-only, hard-enforced).
- No plan-head rewrite/deletion (S6 exonerated it); no new plan vocabulary this milestone.
- No self-play PPO as primary signal; no kaggle_ingest meta runs.
- No ship without explicit instruction + deck md5 ritual (both deck hashes pinned above).
