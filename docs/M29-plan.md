# M29 plan — scale the one lever that worked: winner demonstrations

## Context

M28 Track F is the campaign's first controlled, resolved positive: filtering
the corpus to the 203 games the teacher WON moved the mirror 0.6625 → **0.7256
(n=800, z=+2.22)** with the out-of-loop floor flat — from *half* the data.
Every process change tried this session (MCTS, PPO, value function, phase
conditioning) resolved negative or flat. The one axis with a track record,
M23→M26 replay-BC included, is **better demonstrations**.

The cache has unused headroom on exactly that axis:

| same-deck sub (hash `9294d9d8`) | score | cached eps | wins |
|---|---|---|---|
| 54618168 (our teacher) | 1251 | 373 | **203** (all used) |
| **54773249** | 1182 | 238 | **138** (never used) |
| 54662660 | 1156 | 54 | 20 |
| five smaller | 1067–1140 | ~100 | ~33 |

Pooled same-deck winners ≈ **1.9×** the m28_winners corpus, every game a win
with the byte-identical deck we ship. The M24 "single-teacher beats pooled" law
was measured on mixed win/loss corpora across different decks; same-deck
winners-only pooling is a new cell and gets its own A/B.

**Goal:** a shippable candidate with a defensible winrate gain, through the
mandatory pre-ship human QC, on Piotr's explicit go. Piotr approved both arms.

## Phases

- **P0 — refresh + harvest.** `kaggle_ingest refresh` with the same-deck subs as
  `--opp-subs` (+ our live 54903635 for the ongoing watch), then `harvest`.
- **P1 — corpora.** Arm T = refreshed 54618168 winners (control: m28_winners'
  recipe on more data, if any landed). Arm P = same-deck pooled winners
  (`--deck-hash 9294d9d8 --min-score 600 --winners-only --hand-aware`).
- **P2 — train** both from `m27_both` via `--init-v3m` + card-kind weighting —
  the exact m28_winners recipe, so corpus is the only variable.
- **P3 — battery in the SHIP config** (`modelt:` — O1 on, as live). Three
  candidates: m28_winners+O1, armT+O1, armP+O1. Pins = the live sub's battery
  (lucario 0.671 n=800 · dragapult 0.3375 n=400 · grim 0.6875 advisory ·
  kyogre ≥0.95). Note m28_winners' +6.3pp was measured PLAIN; the O1
  composition must be measured, not assumed (M26 composition lesson).
- **P4 — candidate selection.** Ship bar: mirror resolved-better than the live
  pin AND non-inferior on dragapult/kyogre. If no arm clears, the best
  non-inferior arm goes to Piotr with the evidence, not a recommendation.
- **P5 — pre-ship human QC (hard rule).** `build_submission.sh` export with
  `--deck clone54618168` + md5 ritual, then 3 games with the actual bundle vs
  the grim mill clone saved to `replays/` (`m29_qc_<arm>`), then **STOP** for
  Piotr's manual review. No submit without his explicit go.

## Standing

Workers ≤8 · diary incrementally in `docs/M29.md` · Hermes at phase
transitions · 0=WIN decode · MDE quoted on every claim · dragapult stays
evaluation-only · no goalpost moves after P3 numbers land.
