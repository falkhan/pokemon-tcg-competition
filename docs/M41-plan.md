# M41 — Deck Lab, and the deck question answered by experiment

## Why

The 2026-08-03 leaderboard census (`notebooks/leaderboard_decks.ipynb`,
`scripts/leaderboard_decks.py`) measured what the top of the ladder plays for the
first time. Top 250, 222 labeled (89%):

| family | top-250 | our live mix | our live record |
|---|---|---|---|
| **grim** | **49.5%** | 9.7% | **2-13 (13.3%)** |
| mirror | 18.0% | 19.4% | 20-13 (60.6%) |
| wall | 9.5% | 11.0% | 8-9 (47.1%) |
| lucario | 1.8% | 11.0% | 12-5 (70.6%) |
| starmie | 0.5% | 7.1% | 8-3 (72.7%) |
| **archaludon** | **0.0%** | **13.6%** | 14-7 (66.7%) |

We beat almost exactly the decks that are not up there and lose to the ones that are.
`archaludon` carries 13.6% of our bed weight and three dedicated beds (`arch_d1/d2/d3`)
and appears zero times in the top 250; `iono` likewise.

The typing is structural, not incidental (`data/cards_features.parquet`): the whole
Marnie's Grimmsnarl line (Darkness, 320 HP on the ex) is **weak to Grass**, and our
Alakazam (Psychic, 140 HP) is **weak to Darkness**. Excluding grim, **42.0% of the top
field is Grass** — the field has already found the answer.

`docs/BACKLOG.md` § "The deck question, RE-OPENED by the leaderboard census" holds the
three options (switch to grim / engineer an anti-grim deck / keep teaching ours) and the
blocker: **every measurement to date confounds deck strength with pilot strength.** The
one prior verdict, M40b Track A's 0.436 kill, measured *our net piloting a grim clone* —
a pilot result — while the same note records the *deck* measuring stronger (0.70 H2H,
worst matchup 0.45).

M41 builds the tooling to separate the two, then runs the experiment.

## Two corrections carried in from planning

1. **The sample agents are not deck-agnostic.** `sample-agent*/main.py` are hard-coded
   card-id cascades (`Mega_Lucario_ex = 678`, 15–45 id comparisons each);
   `tcg/teachers.py`'s docstring records the failure mode. The deck-agnostic pilot is
   `rl/generic_pilot.py` (`generic:` / `solver:`), which scores off the global card DB.
2. **`matchrunner` produces no watchable replays** — it drives `cg.game` directly, so
   there is no `visualize` payload. Phase 3 bridges to the `kaggle_environments` path.

Both are fixed in `docs/BACKLOG.md`.

## Phases

**Phase 1 — Deck Lab.** `tcg/cardpool.py` (card pool, attack join, facets, filters,
lazy `cg` text) + `tcg/decklab.py` (deck counts, summary, evolution/energy notes,
save/load, smoke runner) + `app/deck_lab.py` (Streamlit UI, zero logic).
Launch: `uv run --with streamlit streamlit run app/deck_lab.py`.
Deck-building with live legality, saving to `decks/custom/`, and a subprocess smoke test.

**Phase 2 — The fixed-pilot deck probe.** `scripts/deck_probe.py`: candidate × opponent
matrix, both sides piloted by `generic:`, opponents weighted by **top-250 share** rather
than our live mix. Same pilot both sides ⇒ the delta is the deck.

**Phase 3 — Replay capture.** `scripts/watch_games.py` bridges any matchrunner spec into
`rl/eval.py::play_games` (which accepts callables; `make_pilot` returns one), yielding
`replays/*.html` for human review **and** per-game JSON that `rl/postmortem.py --batch`
reads.

**Phase 4 — Loss analysis → rule iteration.** Local-game instruments only
(`postmortem --batch` 7-flag taxonomy, `offline_behavior.py`, `matchrunner --diag`) —
the `live_*` / `m40_*` forensics scripts read Kaggle cached replays and do not apply.
Rules follow the established pattern: `PLAY_FIX_*` in `rl/plan.py`, a single-variable
`_MODEL_FIX_KINDS` entry in `rl/matchrunner.py`, a **mandatory G-11 mechanism probe**,
then the weighted panel gate.

## Hard constraints

- `--workers 8` maximum (12 deadlocks via `libcg.so` corruption).
- The cg engine keeps **one global mutable `Battle` per process** — the app runs games
  in a subprocess, never in-process. `cg.api.all_card_data()`/`all_attack()` are exempt
  (pure table reads, no `battle_start`).
- Evolution completeness is by **card NAME, never by id** — `evolves_from_id` points at
  one printing, and `decks/lucario.csv` runs the off-printing Riolu.
- `matchrunner --workers 8` is not run-reproducible (G-12): n≥800 resolves ~10pp,
  n≥2400 ~5pp. Pre-register the read.
- No pyarrow in the project venv — polars only in the logic layer.
- Ship nothing without an explicit instruction; deck choice is Piotr's call.
