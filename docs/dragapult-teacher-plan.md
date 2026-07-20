# Plan: Dragapult ex teacher + deck integration

**Date:** 2026-07-19 · **Status:** done (pre-milestone infrastructure; not shipped)

## Context
The Pokemon company published a rule-based "Dragapult ex" sample agent
(`a-sample-rule-based-agent-dragapult-ex-deck.ipynb`, repo root). We incorporate it
exactly like the Iono extraction in M2: a third archetype package (rule brain + deck)
usable as teacher, opponent, and league anchor.

## Steps
1. Extract notebook cell 3 (`%%writefile main.py`) verbatim → `sample-agent-dragapult/main.py`.
   The agent already matches the teacher interface: module-level `my_deck`,
   `agent(obs_dict) -> list[int]`, mutable module globals (handled by the isolated
   per-instance loader in `rl/teacher.py` / `tcg/teachers.py`).
2. Build the 60-card deck CSV from the notebook's decklist constants →
   `sample-agent-dragapult/deck.csv` (read at exec time, cwd = agent dir) and
   `decks/dragapult.csv` (the loader's `my_deck` override source).
3. Register `"dragapult"` in `TEACHER_PATHS`/`DECK_PATHS` in **both** `rl/teacher.py`
   and `tcg/teachers.py` (parity pinned by `tests/test_decks_teachers.py`).
4. Add `("dragapult_expert", ("rule", "dragapult", "dragapult"))` to `rl/league.py`
   `ANCHORS` (unordered in `check_anchor_ordering`, like the Iono anchors).
5. Tests: `"dragapult"` in the `load_deck` loop (`test_decks_teachers.py`) and a
   `rule:dragapult` case in `test_parse_spec_kinds` (`test_matchrunner.py`).
6. Deliberately NOT touched: `rl/collector.py` opponent pool and plan_iter defaults —
   training-distribution changes are a milestone decision (see DECISIONS 2026-07-19).

## Verification
- `validate_deck(load_deck("dragapult"))` → legal.
- Full `uv run pytest` suite green after changes.
- Engine smoke + strength read (slot-fair `play_series`, n=200 × seeds {1,2}, 0 errors):
  vs `rule:iono` 0.657 pooled · vs `rule:lucario` 0.495 · vs `rule:tuned` 0.477.
  Dragapult ≈ Lucario-expert strength.

## Gotcha (recorded for future loader debugging)
`load_teacher` chdirs into the agent dir; the teacher's `from cg.api import ...` only
resolves because `cg` is already in `sys.modules` (any `-m`/script entrypoint imports it
from the repo root first). Under `python -c` (where `sys.path[0] = ''` resolves against
the *current* cwd), pre-import `cg.api` before calling `load_teacher`.
