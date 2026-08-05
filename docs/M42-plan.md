# M42 — What the pilot cannot see: five perception defects, measured then fixed

## Why

M41 ended by proving one blind spot the hard way. `rl/combat.py::_charged_best`
drops any attack whose **printed** damage is `<= 0`, so Alakazam #743 `Powerful
Hand` (prints 0, really 20 x cards in hand) made `_turns_to_ready` return
UNREACHABLE at every energy count. Seven features read 0 across 703 live
prompts, 394 of them enumerated the null plan only, and the shipped clone
over-attached at **33.7%** against **4.7%** in its own BC corpus — a 7x
amplification of a defect the training data does not contain, because the net
cannot represent a distinction its inputs hold constant (`docs/M41.md:858-1005`).

The M41 fix was deliberately narrow: an appended option block at columns
100-104, `energy_is_dead` as a damage-free predicate, and `rl/scaling.py` as a
6-entry curated table. It landed as **capability only** — every live bundle
truncates columns 100-104 away, and nothing was wired into any pilot.

Piotr's QC replay review named four more defects of the same family, all things
a deck-agnostic pilot should understand from first principles:

1. don't attach more energy than the costliest attack **or the retreat** needs
2. know when an attack scales with **hand size**
3. know when it scales with **attached energy**
4. know a **stadium is in play**
5. with **0 bench and 0 basics in hand**, never take a Stage 1/2 off a search

This milestone answers, for each: is the information absent,
present-but-wrong, or present-and-ignored? Then it instruments the defect so QC
catches it, then it fixes what the measurement says is real. The order is not
negotiable — **four instruments in M41 read flat-damage semantics and gave a
confident wrong answer**, and three levers were killed by their own probes
before a gate cell was spent. Measure first.

## The review — Phase 0's deliverable

Not one of the five is a plain "missing feature".

| # | Defect | Neural pilot | Rule pilot | Forensics |
|---|---|---|---|---|
| 1 | over-attach vs cost/retreat | **encoded, invisible** — option cols 100-104 exist; every bundle slices `[:100]` (`rl/policy.py:207`) | **has a cap that cannot fire** — `score_attach` tier 3 (`rl/generic_pilot.py:432`) gates on `_turns_to_ready == 0`, which is 99 for any printed-0 attacker. Soft -150/surplus, floor 150, never a ban | OK — `[over-attach]` fixed in M41 to ask `energy_is_dead` (`rl/postmortem.py:428`) |
| 2 | hand-size scaling | **absent everywhere.** Only `has_variable_attack` (FEAT col 35, an undifferentiated OR over `"x" or "for each"`), and my hand size as a bare scalar at state 1625, v4 only. No option x state term. `_marginal_energy_damage` passes `hand_size=0` by construction | **misclassifies it** — tier 2 requires `_ATK[a][0] > 0`, so a printed-0 scaler scores 400 "non-attacker". `scaling` is behind an opt-in fix flag; `nominal_damage` substitutes a constant `NOMINAL_UNITS["hand"] = 8` against a measured mean hand of 17 | none |
| 3 | own-energy scaling | OK — cols 103/104 (M41), invisible for the same slice reason | **punishes correct play** — Ogerpon past its 3-cost trips the same surplus penalty. This is the exact false positive Piotr killed in the forensics, still live in the pilot | OK — scaler early-return (`rl/postmortem.py:432`) |
| 4 | stadium in play | OK — **already visible**: identity pooled at state 126-161, `stadiumPlayed` at 1621, M27 option col 95 `is_stadium AND none in play`. Not the gap | **blind** — `state.stadium` appears once, as a zone in `card_at`'s dict (`tcg/pilot.py:78`). No tier, no `stadiumPlayed` read | none |
| 5 | basis-less Stage 1/2 from a search | **nothing.** `apply_play_overrides` returns early on non-MAIN (`rl/plan.py:690`), so every `TO_HAND`/`LOOK` submenu is raw argmax. No feature computes "is my basis on board or in hand" | **guard exists but leaks** — `fetch_value` scores a dead evolution 60 (`tcg/pilot.py:136-157`), but only in `KEEP_CONTEXTS`; it is a soft 60, not a ban; and it counts the **active's** name in the basis pool, so "0 bench + 0 basics in hand" is not itself a trigger anywhere. The `DISCARD_CONTEXTS` mirror (`-card_usefulness`) is basis-blind, so a live basic can be pitched over a dead evolution | OK — `[fetch-dead-evolution]` (`rl/postmortem.py:504`), same KEEP-only scope |

Two corrections carried in rather than discovered later:

* **Stadium visibility is not the defect.** The neural pilot sees the stadium's
  full 36-column identity. What is missing is stadium **effect** in the damage
  model (Festival Grounds double-attack, Lively Stadium) and any stadium
  reasoning in the rule pilot. Ownership is not recoverable from the
  observation at all (`rl/determinize.py:56`).
* **Defect 5 is inert on the current ship.** `submission/deck.csv` is the
  Ogerpon list — 4x Teal Mask Ogerpon ex, **zero evolutions**. The rule only
  bites on the Alakazam / lucario / dragapult-class decks in `decks/`.

Recorded and **deliberately not fixed here**: `rl/encoders.py:207` and `:717`
resolve an option's `inPlayArea` target against `your_index` instead of
`opt.playerIndex`. Harmless for ATTACH/EVOLVE/tool (own board only), but any
opponent-board `inPlayArea` option would encode *our* Pokemon into columns
53-88. Fixing it moves columns inside every trained width — a silent live
policy change. Phase 1 measures whether it ever happens.

## Phases

**Phase 1 — Instrument before touching anything.**
`scripts/m42_perception_probe.py` (G-11), five staged ladders
`prompts -> situation_true -> option_offered -> chose_bad`, plus the M40
non-vacuity rule: `situation_true == 0` prints SITUATION NEVER OCCURS and exits
non-zero rather than reading as a pass. Extend `rl/postmortem.py` so the
mandatory QC battery screens for these automatically.

**Phase 2 — Make the damage model see scaling attacks (capability).**
`derive_scaling_table()` in `rl/scaling.py`, parsing the engine's own attack
text. The grammar is regular — *"this attack does N (more) damage for each X"*
yields mode, per-unit and base mechanically; `more` implies base = printed.
Acceptance test: the parse must reproduce all 6 curated entries exactly.

**Phase 3 — Append the M42 option block, 105 -> ~117.** Option side only; the
append-and-slice law holds there. A state append would need a warm-start
migration and a retrain.

**Phase 4 — Fix the rule pilot** (both twins, `tests/test_parity.py:230`). Its
bugs are correctness bugs, not policy bets: the neural ship does not contain
`generic_pilot.py`.

**Phase 5 — One neural arm, named and off.** `ATTACH_FIX_DEADENERGY`,
demote-only, MAIN-only, off unless named. A candidate for the next gate, not a
conclusion.

**Phase 6 — Record**, plus two housekeeping items the review surfaced.

## Hard constraints

- **No collect, no train, no ship.** Turning `scaling=True` on for
  `enumerate_plans`, the race block, the threat model and the solver teacher is
  the next milestone's bet — it changes what the TEACHER believes and cannot be
  evaluated without a fresh corpus (`docs/M41.md:1108-1116`).
- `--workers 8` maximum (12 deadlocks via `libcg.so` corruption).
- The cg engine keeps **one global mutable `Battle` per process**.
- Evolution completeness is by card **NAME, never by id**.
- Encoder changes are **append-only**; existing columns must re-encode
  bit-identically, proven against real options rather than argued.
- Every rule-pilot change lands in `rl/generic_pilot.py` and `tcg/pilot.py`
  identically or `tests/test_parity.py` fails.
- Diary every ladder and gate **as it lands, including the zeros**.
- No Hermes/Telegram updates — this box has no Hermes (Piotr, 2026-07-31).
