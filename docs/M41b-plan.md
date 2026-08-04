# M41b — the deck-agnostic encoding gaps: what the net cannot see

> Naming note: this lands chronologically **after** M42, whose diary holds the
> review that motivates it (`docs/M42.md` § "Encoding review"). Kept as M41b at
> Piotr's instruction; if the MILESTONES ordering matters more than the thread,
> rename to M43 before the first commit.

## Why

M42 asked what a deck-agnostic pilot still cannot see. Three answers were
measured, not judged:

1. **879 real aliased option pairs across 407 of 5,319 menus (7.7%)** —
   `scripts/m41b_alias_probe.py`-style measurement (currently
   `scripts/m42_alias_probe.py`). Every one has the same cause: an option
   carries its subject's **card identity** but never its **live state** or
   **which board slot** it is. CARD 415, ABILITY 192, ATTACH 169, ENERGY 57,
   EVOLVE 45, TOOL_CARD 1. 9,807 further aliased pairs are genuinely
   interchangeable duplicates and are correctly ignored.
2. **1,069 of 1,556 attacks (69%) have a TYPED cost and the encoder only ever
   sees its size.** At cost size 3 the pool holds 52 distinct typed cost tuples,
   all encoded as `0.600`. Affordability is a type question and it is the most
   deck-transferable fact in the game.
3. **`evolves_from_id` is a raw card id summed into a pool.** On
   `decks/alakazam_v2_h4.csv` the v2 deck pool reads **649.2** in that column
   against **1.9** for the largest flag — a **342x** imbalance carrying no
   ordinal meaning, since card 1151 is not "more" than card 5.

Plus two absences with coverage but no measured misplay rate: **abilities are
entirely invisible** (218 of 1,056 Pokemon have one; FEAT has no ability column,
so they are learnable only through the per-card-id embedding — exactly the
deck-specific memorisation to avoid), and **weakness/resistance match** exists
only implicitly inside damage ratios.

## Two corrections to the M42 recommendation table

The M42 diary's closing table marked two items "append-safe". **Both are wrong**,
and the plan below is shaped around the correction:

* **A FEAT column append is NOT safe.** `FEAT_DIM` sits *mid-vector* in both
  `SLOT_DIM = FEAT_DIM + 3 + N_ENERGY + FEAT_DIM` and
  `OPTION_DIM = N_OPTION_TYPES + FEAT_DIM + FEAT_DIM + 1`
  (`rl/encoders.py:80,91`). One new FEAT column shifts every column after the
  first FEAT block in every slot and every option. So the ability flag cannot
  go in FEAT as an addition.
* **Retreat cost cannot go into `_retreat_extra`.** That writes option columns
  90-92, *inside* every trained width. It must go in the appended block.

Also confirmed: `OptionScorerV3.forward` truncates **only `options`**
(`rl/policy.py:207-208`), never `state_ctx`. **State appends are retrain-gated**,
so everything in Phase 1 goes on the option side.

## Phase 1 — the M41b option block, 115 -> 144 (append-only, capability)

29 slots appended after the M42 block. Slot indices below are relative to
`OPTION_M42_DIM = 115`.

### 1a. Board-object live state — slots 0-6 (the measured fix)

The 879 aliased pairs, all of them. The subject is on the board in *different
fields* depending on option type, measured over real replays:

| option type | where the board object lives |
|---|---|
| ATTACH, EVOLVE, TOOL_CARD | `inPlayArea` / `inPlayIndex` |
| ABILITY (`area=BENCH` 578, `STADIUM` 232, `ACTIVE` 134) | `area` / `index` |
| CARD (`area=BENCH` 1377, `ACTIVE` 113) | `area` / `index` |
| ENERGY (`area=BENCH` 77) | `area` / `index` |

New helper, generalising the existing `_my_poke_at`:

```python
def _option_board_object(opt, obs):
    """(pokemon, is_active, bench_index) for whichever field points at OUR
    board, or (None, False, 0). inPlayArea wins when both are set."""
```

Slots: `has_object`, `hp / maxHp`, `min(maxHp - hp, 100) / 100`,
`len(energies) / 5`, `len(tools) / 2`, `is_active`, `bench_index / 4`.

`bench_index` is not needed to break the aliasing — the live state does that —
but it lets the net associate the option with the correct per-slot state block
(`173 + slot*87`), which is the join it currently cannot make.

### 1b. Typed attack cost — slots 7-20

`cost_by_type[12] / 3` (slots 7-18), `colorless_deficit / 5` (19),
`can_afford_now` (20). Read from `_ATK[aid][1]`, which is already a tuple of
`EnergyType` ints, so this is bundle-safe and needs no parquet.

The deficit is **not** a per-type subtraction — colorless slots are payable by
any energy, so it is a small matching problem. Compute it exactly: satisfy typed
slots first from same-type energy, then pay colorless slots from the surplus,
and report what is still missing. `_can_afford` stays the boolean authority for
slot 20 so the two can never disagree.

**Honest scoping:** unlike 1a this has no measured misplay rate behind it. The
evidence is coverage (69% typed, 52 tuples collapsing to one value) and it is a
generalisation argument, not an aliasing one — within-menu attack aliasing
measured **0** (only 54 menus offer 2+ attacks). Phase 3's census is what
decides whether the slots carry information.

### 1c. Type matchup — slots 21-24

`opp_weak_to_my_type`, `opp_resists_my_type`, `my_active_weak_to_opp_type`,
`my_active_resists_opp_type`, from `_CARD` weakness/resistance vs the attacker's
`energyType`.

These depend only on the two actives, so they are identical for every option in
a menu — genuinely a *state* feature living in the option block. That redundancy
is the price of slice safety, and it is 4 columns; the alternative is a
retrain-gated state append. Recorded so nobody later reads it as an oversight.

### 1d. Retreat + hand economics — slots 25-28

`active_retreat_cost / 4`, `retreat_payable`, `my_hand_count / 15`,
`my_bench_count / 5`.

Retreat cost is in FEAT and in `rl/combat._RETREAT` and is read by **no**
decision feature today. Hand size exists only as v4 state index 1625, so no v3
net can see it at all; bench count has no scalar anywhere.

### Files

| file | change |
|---|---|
`rl/encoders.py` | `N_OPTION_BOARD/COST/MATCHUP/ECON`, `OPTION_M41B_DIM = 144`; `_option_board_object`, `_percept_board`, `_percept_cost`, `_percept_matchup`, `_percept_econ`; write site in `encode_option_v2`; `np.zeros(OPTION_M41B_DIM)` |
`tcg/encoders.py` | mirror all of it (no-underscore names, that file's convention) — `tests/test_parity.py` compares the two byte-for-byte |
`rl/policy.py` | **no change** — the truncation shim is already generic |
`tests/test_encoders.py` | extend the width chain; per-block behaviour tests under the existing `for mod in (old, new)` pattern; zero-on-inapplicable-option-type |

The write site must dispatch per option type, and every slot must stay 0 on
types it does not describe — the M42 block has a test for exactly that and it
should be extended rather than copied.

## Phase 2 — FEAT hygiene, same width, retrain-paired (default OFF)

`evolves_from_id` is 342x the largest flag and semantically meaningless as a
magnitude. But it **cannot simply be dropped**: `rl/deck_build.py:116`,
`scripts/leaderboard_decks.py:272` and `tcg/cardpool.py:255` all read it from
the parquet, and dropping it from FEAT would change `FEAT_DIM` and shift every
vector.

The way through keeps the parquet untouched and `FEAT_DIM` at 36: **transform
FEAT after load**, by column NAME, in both encoder twins.

```python
FEAT_V2 = False   # M41b: OFF until a net is trained against it
# when True, after FEAT is built:
#   evolves_from_id  ->  has_ability      (0/1, from cg.api skills)
#   hp               ->  hp / 380
#   max_damage       ->  max_damage / 350
#   retreat_cost     ->  retreat_cost / 4
#   min_attack_cost  ->  min_attack_cost / 5
```

Swapping `evolves_from_id` **for** `has_ability` is the whole trick: one column,
no width change, a 342x noise source replaced by the 21%-coverage signal that
is currently absent — and because it lands in FEAT it becomes visible per-slot
across all 12 board slots and in every pooled zone, which an option-block flag
could never do.

**This changes VALUES inside trained widths, so it is a distribution shift on a
live bundle** — precisely what M41 refused to do. Hence `FEAT_V2 = False` by
default: `scripts/m42_column_safety.py` at width 100 must stay
`9b4603c33bd7e86676e321460b3812a6` with the flag off, and the flag is flipped
only for the collect + train. The bundle path (`card_features.npy`, written by
`tcg.shipping.export`) inherits whatever FEAT is live at export time, so the
export must refuse to run with `FEAT_V2 = True` unless the checkpoint was
trained under it — add that as a `ship_verify` Tier-1 check.

Under `tests/fake_cg` there are no `skills`, so `has_ability` reads all-zero;
pin that so the column is never assumed populated in tests.

## Phase 3 — verification, pre-registered

1. **The aliasing kill.** `scripts/m42_alias_probe.py` keyed on the NEW width
   must take `real::*` from **879 to 0**, with
   `harmless_true_duplicates` unchanged at ~9,807. Any residual real pair is a
   slot that was specified wrong. This is the only pass/fail bar in the
   milestone that needs no training.
2. **Feature census — new instrument.** `scripts/m41b_feature_census.py`: for
   every new slot, the non-zero rate and the value spread over real options.
   **Pre-registered kill: any slot that is non-zero on <1% of the options its
   type applies to, or has zero variance, is DROPPED before the retrain.** Dead
   columns are not free — they are capacity and noise, and this is the check
   that stops 1b being carried on a coverage argument alone.
3. **Column safety.** `scripts/m42_column_safety.py --width 100` and
   `--width 115` must both PASS against the pre-change digests, with
   `FEAT_V2 = False`.
4. **Twin parity + suite.** `tests/test_parity.py` byte-identical;
   1062 passed + the same 4 known failures.
5. **Watchable sanity.** `scripts/watch_games.py` on one rule arm and one model
   arm, confirming no crash and no latency regression from the per-option cost
   matching (it runs on every ATTACK option — measure it, do not assume; the
   M41 precedent is `rl.scaling`'s 6.3 ms import).

## Hard constraints

- **Append-only in Phase 1.** Existing columns must re-encode bit-identically,
  proven against real options rather than argued.
- **Phase 2 is default-OFF and retrain-paired.** No export, no ship, while
  `FEAT_V2` diverges from the shipped net's training.
- Every encoder change lands in `rl/encoders.py` **and** `tcg/encoders.py`
  identically or `tests/test_parity.py` fails.
- Evolution relations are by **NAME, never id**.
- Bundle-safe: no polars, no torch, no parquet in anything the submission
  imports. `_ATK` and `cg.api` are the only card-data sources Phase 1 may use.
- `--workers 8` maximum on any run.
- Diary each measurement as it lands, including the zeros.
- No Hermes/Telegram on this box.

## Out of scope

- **No collect, no train, no ship.** Phase 1 is capability; the shipped net
  reads columns 0-99 and will not see any of it. The width-144 retrain is the
  next decision, and it now carries the `FEAT_V2` question with it.
- The `rl/encoders.py:207` / `:717` `inPlayArea`-vs-`playerIndex` target
  misresolution stays unfixed — still latent (0 foreign-board options in 4,388
  prompts), still a column move. **But note it interacts with 1a**: if an
  option ever addresses the opponent's board, `_option_board_object` must not
  silently read our slot at that index. Phase 1 resolves ownership explicitly
  and returns `None` for a foreign board rather than inheriting the bug.
- No attack-identity embedding. Within-menu attack aliasing measured 0, so
  there is nothing to fix; typed cost is the generalisation answer.
- `tests/fake_cg.CardType` still does not match the engine's numbering.
