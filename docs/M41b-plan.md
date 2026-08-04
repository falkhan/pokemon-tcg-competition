# M41b — encoding, learning architecture, and harness

**Part I** is the encoder work M42's review measured. **Part II** is the
pilot-learning architecture and the harness, sequenced against it.

Line-level claims were re-verified against the codebase, the engine API, and
the diaries on 2026-08-04; the corrections and decisions from that review are
folded in below, dated "(review, 2026-08-04)".

## Execution order — decided with Piotr, 2026-08-04

Both forks in § II.5 were put to Piotr and both resolved to the recommendation:
**Stage A then B**, and **harness before Stage B**. The resulting order is the
one thing in this document that is settled rather than proposed:

| # | phase | why here | ends when |
|---|---|---|---|
| 0 | **§ II.1** — epoch amendment to `ARCHITECTURE.md` §15 | pure bookkeeping, but §15 currently forbids the family Stage B belongs to. Do it before anyone reads the list and stops. | the three pre-epoch entries and M8.4 are marked |
| 1 | **Part I** — the encoder (Phases 1-3) | Stage A. Prerequisite: a search teacher cannot be distilled into a student whose inputs alias on 879 real option pairs. | aliasing probe reads `real::* == 0`; column safety PASSes at widths 100 and 115 |
| 2 | **§ II.3a/b** — golden fixtures + cross-instrument agreement | Stage B is ONE gate cell that decides a month of work. Measuring it with instruments that took six corrections in their last milestone is how you buy a confident wrong answer. | every probe script (the 19 `*_probe.py` PLUS `pm_probe2.py`, which the glob misses) has a fires-and-guards fixture in the suite |
| 3 | **§ II.3c** — the regression-adjusted gate estimator | Stage B's plausible effect (+25-60 ELO ≈ 3.5-8.5pp, § II.2) sits at or below the ~14pp a 400-game cell resolves — without variance reduction the gate is underpowered and its null is noise. Decided by re-analysing a finished battery both ways | a win-rate CI half-width comparison, raw vs adjusted; if adjustment fails, Stage B's n rises to ~2-3k |
| 4 | **§ II.3d/e** — gates as a hashed spec, ship guards into CI | cheap, deterministic, and they close the exact holes M41 fell through | `ship_verify` Tier-1 invariants run as tests |
| 5 | **Stage B** — BACKLOG #10, distil the search | the central assumption: is search output learnable by our net at our scale? | one gate cell against the panel, at the n § II.3c's outcome dictates |
| 6 | **Stage C** — the `search_begin/step/end` re-descent probe | half a day, and it moves a 10-50x constant on everything downstream. Cheap enough to run whenever; must precede any re-estimate of #9. | a yes/no on re-descent from an arbitrary node |

**Stage D is explicitly OUT OF SCOPE for M41b.** It is gated on Stage B's
number, and BACKLOG 9's own note governs: *"consumes a whole campaign; do not
start one inside a deadline."*

The load-bearing property of this order is that a **null result at Stage B is
interpretable**. Run before Part I, a null cannot distinguish "search output is
not learnable at our scale" from "the student could not see what the teacher was
reacting to". Run after, it can — and that distinction is the difference between
correctly abandoning a month of work and wrongly abandoning it.

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

## Phase 1 — the M41b option block, 115 -> 145 (append-only, capability)

30 slots appended after the M42 block. Slot indices below are relative to
`OPTION_M42_DIM = 115`.

> **Amendment (Piotr, 2026-08-04, during execution): 144 -> 145, foreign
> boards are encoded.** The first spec read only OUR board and returned None
> for a foreign `playerIndex`, on the recorded premise "0 foreign-board
> options in 4,388 prompts". The kill probe falsified the premise: at the
> ours-only width 144 the residual was **259 real pairs, ALL foreign** —
> DAMAGE (91 groups) and DAMAGE_COUNTER (53) spread-target submenus and
> gust-SWITCH (21) address the OPPONENT's Pokemon via `area`+`playerIndex`
> (the earlier "0 foreign" count had only examined the `inPlay*` fields).
> Piotr's call: resolve the subject on whichever board `playerIndex` names
> and add `is_opponents` as board slot 7. Blind damage-target menus were the
> single most decision-relevant residual — dragapult's bread and butter.

### 1a. Board-object live state — slots 0-7 (the measured fix)

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
    """(pokemon, is_active, bench_index, is_opponents) for whichever board
    Pokemon the option addresses, or (None, False, 0, False). inPlayArea wins
    when both are set; ownership follows playerIndex explicitly."""
```

Slots: `has_object`, `hp / maxHp`, `min(maxHp - hp, 100) / 100`,
`len(energies) / 5`, `len(tools) / 2`, `is_active`, `bench_index / 4`,
`is_opponents` (the amendment above).

`bench_index` is not needed to break the aliasing — the live state does that —
but it lets the net associate the option with the correct per-slot state block
(`173 + slot*87`), which is the join it currently cannot make.

### 1b. Typed attack cost — slots 8-21

`cost_by_type[12] / 3` (slots 8-19), `colorless_deficit / 5` (20),
`can_afford_now` (21). Read from `_ATK[aid][1]`, which is already a tuple of
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

### 1c. Type matchup — slots 22-25

`opp_weak_to_my_type`, `opp_resists_my_type`, `my_active_weak_to_opp_type`,
`my_active_resists_opp_type`, from `_CARD` weakness/resistance vs the attacker's
`energyType`.

These depend only on the two actives, so they are identical for every option in
a menu — genuinely a *state* feature living in the option block. That redundancy
is the price of slice safety, and it is 4 columns; the alternative is a
retrain-gated state append. Recorded so nobody later reads it as an oversight.

One more property to pin in the code comment (review, 2026-08-04):
identical-across-the-menu means the softmax eats any additive contribution —
these four columns can matter ONLY through nonlinear interaction with
per-option features in the score head. The MLP can form that, but it is a
weaker gradient path than a per-option feature, and a linear probe would read
the columns "dead" while they are doing their job. Any future census of them
must judge variance across menus, never within one.

### 1d. Retreat + hand economics — slots 26-29

`active_retreat_cost / 4`, `retreat_payable`, `my_hand_count / 15`,
`my_bench_count / 5`.

Retreat cost is in FEAT and in `rl/combat._RETREAT` and is read by **no**
decision feature today. Hand size exists only as v4 state index 1625, so no v3
net can see it at all; bench count has no scalar anywhere.

### Files

| file | change |
|---|---|
`rl/encoders.py` | `N_OPTION_BOARD/COST/MATCHUP/ECON`, `OPTION_M41B_DIM = 145`; `_option_board_object`, `_percept_board`, `_percept_cost`, `_percept_matchup`, `_percept_econ`; write site in `encode_option_v2`; `np.zeros(OPTION_M41B_DIM)` |
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

The way through keeps the parquet untouched — deck tooling keeps its column —
and **transforms FEAT after load**, by column NAME, in both encoder twins.

Phase 2 splits into two steps with **different risk profiles**, and they must
not be conflated:

* **2a, values only.** `FEAT_DIM` stays 36, so every checkpoint still loads and
  every width still matches. The only hazard is a distribution shift.
* **2b, columns dropped.** `FEAT_DIM` 36 -> 34, so the whole vector is re-laid
  out and the silent mis-slice of 2c becomes possible. **2b must not land
  without the 2c guard.**

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
default: `scripts/m42_column_safety.py` at width 100 must stay bit-identical
under `--out`/`--compare` on a fixed corpus with the flag off (digests are
corpus-relative — the 3.3 correction), and the flag is flipped only for the
collect + train. The bundle path (`card_features.npy`, written by
`tcg.shipping.export`) inherits whatever FEAT is live at export time, so the
export must refuse to run with `FEAT_V2 = True` unless the checkpoint was
trained under it — add that as a `ship_verify` Tier-1 check.

**Review amendment (2026-08-04): the flag must not remain a hand-flipped module
constant.** A module-level bool that changes encoding semantics, edited by hand
"only for the collect + train", is ambient state in the same hazard class as
M41's mid-edit export. The 2c guard generalises to fix this: persist a
`feat_layout` version int in every checkpoint, and have CONSUMERS select the
FEAT transform from the checkpoint's declared version — loaders refuse a
mismatch instead of trusting whatever the module constant happened to be at
import time. `FEAT_V2` may survive as the collect-time bootstrap default, but
the export path and every loader key off checkpoint metadata, which makes the
ship_verify check redundant by construction rather than the last line of
defense.

Under `tests/fake_cg` there are no `skills`, so `has_ability` reads all-zero;
pin that so the column is never assumed populated in tests.

### 2b. The categorical columns — ordinal ids where a category belongs

Four FEAT columns encode an **unordered category as an ordinal scalar**, which
is the wrong representation for two reasons specific to this architecture.
`_pool` **sums** FEAT rows over a zone, so `sum(one-hot) = a histogram` (how
many Grass cards in hand) while `sum(ordinal) = a sum of ids`, which is
irrecoverable — one card with id 600 and two cards with ids 300 are identical.
And the first layer is `Linear`, so an ordinal category can only contribute
something monotonic in the id; representing "Darkness is special" costs hidden
units to build a bump function, and it cannot generalise, because interpolating
between `weakness_id` 4 and 6 is meaningless.

Measured against the pool (1,267 cards):

| column | distinct | verdict |
|---|---|---|
| `card_type` (0-6) | 7 | **DROP.** Provably a pure ordinal duplicate of the 7 one-hots beside it — the one-hot is exactly-one-hot on **0 of 1,267** exceptions. No information lost. |
| `energy_type_id` (0-10) | 11 | **DROP.** 0 mismatches against the `type_*` one-hots wherever one is set; only **4** cards have a non-Colorless id with no one-hot, and those are special energies already covered by the v4 special-energy pools and the card-id embedding. |
| `weakness_id` (-1..8) | 9 | **one-hot -> 9 cols, but DEFER** (below). |
| `resistance_id` (-1..6) | 3 | **one-hot -> 3 cols, but DEFER** (below). |
| `evolves_from_id` | 352 | **NOT a category — an identifier.** Never one-hot it; identity already has the `nn.Embedding` tables (`N_HAND_IDS`, option ids). Swap for `has_ability` as above. |

**Why defer the weakness/resistance one-hots.** Phase 1 slots 21-24 already
deliver the *decision-relevant* half of that information — "is the opponent weak
to my type", both directions — for **4 columns**. The one-hots' unique extra
value is the zone-level histogram ("how much of my remaining deck is weak to
Fire"), which is real but unmeasured. And they are not cheap: **every FEAT
column costs 28 state columns and 2 option columns**
(`STATE_DIM = 209 + 28 * FEAT_DIM`, `OPTION_DIM = 18 + 2 * FEAT_DIM`), so +10
FEAT columns is +280 state columns, a 23% state blow-up. Pay that only against a
measurement.

So 2b as scoped: **FEAT_DIM 36 -> 34** (two proven-redundant ordinals dropped,
one column repurposed). It gets *smaller*, and every remaining column is either
a bounded scalar or a flag.

### 2c. A FEAT width change fails SILENTLY — add the guard

This is the reason 2b cannot be smuggled in as "just hygiene".
`rl/matchrunner.py:767` sniffs `option_dim` from the checkpoint's own
`option_enc.0.weight`, and `forward()` truncates the *current* encoding to that
width. Both assume every earlier width is a byte-identical **prefix** — true for
appends, false for a re-layout. `OPTION_DIM = 18 + 2 * FEAT_DIM` sits at the
FRONT of the option vector, so changing `FEAT_DIM` shifts every column after it
and an old checkpoint would truncate a differently-laid-out vector to its old
width: **wrong answers, no error**. `STATE_V2_DIM` is likewise a module constant
in the `n_ids` arithmetic at `:613`.

So Phase 2 must add a layout guard before it flips anything: persist
`FEAT_DIM` (or a `feat_layout` version int) into the checkpoint and have the
loaders **refuse** a mismatch instead of sniffing through it. Without that, 2b
is a one-way door that silently invalidates every pinned baseline in
`docs/MILESTONES.md`.

The existing warm-start invariant (`load_v2_into_v3` and friends zero-init new
inputs so day one is exactly equivalent) is the right tool for **appends** and
cannot rescue a re-layout — 2b needs a full retrain, not a warm start.

## Phase 3 — verification, pre-registered

1. **The aliasing kill.** `scripts/m42_alias_probe.py` keyed on the NEW width
   must take `real::*` from **879 to 0**, with
   `harmless_true_duplicates` unchanged at ~9,807. Any residual real pair is a
   slot that was specified wrong. This is the only pass/fail bar in the
   milestone that needs no training.
   *Result (2026-08-04): MET at width 145 after the foreign-board amendment —
   real::\* == 0, no menus with real aliasing. The ours-only width 144 left
   259 real pairs (the amendment's evidence). harmless_true_duplicates read
   9,176, not ~9,807: bench_index splits same-id same-state pairs at
   different slots, which the probe's slot-blind fingerprint counts as
   harmless — they stop being aliases altogether, which loses nothing.*
2. **Feature census — new instrument.** `scripts/m41b_feature_census.py`: for
   every new slot, the non-zero rate and the value spread over real options.
   **Pre-registered kill: any slot that is non-zero on <1% of the options its
   type applies to, or has zero variance, is DROPPED before the retrain.** Dead
   columns are not free — they are capacity and noise, and this is the check
   that stops 1b being carried on a coverage argument alone.
3. **Column safety.** `scripts/m42_column_safety.py --width 100` and
   `--width 115` must both PASS against the pre-change digests, with
   `FEAT_V2 = False`. *Correction (2026-08-04): the digest is CORPUS-relative
   (it hashes encodings over `replays/**`), so a pinned hex from another day
   proves nothing — the script's own `--out`/`--compare` contract is the
   check. Result: before-snapshot taken at HEAD via stash, PASS at both
   widths over 31,021 real options.*
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
- **No FEAT width change without the layout guard of 2c.** The truncation shim
  assumes prefix-compatibility and fails silently when that is violated.
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
  misresolution stays unfixed in the TRAINED columns — still a column move.
  **Its "still latent" premise is now half-false** (amendment above): foreign
  addressing via `area`+`playerIndex` occurs in real spread-damage and gust
  submenus; only the `inPlay*` fields measured 0 foreign. Phase 1's
  `_option_board_object` resolves ownership on the board `playerIndex`
  actually names with an explicit `is_opponents` slot, so the M41b block
  reads the right board — while `encode_option`'s v1 columns keep their
  historical (mis)behaviour untouched, as append-safety demands.
- No attack-identity embedding. Within-menu attack aliasing measured 0, so
  there is nothing to fix; typed cost is the generalisation answer.
- `tests/fake_cg.CardType` still does not match the engine's numbering.

---

# Part II — a battle-tested architecture for pilot learning and the harness

Piotr, 2026-08-04: *"we definitely need to have a proper and battle-tested
architecture for the pilot learning and harness."*

Most of the learning half is **already scoped and measured** in
`docs/BACKLOG.md` items 7-10. This section does not re-derive it. It does three
things the backlog does not: it sequences that work against Part I, it adds the
**harness** half, and it removes a blocker that currently gate-keeps the right
approach.

## II.0 The evidence that already exists — do not re-litigate it

| finding | milestone | consequence |
|---|---|---|
| Wrapping the net in the solver = **+125 ELO [+90, +163]** | M40 S3 | a working policy-improvement operator. AZ's entire thesis, measured on OUR engine. |
| BC retains only **0.20 [0.07, 0.33]** of demonstrator edge | M40 X5 | **imitation provably cannot reach the band.** The current ceiling is real, not a tuning problem. |
| Imitation fidelity saturates ~0.527 on elite replays | M10 | the same conclusion from the other side. |
| Game-structure-matched net: **+8-11pp** in a CPU-scale card-game study | BACKLOG 8 | the largest single design win available — and it needs a corpus that does not exist yet. |

So the direction is not in doubt: **search is the lever, imitation is spent.**
What is in doubt is cost, and that is where sequencing matters.

## II.1 A blocker to remove first: ARCHITECTURE.md section 15 is stale

Section 15, "Measured dead ends — do not re-propose", currently forbids the
family the evidence above supports. Three of its entries are **pre-epoch**:

* *"M8.1 development-tier solver — pooled 0.492"* — solver-based, so it routes
  through `score_leaf`.
* *"Any improvement operator built on `score_leaf`'s non-lethal ranking — M12 —
  the bottleneck is `score_leaf` itself"* — explicitly `score_leaf`.
* *"Override-style consumption of any eval signal — M8.1, M12, M13"* — two of
  the three cells are solver-based.

The record states it plainly (corrected in review, 2026-08-04 — the first
draft mis-attributed this). The M37 audit found the prize-term inversion was
**systemic — 4 sites repo-wide, only 1 of them known**: `score_leaf` is one,
the collector, selfplay, and `plan.py` carried the others (the M37-audit row,
MILESTONES.md:1411). The EPOCH MARKER draws the consequence: *"every
solver-backed bed number recorded before 2026-07-31 is pre-fix era and NOT
comparable."* Those three entries were measured on an inverted leaf, and M40
S3's post-fix **+125 ELO** is the direct contradiction.

A fourth entry needs a narrower note. *"Inference-time MCTS on the classifier
value head — M8.4 — sims ladder flat"* did **not** use `score_leaf`, so the
epoch does not void it — but it ran on the *classifier* value head, which M12
found to be the wrong signal and which the E0-validated outcome-grounded value
(0.642 matched-pair, 68% within-game variance) has since replaced. Different
reason, same status: untested on the current substrate.

**Action:** add an epoch amendment to section 15, in the style of the one already
there for PPO (*"'More PPO iterations is dead' was measured on v2 nets and does
not transfer to V3"*). Mark the three entries pre-epoch and M8.4 as
substrate-superseded. This is bookkeeping, but section 15 is a "do not
re-propose" list and it currently points away from the only lever with a
positive post-fix measurement.

## II.2 The learning arc, sequenced

**Stage A — Part I of this document (the encoder).** Prerequisite, not optional.
You cannot distil a search teacher into a student whose inputs alias on 879 real
option pairs and whose damage model scores its own win condition at 0. BACKLOG 8
("game-structure-matched net, +8-11pp") is *precisely* a claim about
representation, and Part I is the representation work it presumes.

**Stage B — BACKLOG #10, distil the search.** The backlog already calls this
*"the cheap 80% of #9, and the next thing to try"*: collect a corpus whose
labels are the **composite's** actions rather than the bare net's, and fine-tune
on it. AZ's inner loop run once — no MCTS, no belief states, no engine work.
`scripts/m40_s2_collect.py` already does resumable collection against `solved:`
arms, so *the change is the arm, not the machinery*. **Hours, not weeks.**

It tests the single assumption everything downstream rests on: **is search
output learnable by our net at our scale?** If distilling a measured +125 ELO
teacher does not move the gate, the full build almost certainly would not
either — a day spent instead of a month.

Stage B should be re-run **after** Stage A, and the pairing is the point: if
distillation underperforms because the student cannot *see* what the teacher is
reacting to, Part I is exactly the fix. The A-then-B order is what distinguishes
"search is not learnable" from "the student was blind".

**The power arithmetic (review, 2026-08-04).** The plausible effect is bounded
by the record: the teacher's edge is +125 ELO (M40 S3) and BC retention of a
demonstrator's edge measured 0.20 [0.07, 0.33] (M40 X5) — so even if Part I
doubles retention, the student's plausible gain is **+25 to +60 ELO ≈
3.5-8.5pp** near 0.500. A 400-game cell resolves ~14pp (±7pp half-width).
Run naively, Stage B's null would be uninterpretable for a second reason the
A-then-B ordering does not touch: **power**. Detecting a true 3.6pp at 80%
power takes ~3,000 games. Hence the hard pairing with § II.3c: either the
adjusted estimator shrinks the interval, or the cell runs at n≈2-3k — eval
games are cheap next to the month they gate. Bar and n go into the § II.3d
spec file before the run.

**What the teacher actually emits (engine review, 2026-08-04).** Read
`wrap_with_solver` (`rl/turn_solver.py:526`) before designing the corpus: the
composite is not a search policy — it is the net with a sparse override.

* The inner net runs on EVERY prompt (its v4 memory side effects require it)
  and the solver replaces the answer only when a trigger fires AND the tier's
  bar clears (>=1 prize / win, or the dev margin). On every other prompt the
  composite's action IS the bare net's action. A distillation corpus therefore
  agrees with the student's own argmax almost everywhere, and the whole
  +125 ELO lives in the sparse overridden prompts — uniform cross-entropy is
  mostly the student learning itself, a plausible mechanism for X5's 0.20 that
  Part I does NOT fix. The wrapper already counts exactly the needed quantity
  (`stats["fire_*"]`, `stats["changed"]`): persist per-row `fired`/`changed`
  flags in the collect, report the divergence rate as a collect-time
  measurement, and weight or isolate divergent rows in the loss (upweighting,
  or divergent-rows-only with a KL anchor to the base policy on agreeing
  rows). If the solver changes <~2% of prompts, loss design is not garnish —
  it is the experiment.
* **Soft targets already exist.** `score_siblings` (`rl/turn_solver.py:461`)
  returns per-sibling deep scores at the root — ranked labels with margins, no
  new engine work. Distilling margins or the ranking preserves strictly more
  of the teacher than one-hot argmax (the ExIt/AlphaZero lesson: the search's
  *distribution* is the label). Epoch note: this is the M12 ranker machinery,
  whose dead-end entry is one of the three § II.1 marks pre-epoch — consumed
  as LABELS it is exactly what the amendment re-opens; consumed as a live
  override it stays dead (the override law is untouched).
* **The state distribution is the teacher's, not the student's.** Composite-
  as-actor collects states the composite reaches; the student will act from
  its own. Covariate shift is the textbook BC failure and a second candidate
  cause of X5's 0.20. Cheap mitigation, DAgger-shaped: mix in shards where the
  BARE net acts and `solve_turn` labels without overriding — the wrapper
  computes both actions on every prompt anyway, so logging `(base, solver,
  fired, changed)` per row costs nothing extra, and even a 50/50 mix bounds
  the shift.
* Two design checks before the run: how `rl/bc.py` treats **multi-select**
  rows today (the solver picks combos under `MAX_MULTI_COMBOS`, the net ranks
  options independently — the loss must not pretend a set label is a single
  action); and the student's value head gets its own E0 read — 0.642 is
  checkpoint-specific (`m39_retain_b`; the same instrument read 0.469 on
  `cont3`), so a distilled student inherits nothing.

**Stage C — one cheap engine probe that moves a 10-50x constant.** BACKLOG 9(b)
records the open question: whether `cg.api.search_begin/step/end` supports
**re-descent from an arbitrary node** or only forward DFS. Reading the header:
`search_step(search_id, select)`, `search_release(search_id)`, and *"memory used
during the search will be reused"* — per-state ids plus an explicit release
strongly suggest multiple concurrent states, i.e. re-descent. **A half-day probe
that decides a 10-50x constant on the entire search programme**, and it should
run before anyone estimates #9 again. It depends on nothing else in this plan —
run it in any idle half-day, e.g. while a Stage B collect occupies the box.

Two more facts from the same header, worth recording because they lower 9(a):
`search_begin` takes **predicted** opponent deck / prize / hand, and
`manual_coin=True` lets the caller choose coin outcomes. The API is *natively a
determinization interface* — ISMCTS over belief samples is the shape it was
built for, so 9(a) drops from "build belief-state search" to "sample beliefs and
feed them in". Strategy fusion remains the real risk and is not solved by the
API.

**Stage D — the full build (#9 PSRO / #7 PPO / #8 inductive bias).** Gated on B.
The backlog's cost note stands and should be quoted at whoever proposes it:
*"~58k games/day; convergence 1e5-1e6 games, so 2 days to 1 month, times
determinization count. Consumes a whole campaign; do not start one inside a
deadline."* And 9(d): a card pool is a *family* of games plus an outer
deckbuilding optimisation, so the correct frame is **PSRO / double oracle**, not
AlphaZero — the bed roster is already a hand-maintained version of that
population.

## II.3 The harness — the weaker half, and the one nobody has systematised

The learning direction has a measured basis. The **measurement layer does not**,
and its failure record is worse than the model's:

| failure | cost |
|---|---|
| M37 audit: prize inversion at 4 sites, 1 known | a whole era of solver numbers void, including the campaign bar |
| M40 X5: bed ABSOLUTES void as band claims | clone retention asserted 0.33-0.50, measured **0.20** |
| M42: **six** instrument corrections in one milestone | four near-misses that would have shipped rules against phantom defects |
| M41: export ran mid-edit, shipped a half-finished encoder | caught by a QC sweep that could easily have passed |
| M41: `test_bundle_twins` compared only `plan.py` | a twin divergence survived a full milestone |

Two of the six M42 corrections were caught only because each produced something
impossible to ignore — a rate above 100%, and a flat contradiction between two
instruments (`docs/M42.md:288` is explicit that both discoveries were
accidents; the first draft of this plan said "four", which the diary does not
support). Review caught the rest, and review does not scale. That is not a
system.

### II.3a Golden fixtures for every instrument (highest value)

Every probe and every forensic flag gets a synthetic fixture whose ground truth
is true **by construction**, asserted in the suite. M42 did this for the
post-mortem flags (a `fires` case and a `guards` case each, 27 tests) and it is
why those flags are the only instruments in the repo currently trustworthy.
Extend it to every probe — and note (review, 2026-08-04) that the glob
`scripts/*_probe.py` matches 19 files and silently misses
`scripts/pm_probe2.py`. Rename it to match, or enumerate explicitly; otherwise
the completion criterion can pass while skipping a probe, which is precisely
the class of hole this section exists to close.

Concretely, this is what would have caught 4 of the 6 M42 bugs before a single
game was played: a fixture where the pilot has *no legal way out* pins that
`retreat_stranded` must not count it; a fixture where the stadium is played at
prompt 5 of a turn pins the per-turn semantics; a fixture with a benched
evolution pins that an in-play Pokemon is never a dead fetch.

### II.3b Cross-instrument agreement, as a check rather than a coincidence

The stadium bug surfaced only because the probe and the post-mortem flag
disagreed on the same games. Make that deliberate: for any behaviour measured
two ways, a CI test asserts the two instruments agree on a fixed replay corpus.
Keep the two implementations **independent** on purpose — a shared helper would
make them agree while both being wrong, which is the opposite of the property
wanted.

### II.3c CRN / paired evaluation is impossible here — reduce variance elsewhere

Worth settling because it is the field's standard first answer and it does
**not** apply. `cg.api` exposes **no seed at all**, and `docs/VALIDATION.md:228`
records why a seed would not help anyway: the engine RNG is per worker process
and `mp.Pool` assigns chunks by timing, so *"every cell is an independent sample
and a 'seed' is just a label."* Five identical commands at seed 1 gave
**0.471 / 0.494 / 0.526 / 0.506 / 0.537**.

So common random numbers are off the table. The remaining lever is the **prize
differential** `_engine_game` already records (`stats["final"]`, per-seat deck
counts and prizes remaining) — but HOW it enters matters, and this was decided
with Piotr, 2026-08-04: **as a control variate, not as a replacement
estimand.** Scoring margin *instead of* win/loss tightens a CI around a
*different quantity* — every gate bar in the record is defined on win rate, and
a margin CI does not decode a win-rate bar. The standard tool is regression
adjustment (CUPED): estimate the win rate, subtract the margin's correlated
noise — fit win ~ margin within the cell, report the adjusted mean and its
interval. Same games, same estimand, tighter interval, and every historical bar
keeps its meaning.

**Pre-register the check:** re-analyse a completed battery's JSONLs both ways
and compare win-rate CI half-widths, raw vs adjusted. If adjustment does not
shrink the interval materially, drop it and say so — do not carry it on theory.
**Pre-register the fallback too:** if dropped, Stage B does not run at n=400.
The power arithmetic in § II.2 puts the plausible Stage B effect at ~3.5-8.5pp
while a 400-game cell resolves ~14pp — the honest alternative is n≈2-3k for
the Stage B cell, and whichever n it is goes into the § II.3d spec file before
the run.

### II.3d Gates as code, not as a checklist

Every milestone hand-rolls a `*_decide.py`. Replace with one runner over a
**pre-registered spec file** (arms, bars, n, the read) written and hashed
*before* the run, which refuses to decode a run that does not match its spec.
That is what stops a bar moving after the numbers land, and it is the machine
version of a discipline the diaries currently keep by hand.

### II.3e Push the ship guards into the gate itself — there is no hosted CI

A fact the first draft glossed (review, 2026-08-04): this repo has **no CI
service at all** — no workflows, no runner. "CI" here can only mean the pytest
suite plus the QC battery, and both are scripts someone remembers to run — the
exact failure mode this section exists to close ("M41: export ran mid-edit" was
not a missing check; it was a present check nobody ran at the right moment).

Decided with Piotr, 2026-08-04 — two structural moves, no hosted CI:

1. **The export becomes self-gating.** `tcg.shipping.export` runs the
   `ship_verify` Tier-1 invariants itself and REFUSES to write the tarball on
   any failure: twin parity (M41), column safety at every trained width, the
   FEAT layout version (Part I § 2c), bundle purity, the encoder width chain.
   All cheap and deterministic. The gate travels with the dangerous action
   instead of preceding it by convention.
2. **One hardened gate script.** `scripts/ci_gate.py`: the full pytest suite +
   `ship_verify` Tier-1 + the probe golden fixtures of § II.3a, one command,
   one PASS/FAIL line — the thing a human runs before any consequential
   action, and the first thing `scripts/qc_battery.py` runs so the replay QC
   never measures a broken artifact.

The invariants also land as ordinary pytest tests so the suite catches them at
the earliest possible moment; the two moves above are what make them
unskippable at the moments that have actually burned us.

## II.4 What Part II does NOT commit to

No architecture change is justified until **Stage B** returns a number. The
point of the backlog's item-10 framing is that a day of distillation decides
whether a month of PSRO is worth starting, and that ordering must not be
inverted because a rewrite is more interesting than a probe.

Nor does Part II propose belief-state search, batched inference, or a population
loop as work items yet. They are Stage D, they are correctly parked on compute
grounds, and 9(b)'s re-descent probe may change their cost by more than any
design decision available today.

## II.5 The two forks — RESOLVED 2026-08-04

Both were put to Piotr and both resolved to the recommendation; the decided
order is at the top of this document.

1. **How far to commit** — Stage A then B. Not Stage D: BACKLOG 9's *"do not
   start one inside a deadline"* stands, and Stage B is the day that decides
   whether the month is worth spending.
2. **When to harden the harness** — before Stage B, for the reason that made the
   question worth asking: Stage B is a single gate cell deciding a month of
   work, and the instruments that would measure it took six corrections in
   their last milestone.
