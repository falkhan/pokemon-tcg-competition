# M28 plan — three axes off the M27 dead end

## Context

M27 established that **teacher fidelity is not the binding constraint**: six
milestones of fidelity work (M22c→M27) moved no gate beyond noise, and `m27_both`
put Boss's Orders / stadiums / the trainer aggregate at teacher rates while
measuring **non-inferior on every strength axis** (lucario 0.6863 n=800,
dragapult 0.3738 n=800, grim 0.6700 n=400, kyogre 0.9700 — nothing resolved).

The live decomposition (132 games, wr 0.508 — `docs/M27.md`) found one
structural cell and a lot of flat: five archetypes are 62% of the field at
0.48–0.70, and **Darkness decks are 8.3% of the field at 0.18** (2W–9L,
Fisher p=0.029) because 12 of our 19 Pokémon are weak to Darkness.

Piotr's call: run **three axes** — MCTS (killed only on the v1 value head), PPO
from the now-much-stronger clone base, and the Darkness problem — with the
Darkness track going all the way to a **full deck search**.

**Standing constraint:** measurement is the meta-problem. Live MDE ≈29pp at
n≈45; a +3–4pp fix is invisible to every instrument we own. Each track below
states what would count as a result *before* it runs.

---

## 🔴 Hard rules

- `rule:dragapult` stays the sealed out-of-loop floor — **evaluation only**,
  never in a training pool, never in a deck-search fitness function. Same for
  `sample-agent-dragapult/main.py` (unread).
- Workers ≤8. Diary incrementally into `docs/M28.md`. Hermes at phase
  transitions. Control retrain before any per-class claim.
- No ship unless Piotr explicitly asks; pre-ship human QC (3 bundle games →
  `replays/` → manual review) applies if that changes.
- **Deck changes have burned us twice** (M18.1, M22c fossil-deck ships). Any
  deck work carries the `--deck` + md5 ritual, and the cloned policy is tied to
  the deck it was cloned on — see Track C's re-encoding requirement.

---

## Track A — Darkness (build the instrument first)

The 0.18 cell is the only structurally-evidenced lever. It is currently
**unmeasurable**: the M26 grim gate has us at 0.6875 offline against a family we
go 2–9 against live, and the clone behind it scores **0.148 vs `rule:dragapult`**
(measured in M27) — a strawman.

**A1 — register the deck.** ✅ done: `decks/grimmsnarl_dark.csv`
(md5 `dc8c23cd`, 10 Darkness / 4 Water / 4 Psychic attackers), so it is a
first-class named deck rather than a `data/kaggle/` path.

**A2 — select the sparring pilot by measurement, not assumption.** Run every
pilot we can put on that deck against a fixed reference and pick the strongest:
`model:m25_bc_grim_54861775` (the killed clone) · `solver:grimmsnarl_dark` ·
`generic:grimmsnarl_dark` · `rule:<agent>:grimmsnarl_dark` for each sample-agent
brain. Reference: `rule:lucario` (a fixed, non-Darkness yardstick) at n=200×2.
**Result = the strongest becomes the canonical Darkness opponent**, and its
number is the new pin. If the best available pilot is still far below the live
agents' strength, say so — an advisory gate that we know is a strawman is worth
less than no gate, and that judgement gets recorded.

**A3 — re-measure the current champion against it.** `m27_both` and the shipped
`m25_bc_alakazam_v3h` vs the A2 opponent, n=200×2. If we still read ~0.69 while
live says 0.18, the gate is *still* a strawman and Track C proceeds on live
evidence alone.

## Track B — MCTS resurrection (cheapest to falsify)

M8.4 killed MCTS on a flat sims ladder (0.515/0.490/0.495) — **on the v1 value
head**, a caveat `docs/M22.md` states explicitly. We now have a v3h value head
trained on 28k rows of a 1251-score agent.

**Blocker to size honestly:** `rl/mcts.py` is written against `OptionScorer`
(v1) and the v1 encoders (`encode_state` / `encode_option`, `COMBAT_SLICE`).
`mcts_search` itself (PUCT/expand/backprop, `rl/mcts.py:160`) is implemented and
architecture-agnostic — it consumes `Node.P` / `Node.value`. The port is
therefore confined to `evaluate()` and `make_node()`: swap the v1 encode calls
for `encode_state_v3` + `encode_option_v2`, feed `OptionScorerV3` with a zero
plan, take priors from `logits` and value from the value head.

**B1** — port `evaluate`/`make_node` to V3 behind a flag; parity-test that a
1-sim search reproduces the greedy argmax exactly.
**B2** — re-run the M8.4 sims ladder (16/32/64) on `m27_both`'s value head vs
`rule:lucario`, n=200×2 per rung.
**Kill bar (pre-registered):** the M8.4 result was a *flat* ladder. If sims
16→64 moves the win rate by less than its MDE (~10pp at n=400), that reproduces
the kill on the new head and MCTS is closed for good — record it and stop.
Latency also gates it: the G6 budget is ~50ms/move and the M22 `solved:` sweep
already found search changing almost no actions.

## Track C — PPO from the clone base

M20/M21 ran PPO from a **0.48** mirror base; M22c-RL swapped the teacher to the
sample agent and out-of-loop never moved. We are now at **0.686**. Untested from
this base.

**C1** — PPO from `m27_both` (or the shipped ckpt — pick by A3), KL-anchored to
the clone, opponents = the field-representative gauntlet from Track D, NOT the
sealed dragapult.
**Kill bar:** M20's lesson was peak-then-decay; the anchor exists for that. Kill
if out-of-loop (`rule:dragapult`) drops below the 0.3738 pin by more than its
MDE at the n run, or if the mirror collapses as in the M21 Gate A kills
(18–26pp, well above MDE — that shape is detectable).
**Ownership (corrected):** the "PIOTR (the core)" markers in `rl/ppo.py`'s
docstring are stale — `docs/M21-plan.md:13` records the transfer ("Piotr steps
back to orchestrator; Claude owns the RL core now, including the GAE/PPO math"),
and `compute_gae`/`ppo_update` are implemented, no `NotImplementedError`. The
core is editable; keep the discipline the old rule enforced (twins in parity,
suite green, algorithm changes flagged in the diary + Hermes).

## Track D — full deck search (Piotr's call)

**D1 — a field-representative gauntlet.** The current battery is three opponents
that between them do not resemble the live field. Build a gauntlet weighted by
the *observed* field shares (`docs/M27.md`): alakazam mirror 15.9%, lucario+riolu
15.2%, drakloak 11.4%, archaludon 9.8%, lucario+solrock 9.1%, Darkness 8.3%.
Decks come from `opp_decks.parquet`; pilots from A2's selection method per deck.
This gauntlet is the deck-search fitness function AND the honest replacement for
the current battery.

**D2 — search the 60** with `rl/deck_build.py` / `rl/deck_search.py` against D1.

**D3 — the coupling that must not be forgotten:** the policy was behaviour-cloned
on `clone54618168`, and `encode_state` carries **deck-context pools** of the full
60 and the remaining deck (`rl/encoders._deck_pools`). A new decklist therefore
moves the policy off-distribution *by construction*. Any candidate deck must be
re-measured with the policy it will actually ship with, and probably needs the
corpus re-encoded against the new deck before the pairing is judged. Budget for
that or the result is another M18.1.

---

## Track E — a real value function (added 2026-07-22, after B and C both died)

**Why this is now the top item.** B (MCTS) and C (PPO) both failed, and both
failures trace to the same organ: the value estimate. `rl/value_train.py`'s
docstring already states the measurement — *"bc_v1's original value head scored
**0.62 sign-accuracy (≈ chance)**; a supervised head hits **~0.87** on the same
frozen features"* — and names the mechanism my M28-B post-mortem reached
independently: *"the value head was OOD on determinized/deep search states … so
train it on exactly those states"*, flagged there as **"documented-but-never-
executed"**. Our auxiliary head is a `0.5 × Huber` afterthought on a BC
objective. This is the common root of M12 (solver), M22 (`solved:`), M28-B
(MCTS) and PPO's critic.

**E1 — port `rl/value_train.py` to v3.** It is v1-only today (`OptionScorer`,
`encode_state`/`encode_option`). Same shape as the `V3Evaluator` port: sniff
`plan_enc.0.weight`, use `encode_state_v3` + `encode_option_v2`, carry
`state_ids` through the npz, and train through `OptionScorerV3._trunk` with a
zero plan. Body and policy stay frozen — only `value_head` gets gradient.

**E2 — collect with `--search-plies > 0`.** The states MCTS actually evaluates
are determinized search states, not on-policy game states. `_search_visited`
(`rl/value_train.py:29`) already generates exactly those, with the negamax
perspective sign. This is the never-executed half of the fix.

**E3 — the gate is sign-accuracy, measured before anything downstream.**
Pre-registered: the port is worth continuing only if the trained head clears
**≥0.75** held-out sign-accuracy against the auxiliary head's baseline (expect
~0.6). Report the baseline→trained delta on the same held-out split. If it does
not clear, the "bad leaf" theory is wrong and Track E dies there — cheaply,
before any search is re-run.

**E4 — only then, re-run the M28-B ladder** with the retrained leaf.
Pre-registered: MCTS reopens only if 16 sims is **non-inferior** to no-search
(the current gap is −11.0pp); a second flat/negative ladder closes lookahead
permanently, this time with the leaf excuse spent.

**Kept honest:** a better critic is *also* the ingredient PPO lacked
(`--value-ckpt` already wires a `value_train` checkpoint into `rl/ppo.py:481`),
so E has two downstream consumers, but neither is promised anything by E3.

## Track F — demonstration quality (the only axis with a track record)

Replay-BC is the one thing that ever moved the out-of-loop number: 0.22 → 0.29
→ 0.31 → 0.34 → 0.37 across M23–M26, and **all of it came from better data, not
better process**. Meanwhile our corpus is 370 games from a single agent and
**166 of them (45%) are games that agent LOST**, cloned at uniform weight, with
`replay_bc build --winners-only` (`rl/replay_bc.py:330`) existing and never used
(`docs/M27.md`). Fidelity is identical on won and lost rows (0.667 / 0.668) —
we clone the losses as faithfully as the wins.

**F1** — build a winners-only corpus and retrain. Cost: halves an already small
corpus (370 → 203 games), so this is a real bias/variance trade, not a free win.
**F2** — if F1 helps, try outcome-*weighted* rather than outcome-*filtered*
(bc.py:549 already implements the soft form) to keep the losing games' states
while down-weighting their actions.

## Track G — phase-conditioned BC (Piotr's "weights over game")

`docs/M27.md` Probe 2 measured competence collapsing through the game:
val_acc **0.738** at deck 30+ → **0.601** at deck 7–15 → 0.644 at deck ≤6, with
only 13.2% of the corpus at deck ≤6. That is exactly the region where deck-out
is decided. The net's only phase signal is `turn/30` and `deckCount/60` — 2 raw
scalars among 1,708, with no interaction term, which is *structurally identical*
to the supporter defect M27 fixed.

M27's lesson is the design constraint: **features alone did nothing; features ×
weighting was super-additive.** So run both, as a 2×2:

**G1 — phase row weighting.** `--phase-weight` on `rl/plan_iter.py`, keyed on
the row's remaining deck count, upweighting the late-game rows where fidelity is
worst. Generalizes `apply_card_kind_weights`; same policy-CE-only channel,
`evaluate()` stays unweighted.

**G2 — phase-interaction option features.** Append to the M27 option block
(`OPTION_M27_DIM` → +3) genuine *interactions* that vary per option AND per
state — `is_turn_ending × deck_low`, `is_ability × deck_low`,
`is_supporter × deck_low`, where `deck_low` grades how close we are to decking
out. A per-state constant would be useless (the score head already sees the
state tower); the interaction is what the two-tower architecture cannot form
cheaply.

**Gate:** the M27 instruments (`turn_discipline_report`, `trainer_play_report`,
`class_report`) for behaviour, then the standard battery for strength, against
a **no-op control retrain** (mandatory — M26 calibration lesson). Strength, not
fidelity, decides: M26 and M27 both bought fidelity and no strength.

## Sequencing

A2 → A3 first: cheapest, unblocks D1's Darkness cell, and its outcome decides
whether Track D can be judged offline at all. B in parallel (independent, and
its kill bar is pre-registered so it either dies fast or earns more). C and D
after A3, because both need D1's gauntlet to be judged honestly.

## Verification

- Every strength number: 2 seeds minimum, pooled, decoded with the **0=WIN**
  rule, quoted with its MDE and a `not resolved` verdict when |z|<1.96.
- `uv run pytest tests/ -q` green (630 at `feature/m27` head) before any claim.
- Any deck touched: `--deck` explicit + tarball md5 vs `decks/<name>.csv`.
- Re-measure the shipped baseline on any NEW instrument before reading a
  candidate against it (the A3 rule, generalized).
