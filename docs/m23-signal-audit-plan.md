# M23 signal audit — locate the bottleneck before rewriting anything

**Status:** ✅ COMPLETE (2026-07-21 evening, same-day). All probes resolved; verdicts in
`docs/M23.md` (18:27 entry). Headline: **architecture exonerated (E1 0.5625 ≥ 0.54), the
self-play data carries no signal for winning behaviors (S2 ~zero/negative), everything
else healthy** → M24 = replay-BC primary signal + exogenous opponents. Tools:
`scripts/m23_signal_audit.py`, `scripts/m23_e1_gates.sh`.

**Decisions (Piotr, 2026-07-21 evening):**
- Audit runs BEFORE any architecture rewrite; its branches select the M24 change.
- S4 edit to `rl/ppo.py` (advantage-by-option-type logging) approved.
- Clone-as-teacher decision deferred until S2/S3 verdicts are in.
- Replay-BC on other teams' public replays is ALLOWED as a ship path — if E1 exonerates
  the architecture, the leaderboard-replay corpus becomes the primary training signal.

**Pipeline-review corrections (exploration, this session):**
- The E1 "adapter gap" below is already closed: `BCDatasetV3` (rl/plan_iter.py:586-598)
  shims plan-less shards to plans=zeros / plan_labels=-1 / weights=1, so
  `rl.plan_iter train` on `data/bc_clone_54618168/` trains V3-as-BC directly. The REAL
  gap is the fresh-init path hardcoding `n_state_ids=20` while the clone shards carry
  width 12 — fixed width-driven (from the data) this session.
- S1 wrinkle: our PPO checkpoints are v4 (states 1702, 25 ids) vs the clone corpus's v2
  encoding (1361, 12 ids). S1 zero-pads the v4 extra block + missing ids (the legitimate
  "no history / empty" defaults). Absolute agreement is depressed by the shift; the S1
  read is the WITHIN-checkpoint contrast between option-type classes, which cancels most
  of it. The V2 clone runs as a harness control (must reproduce ~0.664 fidelity).
- S2/S3/S5/S7 data source: the surviving m23p1 final-iteration shards in `data/ppo/`
  (400 games, v4 schema incl. stored `plans` — collection-time logits reproducible with
  `ppo_current_m23p1.pt`). Do not clobber `data/ppo/` before these probes run.

## Why this exists

The Phase 1 leg plateaued at ~35–39% vs the teacher, which nominally points at Phase 2
("rewrite the plan head"). But `bc_clone_54618168` — an **OptionScorerV2**: no plan head, no
search, no v4 memory — plays 0.59 vs the sample agent by imitating a strong policy. **Strong
play is representable in a function class strictly smaller than ours, so capacity is not the
bottleneck.** The open question is why our RL loop can't *reach* that region: exploration,
credit assignment, optimization, or a genuinely dead component. This audit decides which,
with pre-registered thresholds, before any milestone-sized rewrite is scoped.

Standing discipline: every signal below gets its threshold written down BEFORE the number is
looked at; sub-MDE deltas are not results; in-loop evals are triggers, never evidence.

## Ordering and the decision tree

Run S1–S3 first (nearly free, existing data). Then E1 (the discriminator). S4–S7 are logging
additions that ride along with the next training leg, whatever it is.

```
S1 agreement collapses on PLAY-supporter/EVOLVE only
        → deficit is localized card-economy decisions (not diffuse weakness)
S2 no supporter↔win correlation in OUR self-play data
        → the signal PPO needs is ABSENT from the data → opponent/exploration problem,
          no optimizer or architecture fix will help
S3 sampling policy rarely tries supporter lines (<~5% mass in candidate states)
        → exploration ceiling → guided exploration / different behavior policy, not a rewrite
E1 V3-as-BC reaches clone-level strength
        → architecture EXONERATED → the bottleneck is the RL training signal;
          leaderboard-replay corpus becomes the primary signal source
E1 V3-as-BC clearly below the V2 clone
        → our arch is implicated (prime suspect: plan-conditioning path) → S6 decides
          whether Phase 2 = enrich the plan head or DELETE it
```

## S1 — shadow-eval agreement by option type (existing data, ~1h)

Run our checkpoints (`ppo_current_m22cRL`, best m23p1) over the 26,957 encoded decisions in
`data/bc_clone_54618168/` (states are encoded exactly as the live inference path — verified
in M23) and log top-1 agreement with the 1251-score target, broken down by option type × turn
bucket. No training, pure forward passes.

- Read: agreement on ATTACK/ATTACH vs PLAY-supporter/EVOLVE. **Pre-register:** a >20pp
  agreement gap between attack-class and card-economy-class decisions = "localized deficit".
- Also log the probability our policy assigns to the target's chosen action (not just argmax)
  — mass tells more than rank at a plateau.

## S2 — supporter↔win correlation in our own self-play (existing counter, ~1h)

Over fresh self-play shards (NOTE: `rl/ppo.py` deletes `ppo_shard_*.npz` each iteration —
only the final iteration's shards survive a run; collect a dedicated small batch if more is
needed), correlate early-game supporter plays (t2–t4, the M23 counter) with game outcome.

- **Pre-register:** if the correlation is materially positive and the policy still declines
  supporters, credit assignment is failing (→ S4/S5 matter most). If it is ~zero, the signal
  is absent from our data distribution and the fix is opponents/exploration, full stop.

## S3 — behavior-policy action-type frequencies (existing shards, ~30min)

From collect shards: frequency of each option type being *sampled* during training (the
plan-tau'd behavior policy, not greedy eval), restricted to states where that type was on the
menu. **Pre-register:** supporter-play sampled in <5% of its candidate states = exploration
ceiling; PPO cannot reinforce what never happens.

## E1 — the discriminator: distill strong play into OUR arch (~half day)

BC-train an **OptionScorerV3** (our exact architecture) on the clone corpus
(`data/bc_clone_54618168/`, optionally + more snowballed replays) and run the same probes as
the V2 clone: vs `rule:lucario` and vs `rule:dragapult`, n=200×2 seeds each.

- Known gap to close first: `rl/bc.py train` handles v1/v2 only; `rl/plan_iter.py` trains V3
  on plan-schema shards. The adapter is either (a) a V3 path in bc.py feeding null/zero plan
  vectors, or (b) extending the replay shards with null plan columns for plan_iter. (a) is
  smaller. The M23 width-driven collate work already removed the encoding mismatch.
- **Pre-register:** V3-as-BC within ~5pp of the V2 clone vs `rule:lucario` (i.e. ≥~0.54 at
  n=400 pooled) = architecture exonerated. Clearly below (≤~0.45) = arch implicated → S6.
- Caveat carried from M21: BC-relabel of a post-PPO checkpoint un-learns PPO gains (Gate A).
  E1 trains FRESH from strong replays — do not warm-start from a PPO checkpoint.

## S4 — advantage mass by option type (logging addition, next leg)

In `rl/ppo.py` where GAE advantages already exist: per-iteration mean/std of advantage for
PLAY-supporter / ATTACH / ATTACK / END actions (TB scalars + one print line).
⚠ `rl/ppo.py` is Piotr's — sign-off before touching, same as the promotion edit.
- Read: supporter advantages ≈0 or noise-drowned while attack advantages are clean →
  credit is not reaching card-economy actions.

## S5 — value-head calibration by turn (logging addition)

Correlation of V(s_t) with final outcome, bucketed by turn, computed from collect shards
(values are already recorded for GAE). **Pre-register:** if early-turn (t1–t4) correlation is
near zero while late-turn is strong, the critic cannot price setups — early actions have no
usable baseline, and Gap-A behavior can't be learned from prize+win reward alone.

## S6 — plan-consistency rate (measurement, decides the plan head's fate)

Fraction of within-turn actions consistent with the committed plan (target/attacker named by
the plan vs what was actually attacked/loaded). Computable from replay/series instrumentation
in the `rl/behavior.py` style. **Pre-register:** consistency at chance level = the plan head
is decorative (M21 two-head deadlock, generalized) → the Phase 2 conversation includes
DELETING it, not only enriching it.

## S7 — policy entropy by turn (logging addition)

Per-turn-bucket entropy of the option head during collection. Premature determinism in t1–t4
would explain why exploration never finds setup lines despite plan-tau.

## Inputs already in place (do not re-derive)

- Clone + corpus: `checkpoints/bc_clone_54618168.pt` (fidelity 0.664; 0.59 vs rule:lucario,
  0.25 vs rule:dragapult), shards `data/bc_clone_54618168/`, target deck
  `data/kaggle/clone_54618168_deck.csv`. Target's replay behavior profile in `docs/M23.md`.
- Supporter counter + seat-correct `from_series` (`rl/behavior.py`), width-driven v2
  trainer/loader (`rl/bc.py`, `rl/matchrunner.py`, `rl/policy.py`).
- Dragapult stays evaluation-only (now hard-enforced in `parse_pool`).
- Phase 1 verdict + gate battery numbers: `docs/M23.md` (check there first — this audit's
  framing assumes the plateau held through iter 19 and the dragapult gate).
