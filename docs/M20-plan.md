# M20 plan — PPO revival probe: over-attach as a negative reward (Piotr's idea)

**Status:** approved 2026-07-19 (Piotr: implement fully, explain in plain
language, KL-anchor as leg B, over-attach penalty only, 2 legs × 8 iters).
Diary: docs/M20.md. Owner note: `ppo_update`/`compute_gae` are Piotr's core —
this milestone's diffs there are pre-authorized by the plan approval but
flagged for his review.

## Why this doesn't violate the M8.3 dead end

The ruling (MILESTONES.md:454) killed "**more PPO iterations on the current
loop**": sparse ±1 outcome reward (+ potential-only shaping) on V1/V2 nets
from BC anchors, best result 0.359 vs 0.345 @ n=800 (noise). Three things
here are new and untested:

1. **The reward**: a per-step **action penalty** for over-attach — a
   non-potential term that deliberately changes the optimal policy. All
   prior shaping was potential-based (provably optimal-policy-preserving).
2. **The net**: OptionScorerV3o (plan-conditioned, hand-aware,
   option-identity) postdates every PPO run by 3+ milestones.
3. **The anchor**: plan5 at 0.4294-vs-solver (the M8.3 era anchored on 0.345).

And M19 gives the motivation: BC class-reweighting fixes behavior by paying
strength (−3pp to −10pp, dose ∝ reweighted-row fraction). PPO's clipped,
advantage-scaled updates + a KL rubber band are the surgical version of the
same intent.

## How this works — plain language (Piotr's requested explainer)

- **PPO in five sentences.** The net plays games and we record, for every
  decision, what it saw, what it picked, and how confident it was. Each
  decision gets a score ("advantage"): did things go better after this
  choice than the critic expected? Choices with positive advantage get
  nudged more probable; negative, less probable. The "clip" caps how far
  any single update can push a probability, so one noisy batch can't wreck
  the policy. Repeat: play → score → nudge.
- **Why the penalty is different from what we tried before.** Old shaping
  was "potential-based": a bonus for *reaching* a better board state, built
  so it mathematically cancels out over a game and can't change what the
  best policy is — it only speeds learning. The over-attach penalty is the
  opposite kind on purpose: every time the net attaches energy to a Pokémon
  whose best attack is already paid, that decision's reward gets docked
  immediately. Dense, targeted, and it *does* redefine what "best" means:
  best play now includes "don't waste attachments".
- **The KL-anchor (leg B).** KL is a measure of how far the new policy's
  probabilities have drifted from a reference policy. We add a loss term
  `kl_coef × KL(new ‖ plan5-frozen)` — a rubber band tied to the champion.
  The penalty pushes down over-attach; the rubber band pulls everything
  else back toward plan5. This is exactly the failure mode M19 measured in
  BC reweighting (fix the behavior, lose the rest), addressed structurally.
- **The knobs.** `--defect-penalty`: size of the dock per over-attach
  (0.1 = same magnitude as taking one prize under the existing
  prize-shaping). `--kl-coef`: rubber-band stiffness (0 = off).
  `--lr 3e-5`: small steps (M8.3 found bigger steps diffuse the policy).
  `--entropy-coef`: exploration bonus; if the "entropy" printout RISES over
  iterations the policy is dissolving toward random — kill signal.
- **Reading the per-iter printout.** `defect/g` = over-attaches per game in
  that iteration's own games (the target: should fall). `entropy` = policy
  sharpness (should stay flat-ish, ~0.6). `vs_solver` = strength spot-check
  at n=200 (noisy — only n=800 verdicts count). `ratio` ≈ 1.0 = updates
  are in-trust-region.

## Implementation (files + changes)

### `rl/collector.py`
- **V3 learner path** (sniff `plan_enc.0.weight`): arch from
  `option_dim_of` + the state-width formula (copied from
  `rl/matchrunner.py:341-345`); `encode_state_v3` when n_ids>12;
  `encode_option_v2` (94-wide). Turn-scoped **plan state machine** exactly
  as `matchrunner.py:356-384` (plan once at each turn's first own MAIN via
  `act_plan`, hold for submenus, zeros otherwise) — plan choice stays
  greedy; exploration remains option-level sampling (closest to the
  anchor's serve-time behavior).
- New shard column **`plans`** (the plan vector each decision was scored
  under). V1/V2 shards unchanged.
- **Defect-penalty hook** at the existing reward site: if the sampled
  action is an ATTACH option (MAIN context) whose target's charged-best is
  already paid (`rl/combat._turns_to_ready(target, opp_active, board_ids)
  == 0`, target resolved from live obs like `rl/encoders._my_poke_at`) →
  `reward -= defect_penalty`, count it.
- Workers return `(shard_path, n_defects, n_learner_games)`; `collect()`
  prints and returns the aggregate defect rate.
- V3 without a decks file is allowed (fixed learner deck — the probe pilots
  `lucario`, the measured pairing).

### `rl/ppo.py`
- `_load_model`: V3 branch (same sniff/dims as matchrunner).
- `load_shards`/`collate_ppo`: carry `plans`; V3 forward gets the plan
  tensor in `ppo_update`.
- **KL-anchor**: `ppo_update(..., ref_model=None, kl_coef=0.0)` — per
  minibatch, ref logits no-grad, `kl = Σ p_new·(logp_new − logp_ref)` over
  the real options, `loss += kl_coef * kl`; logged as `kl`.
- `train()`: `--defect-penalty`, `--kl-coef` (ref = frozen copy of the
  start checkpoint), `--learn-deck`, `--tag` (namespaces ppo_best/ppo_it
  checkpoints + the TB run dir so M8-era artifacts aren't clobbered);
  per-iter defect-rate print + TB scalar. Plan head gets no plan-level
  policy-gradient term; it drifts only via the shared trunk — accepted,
  plan behavior monitored via eval.

### Tests (`tests/test_ppo.py` — extend or create)
- `_load_model` V3 sniff → right arch/dims.
- `collate_ppo` plans threading; V3 forward on a synthetic batch.
- Defect detector: saturated target fires, one-short doesn't (fake_cg pool).
- KL = 0 when ref == policy; > 0 after perturbation.

## Probe protocol (pre-registered)

- Start `checkpoints/osv3o_plan5.pt`, learn/eval deck `lucario`,
  400 games/iter, 8 iters/leg, eval every 2, lr 3e-5, dev shaping 0.05,
  entropy-coef 0.001.
- **Leg A:** `--defect-penalty 0.1 --kl-coef 0`.
- **Leg B:** `--defect-penalty 0.1 --kl-coef 0.1` (one tuning adjustment
  allowed if KL vanishes or dominates; diaried).
- Smoke first: 1 iter × 20 games (plumbing + defect counter).
- Verdicts: per-iter rollout defect/g (primary), entropy, vs_solver n=200;
  best iter per leg → **n=800 2-seed mirror + 40-game flag count**; meta
  2-seed only if mirror ≥ plan5 −1pp.
- **Kill rules:** K1 defect rate unmoved after 8 iters @0.1 → escalate to
  0.3 for 2 iters → still unmoved → dead. K2 defect fixed but both legs'
  n=800 mirror < plan5 −2pp → the trade survives PPO too → dead (A vs B
  isolates the anchor's contribution). K3 entropy rises >0.1 in-leg →
  stop that leg (M2 diffusion signature).
- Ship bar unchanged (2-seed pooled vs plan5 pins); ship call = Piotr.

## Process

Hermes telegram at every stage transition + reporter on >30min runs;
incremental diary in docs/M20.md (kills first-class); no commits until a
ship. Full pytest green before any run.
