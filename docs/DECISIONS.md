# Decisions & Observations Log

A running, append-only log of the non-obvious findings and the pivots they caused. Newest
first. Milestone diaries (`docs/M*.md`) have the full narrative; this is the fast index of
"what did we learn and what did we change because of it." Add an entry whenever a
measurement changes the plan.

---

### 2026-07-08 · Search (MCTS or 1-ply) doesn't beat the greedy policy even with a good critic → value head is a classifier, not a fine-grained ranker
**Observation:** with the good value head (0.87 sign-acc), MCTS was 53% vs greedy at n_sims=32
but got WORSE with more sims (64→40%, 128→45%) — the value head is out-of-distribution on
determinized/deep search states, so more search converges to a wrong answer. Shallow 1-ply
value lookahead was also worse than greedy (38%).
**Reading:** a 0.87 win/loss *classifier* isn't precise enough to *rank sibling actions*
(their value differences are small and swamped by determinization noise). The well-trained
BC policy head is a better action-ranker than value-based lookahead.
**Implication:** to make search help, the value net must be trained ON the states search
actually visits (AlphaGo-Zero style, fixes OOD) AND/OR determinization must be much better
(archetype-inferred, multiple samples). Otherwise the greedy BC policy is the best pilot we
have — and it's below the rule experts, so the DECK and a rule-agent baseline become the
highest-leverage levers. See docs/M2.md synthesis.

### 2026-07-08 · Supervised value-head training WORKS where PPO failed → the critic was trainable all along
**Context:** 3 PPO attempts all failed to improve on bc_v1 (stall at low lr, degrade at high
lr) — policy-gradient advantages are too noisy for this sparse-reward, mirror-heavy problem.
**Decisive experiment (decoupled critic):** froze bc_v1's policy+body, trained ONLY the value
head by supervised regression to self-play outcomes (11.5k states vs the opponent pool).
Result: bc_v1's original value head was **useless** (MSE 1.0, sign-acc 0.62 ≈ chance); the
trained head hit **MSE 0.35, sign-acc 0.87** — on the *same frozen features*. So the info was
there, PPO just couldn't extract it; direct supervision nails it.
**MCTS with the good critic:** 47% → **53%** vs greedy bc_v1 (correct determinization) — search
*finally beats its own policy*, though modestly (likely sim-count-limited at n_sims=32).
**Pivot:** stop training the critic with PPO. Train the value head **supervised on self-play
outcomes** (stable, fast, effective) — an AlphaGo-Zero-lite recipe: self-play → outcome
targets → supervised value head → MCTS with enough sims. (`checkpoints/bc_v1_value.pt`,
`data/value_train.npz`) Sim-count sweep + archetype-inferred determinization are the next levers.

### 2026-07-08 · Opponent IS observable AND already well-encoded → value head problem is TRAINING, not representation
**Observation (Piotr):** we don't see the opponent's hand/deck/prizes, but we *do* see
their active/bench Pokémon (+ attached energy/tools) and their full discard pile — verified
in-engine. So the archetype is readable by mid-game.
**First (wrong) reaction:** I'd blamed PPO's flat `value_loss` on the outcome being
"unpredictable from a hidden opponent," and proposed a `revealed_opponent` identity feature
(per-card-id counts). Built it, STATE_DIM 1206→2474.
**Cheap validation BEFORE retraining (saved a wasted run):** a linear probe on mid-game
states predicting *which opponent* (mirror/Lucario/Iono) — the existing **pooled** features
already hit **98%**, the new 1268-dim feature 92% (it overfits). So the archetype signal was
**never missing**; it's already in the encoder. **Reverted the feature** (redundant, adds
overfitting).
**Conclusion:** the value head can already see the opponent — so its failure to learn is a
**training** problem (weak advantage signal, entropy diffusion, few iters, small value head),
not a representation gap. Fix training (entropy 0.01→0.001 already done; more iters / higher
lr / bigger or separately-trained value head) or move to Path B (AlphaGo-Zero value targets).
**Meta-lesson:** validate a feature's premise with a cheap probe before paying for a retrain.

### 2026-07-08 · Inference-time MCTS on a BC/PPO net does NOT beat the policy → value head is the blocker
**Observation:** MCTS(bc_v1) = 47% vs its own greedy policy (92% vs random, so no gross bug);
MCTS(ppo_best) = 27% (worse). PPO's `value_loss` stayed flat ~0.30 for 25 iters; entropy
*rose* 0.82→1.0.
**Reading:** the value head (a 0.5-weight BC auxiliary) is too weak; shallow search amplifies
its errors. Entropy bonus (0.01) overpowered the weak advantage signal and diffused the
policy. This is the AlphaGo Zero lesson — you must TRAIN on MCTS/critic-quality targets, not
just run MCTS at inference.
**Pivot:** `ENTROPY_COEF` 0.01→0.001; fix the value head (via the representation change above
and/or Path B AlphaGo-Zero training) before expecting search to help.

### 2026-07-08 · Behavior cloning can't imitate hidden-state reasoning → BC has a hard ceiling
**Observation:** cloning the *stronger* Lucario expert gave a *weaker* agent (bc_lucario 38%
vs our bc_v1) at 55% imitation, vs 90% for the simpler Kyogre generic pilot.
**Reading:** the Lucario expert's edge is lookahead (`AttackPlan`) that isn't in the
observation. **agent strength ≈ teacher strength × imitation fidelity**, and fidelity
collapses on unobservable reasoning.
**Pivot:** stop trying to BC our way past the experts; pursue search (MCTS) / RL to acquire
lookahead the teacher can't demonstrate.

### 2026-07-08 · Deck quality dominates; our base (Kyogre) is weak
**Observation:** slot-fair round-robin — Lucario ≫ Iono ≫ Kyogre(bc_v1). Disentangled: pilot
is fine (bc_v1 ties the generic rule pilot on Kyogre 51%), the *deck* is the problem
(Kyogre 30% vs Lucario under the same brain).
**Pivot:** deck is a first-class training target, not a fixed constant. Built
`decks/{kyogre,lucario,iono}.csv`, extracted the Iono rule agent, parametrized
`load_teacher(agent=, deck=)`. Deck search remains a milestone (M3).

### 2026-07-08 · Player-0 has a large first-move advantage → all head-to-heads must slot-swap
**Observation:** teacher-vs-teacher mirror = 61% for player 0 without swapping, 47% with.
**Pivot:** `rl/eval.play_games` swaps slots by default; the M1 "70% vs teacher" was inflated
by this bias.

### 2026-07-08 · Kaggle runtime provides numpy but NOT the `cg` engine
**Observation:** first submission died `ModuleNotFoundError: No module named 'cg'`.
**Pivot:** bundle `cg/` inside the submission tarball; the build gate verifies archive
contents. (docs/M0.md lesson 8)

### 2026-07-07 · An untrained net is an arbitrary *consistent* player, not a random one
**Observation:** M0 random-weights agent lost ~100% to random; it took the ATTACH option 2
of 1439 times — a fixed, systematically bad policy.
**Pivot:** motivates exploration (sampling at collection, entropy bonus) so a deterministic
policy can discover the moves it irrationally avoids.
