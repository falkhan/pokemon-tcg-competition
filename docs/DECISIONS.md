# Decisions & Observations Log

A running, append-only log of the non-obvious findings and the pivots they caused. Newest
first. Milestone diaries (`docs/M*.md`) have the full narrative; this is the fast index of
"what did we learn and what did we change because of it." Add an entry whenever a
measurement changes the plan.

---

### 2026-07-19 · Dragapult ex company sample agent ≈ Lucario-expert strength → third archetype package added
**Observation:** the Pokemon-company "Dragapult ex" rule agent (repo-root
`a-sample-rule-based-agent-dragapult-ex-deck.ipynb`, "Advanced Level" — Phantom Dive
multi-KO planning with card counting and turn-log inference) integrated as
`sample-agent-dragapult/` + `decks/dragapult.csv`, same extraction as Iono (M2).
Slot-fair series, n=200 × 2 seeds, 0 pilot errors in 1200 games:
vs `rule:iono` **0.657** pooled; vs `rule:lucario` **0.495**; vs `rule:tuned` 0.477 —
parity with our strongest teacher, far above Iono.
**Pivot:** registered as teacher + deck in `rl/teacher.py` / `tcg/teachers.py` and as a
`dragapult_expert` league anchor. NOT yet added to the `rl/collector.py` opponent pool or
any collection mix — changing the training distribution is a milestone-level decision
(M7.5 "you become what you train against"). Candidate levers for the next milestone:
dragapult as expert teacher for plan_iter collection, and the dragapult deck itself
(deck surgery remains the top strength lever).

### 2026-07-11 · The turn solver is a WRAPPER around the pilot, not a scorer hook — and boost items are detected empirically, not classified
**Decisions (M7.4a):** (1) `rl/turn_solver.py` wraps `make_generic_pilot` (the rl/hybrid.py
shape) instead of hooking `score_option`: the shipping pilot and its `tcg/pilot.py` twin stay
byte-identical under the >400-scenario parity pins, and the solver follows the M7 rl/-only,
no-twin rule. Submission wiring waits for the §7 A/B gate (M7.5 flip is two lines; a new
import-tier test keeps the module bundle-pure meanwhile). (2) No effect text exists anywhere
in the repo, so "is this trainer a damage boost?" is answered by PLAYING it in the forward
model and observing the prize delta — the search subsumes classification. (3) Within my own
turn the opponent never acts, so the search needs NO opponent determinization, and my own
draw reveals are handled by recomputing at every prompt instead of caching a plan. (4) The
milestone's "curated combo suite from forensics" cannot exist offline (forensics emits
aggregate W/L tables, not positions, and is [NET]-gated) — substituted with hand-authored
scripted-tree fixtures in `tests/test_turn_solver.py`; a real forensics-derived suite stays
deferred behind M7.0 [NET].

### 2026-07-10 · The PPO loss bug was real — every "PPO failed" conclusion was trained on a corrupted objective
**Confirmed and fixed (M7.4b):** `rl/ppo.py` and `tcg/ppo.py` both computed
`loss = policy_loss * VALUE_COEF * value_loss − ENTROPY_COEF * entropy` — MULTIPLYING the
policy loss by the value error where the docstring (and PPO) say ADD. Consequences: the policy
gradient was scaled by an arbitrary positive factor that shrank exactly as the critic improved,
and flipped sign whenever policy_loss and value_loss disagreed in sign — a corrupted objective,
not a hard optimization problem. All three stalled PPO attempts (M2–M4, DECISIONS 2026-07-08)
trained on this form, so "PPO doesn't work here" was never honestly measured. Fixed to the
additive objective in both twins; `tests/test_ppo.py::test_loss_is_the_additive_ppo_objective`
pins the corrected form literally (the old cross-module pin only guaranteed the twins matched).
The M7.4b [ENGINE] retry (fixed loss + critic warm-start + multi-deck self-play + generic-pilot
opponents) is the first honest PPO measurement this project will have.

### 2026-07-09 · Measured: prioritizing the bench closer above attach-active STARVES the active — race charging must be gated on an attack-ready active
**Measurement (M7.2b gates, real engine):** vs-expert 0.329 (PASS, up from ~0.25–0.30) but
floor 0.715 (FAIL, need ≥0.90; 57/200 losses, 0 draws — all self-deck-outs) and vs-random
0.795. **Root cause:** `SCORE_ATTACH_RACE_CLOSER_BENCH` (2680) > attach-active (2600) sent
*every* energy to the benched closer; the active (Riolu: attack needs 1 energy, retreat 2)
could then neither attack nor retreat, so no ATTACK option existed for close mode to boost,
trainers fired every turn, and the pilot milled itself — the exact M6 failure the change was
meant to fix, resurrected by the fix. **Lesson:** "keep charging THE ONE attacker" is only
safe once the active is functional; the bench tier now requires `turns_to_ready(active) == 0`
(feed the active first, then bank the closer). The vs-expert gain came from the other race
terms (promote, best-attack loading), which survive unchanged.

### 2026-07-09 · The expert's AttackPlan is a SINGLE-TURN commitment; race math ports it deck-agnostically, with a conservative close mode
**Discovery (M7.2b):** sample-agent/main.py's `AttackPlan` — the experts' documented multi-turn
edge — is actually a per-turn, single-target attack commitment (reset every turn, commits only
when ≤1 attach from firing). The *multi-turn* behavior emerges from re-committing each turn plus
`energy_score` banking energy on under-charged attackers. So L1 (rl/combat.py + both pilots)
ports a **race table** (`charged_best` / `turns_to_ready` / `hits_to_ko` / `turns_to_first_ko =
max(gap,1)+hits−1`), not a plan object: attach bonuses my fastest closer (bench closer outranks
even the KO tier — attaching doesn't end the turn), "loaded" now means the BEST attack charged
(the old cheapest-attack check stopped charging Mega Lucario after 1 of 2 energies), promote
subtracts attaches-still-needed. **Decision:** close mode v1 (chip attacks jump above trainers,
the floor-test self-deck fix) triggers only when the opponent board is *harmless* (all known ids,
zero printed damage) — deliberately conservative so the >30% vs-expert gate can't regress through
it; the full prize-race comparison is deferred until the [ENGINE] gates are measured.

### 2026-07-09 · Evolution chains link by NAME, and attackId maps to cards by cumulative n_attacks
**Two data discoveries while building the deck factory (rl/deck_build.py, M7.1):**
(a) `evolves_from_id` points at one specific printing, but decks legally play any same-named
card — the tuned Lucario deck runs Riolu #677 (80 HP) while Mega Lucario ex's id points at
Riolu #974 (70 HP). Naive id-walking misses real lines; walk by NAME, take the best printing.
(b) `attacks_features.parquet` has no card_id column, but attack ids are consecutive in
card_id order — cumulative `n_attacks` assigns all 1,556 attacks with zero max_damage
mismatches. That unlocks true damage-per-energy (best attack's cost) engine-free;
`min_attack_cost` alone is the *cheapest* attack and mis-ranks attackers (Mega Lucario:
min cost 1 = 130 dmg Aura Jab, best = 270 dmg Mega Brave at cost 2). Both facts are pinned
in tests/test_deck_build.py. **Consequence:** the whole M7.1 factory (lines → scoring →
templates) runs without the engine; only the fitness gate is [ENGINE].

### 2026-07-09 · Kaggle ingestion built offline-first; the schema is an assumption until the [NET] spike
**Built (rl/kaggle_ingest.py, M7.0):** injectable fetch layer (immutable gzip cache, ~1 req/s
throttle) over `EpisodeService/{ListEpisodes,GetEpisodeReplay}`, a schema-tolerant
`parse_episode`, deck harvesting with exact-hash + pooled-FEAT-cosine archetype clustering,
frozen `meta_v<N>` snapshots, per-archetype forensics, and a BC-on-winners audit. 21 offline
tests on canned env.toJSON()-shaped fixtures; full suite green. **Decision:** fetch and parse
were split so the sandbox's lack of kaggle.com access blocks nothing except the final
verification — the three schema bets (deck at first action, endpoint reachable, opponent obs
present) are tracked in docs/M7.md with a runbook; `python -m rl.kaggle_ingest verify` is the
go/no-go command. Hard asserts live in `verify`, not the parser, so `refresh` stays resumable
over a mixed-quality episode set.

### 2026-07-08 · Deck search: robust + honest, but the tuned deck can't be improved against a small field
**Built (rl/deck_search.py):** matchup evaluator, openskill rating, `evolve()`, `hill_climb()`,
`mutate_flex()`. **Findings:** (a) noisy population openskill *regressed* (picked a deck that lost
to the seed 44%) — use large-sample hill-climbing instead; (b) mirror-only hill-climbing overfits
(evolved deck 60% vs seed mirror but 79% vs Iono < seed's 82%) — same mirror-overfit as PPO;
(c) with a *diverse fixed field* (Iono + bc_v1), seed fitness is already 0.84 and **0/30** mutations
improve it. **Conclusion:** the Lucario deck is well-tuned; deck search is bottlenecked by opponent
diversity (2 rule pilots + bc_v1 ≠ the real 4500-team meta), not by method.
**Comprehensive (M1–M4):** the provided rule agents + tuned decks beat everything we can build
(bc_v1 640 vs ~1100+ top; rule Lucario beats bc_v1 ~86%) OR improve. Leaderboard climb via
from-scratch neural/deck-search on a CPU budget is not panning out. Honest strategy options:
(a) engineer/extend a rule agent (manual heuristic + deck work — legit, likely what top teams do),
(b) submit bc_v1 as our trained agent (640, real but mid-pack), (c) accept the learning outcome
and stop. The RL/NN learning goal is richly met; the leaderboard-win goal hits the reality that
domain-engineered agents dominate here.

### 2026-07-08 · Combat-lookahead features don't rescue Lucario BC → the wall is comprehensive
**Hypothesis:** the experts' edge is *computable* combat lookahead (damage/KO/prize); exposing
it as features would lift BC fidelity past its 55% ceiling and yield a strong Lucario pilot.
**Result (docs/M3.md):** fidelity 55%→56.4% (flat); win rate vs bc_v1 champion **25%** (worse
than the 38% *without* combat features). Refuted on both metrics. State-level aggregate
features don't drive the expert's fine decisions, and ~56% is near the structural aliased-option
ceiling.
**Comprehensive finding:** across BC (Kyogre/Lucario ± combat features), PPO, MCTS, and
supervised-value+search, **nothing beats bc_v1+Kyogre**, and it loses to the rule experts. The
rule experts encode multi-turn planning that is very hard to learn or search on a CPU budget.
**Next is a STRATEGY decision, not a tweak** — options: (a) engineer an improved rule-based
agent (Piotr rejects verbatim reuse, but tuning/deck-building is fair), (b) full AlphaGo-Zero
(train value on search-distribution states + MCTS policy targets — big, low-confidence given
MCTS barely beats the policy), (c) accept bc_v1 (90% vs random) and pivot effort to deck search,
(d) reassess scope. Learning goal (RL/NN mastery) is richly met regardless.

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
