# M21 — Encoder v4 (complete observable state) + PPO with annealed KL and opponent-mixture curriculum

## Context

Strategic review answered three questions:

1. **Is the state representation sufficient?** No. The v3 encoder ignores the entire `obs.logs[]` event stream (opponent plays/attaches/draws — the observable proxy for their hidden hand and next attacker), plus face-up prize ids, `appearThisTurn`/`preEvolution`, special-energy identity, select-prompt constraints (`minCount`/`maxCount`/`remainDamageCounter`/`remainEnergyCost`/`contextCard`/`effect`), and several global flags.
2. **Can the agent learn card play by exploration (prize + win reward) instead of hardcoded rules?** The shipped net already scores engine-legal actions directly — nothing is rule-gated live. The PPO path already uses exactly the proposed reward (0.1×prize-delta shaping + terminal ±1, `rl/collector.py:256-292`). What caps behavior is the *teacher*: BC/DAgger imitates a frozen heuristic+search pilot, and the fixed KL anchor regularizes toward its blind spots (flat `SCORE_EVOLVE`, no setup search). **Decision (user-confirmed): warm-start from the champion, DAgger only to re-baseline new features, then PPO legs with the KL anchor progressively annealed toward zero** — initialization bias washes out, anchors don't. From-scratch rejected (M8.3 evidence, sparse reward, compute budget).
3. **Train against different decks?** Infrastructure exists (`OpponentSpec` vocabulary, `collect(pool=(specs,weights))` at `rl/collector.py:337-350` — currently never passed by `rl/ppo.py`). **Decision (user-confirmed): interleaved weighted mixture with replay, not sequential 10k blocks** (M7.5 "you become what you train against" precedent, `rl/collector.py:57-61`).

**Scope decision (user-confirmed): split.** M21 = encoder v4 + DAgger re-baseline + PPO legs with curriculum mechanism. M22 = deferred items (below). Campaign bar stays 0.55 vs `solver:lucario` (n≥800 pooled); champion pin = `ppo_best_m20legB.pt` at 0.488; M21 realistic target 0.50–0.53.

**Ownership update (user directive):** Piotr steps back to orchestrator; Claude owns the RL core now, including the GAE/PPO math in `rl/ppo.py`. This unblocks plan-head-in-PPO (below). Update/delete the `piotr-owns-rl-core` memory at implementation start.

**Replay feedback driving this revision (latest submission):** the agent (a) never plays Boss's Orders — missed a game-winning gust onto a benched, energy-less, one-shot-able Mega Lucario Ex; (b) failed to retreat a damaged active Lucario ex and promote a 3-energy full-HP Lucario ex from bench. Root-cause hypothesis: gust targeting lives in the *plan head* (bench-target plans via `GUST_IDS`, `rl/plan.py:24,120-165`) and the plan head is **not trained by PPO** — greedy plan selection at collection (`rl/collector.py:172-180`), no plan logprob in the update. Option-level exploration alone can never fix this. Bench awareness features exist (12 slots × 87 incl. per-slot HP/energy, M19 `_retreat_extra` bench-ready) — so (b) is a policy failure, not blindness; confirm in forensics.

## Part 1 — Encoder v4

Design principles: **append-only** (v3 numeric block + 64 context one-hot stay byte-identical; v4 block appended after, new id slots at tail), **pooled not per-id** (the reverted revealed-cards lesson, `rl/encoders.py:47-49`), **twin doctrine** (`rl/encoders.py` AND `tcg/encoders.py`), **explicit versioning** via `register_buffer("enc_ver", tensor(4.0))` (rides into npz via `save_npz` state_dict iteration, `tcg/network.py:203`) — width-sniffing becomes ambiguous at v4; legacy sniff branches untouched.

### 0. FORENSICS FIRST (hard rule + user's replay observations)

Before any code: pull latest Kaggle replays of `ppo_best_m20legB` (54836093) and quantify:
- **Gust usage**: count Boss's Orders in hand vs played; enumerate missed kill-shot gust opportunities (opp benched Pokemon with 0 energy, one-shot-able, ≥ prize swing). Establish a baseline rate (expected ≈0 per user's observation).
- **Retreat/promote misses**: decisions where a bench slot had strictly better (energy-ready, higher-HP) attacker than a damaged active and no retreat happened.
- **Encoding awareness check**: replay the missed-retreat state through `encode_state_v3` and confirm bench HP/energy/ready features are present and correct at that decision (rules out a feature bug before blaming the policy).

These two counters (gust-opportunity conversion rate, promote-strongest rate) become tracked behavioral metrics in every leg's battery — diagnostics, not rewards.

### 1.0 NEXT — logs-delta semantics probe (~30 min)

`Observation.logs` = "events since last selection" (`cg/api.py:441`) but it is unverified whether the delta is per-player or global in the direct engine loop. If global, a per-seat consumer silently misses events → train/serve mismatch (worst failure mode). Probe: scripted direct-loop game dumping log lengths/types per prompt per seat; cross-check one `kaggle_environments` game.
- Per-player → design as-is.
- Global → fallback 1: training loops observe at every prompt for both seats (they see all prompts); fallback 2: degrade to state-diffing between own prompts (loses attack ids, keeps attach-target/appear/evolve signals).

### 1.1 New module `rl/memory.py` (bundle-pure: numpy + cg.api only)

`OppMemory` with `reset()`, `observe(obs)` (consume logs once per own prompt), `features(state) -> (48,)`, `ids() -> (5,)`:
- Last opp ATTACK: attacker id + attack features (reuse `_attack_extra` math, `rl/encoders.py:349`)
- Opp attacked/passed-last-turn flags; attack count
- **Opp last energy-attach target as 7-way one-hot** (their announced next attacker — highest-value bit in the gap list), resolved via `serialTarget` vs board serials
- Opp known-hand pool: serial→card_id dict from revealed MOVE_CARD-to-hand events, removed on play/attach/evolve → pooled FEAT(36) + count
- Extra-draw intensity; last-4 opp played ids + last attacker id = 5 id slots

### 1.2 Pure-state extras in `encode_ctx_v4` (both encoder twins)

- Per-slot extras 12×5=60: `appearThisTurn`, `len(preEvolution)/2`, special-energy count, **`can_attack_now` flag, best-affordable-attack damage /300** (reuse `rl/combat.py` affordability math) — makes bench strength/readiness maximally salient per-slot for the retreat/promote failure
- Special-energy identity pools (2×36=72); face-up prize pools + counts (74)
- Global scalars (8): opp deckCount, benchMax both sides, stadiumPlayed, retreated, turnActionCount, firstPlayer, my handCount
- Select-prompt extras (76): min/maxCount, remainDamageCounter, remainEnergyCost + FEAT of `effect` + FEAT of `contextCard`

**V4_EXTRA_DIM = 338; N_STATE_IDS_V4 = 25. Network input 1708 → 2126 (+24%).** Pre-registered tier-2 cut line if Phase A fails: drop prize pools + contextCard + special-energy pools (−180 dims).

### 1.3 Network + migration (change BOTH `rl/policy.py:143` and `tcg/network.py:146`)

- `OptionScorerV3(extra_dim=0)` param; input width += extra_dim; `enc_ver` buffer when extra_dim>0. Reuse the class → key names stay test-pinned, `save_npz` and `submission/main.py` `_linear` replay unchanged.
- `migrate_v3_to_v4(sd_v3)` in `rl/plan_iter.py` next to `load_v3_into_v3h` (`rl/plan_iter.py:663`, 4th use of the pattern): old `state_enc.0.weight` columns copied (plan/id columns shifted right), v4 + new-embedding columns **zero-init** → migrated net ≡ champion at start (parity test: garbage v4 inputs × zero weights = identical logits).

### 1.4 Wiring (complete call-site list)

| Site | Change |
|---|---|
| `rl/encoders.py` + `tcg/encoders.py` | v4 block, `encode_ctx_v4` |
| `rl/memory.py` (new) | `OppMemory` |
| `rl/policy.py` + `tcg/network.py` | `extra_dim`, `enc_ver` |
| `rl/collector.py:150-217` | v4 sniff, learner-seat `OppMemory`, observe-before-encode, per-game reset (next to `pstate` at `:240`) |
| `rl/matchrunner.py:329-390` | v4 model-pilot branch; memory in pstate closure, reset pattern `:360-365` |
| `rl/plan_iter.py` | encoder dispatch on `enc_ver`; two per-seat memories in self mode; `collate_v3` width fix (read from batch, not hardcoded); `BCDatasetV3` zero-pad shim so old shards mix in |
| `rl/ppo.py:265-283` | `_load_model` v4 sniff branch only |
| `submission/main.py:51-66,133-175` | v4 sniff; module-level `_MEM = OppMemory()` with the `_PSTATE` reset precedent (select-None + turn-drop); observe once per call |
| `tcg/shipping.py:81,155,237` + `build_submission.sh` | bundle `rl/memory.py`; extend `encoder_parity_check` to drive memory through a 40-step game (train-vs-bundle parity incl. accumulated memory) |

### 1.5 Tests

Migration parity; memory unit tests on synthetic log sequences; encoder goldens from a recorded game; memory-aware parity check; version-sniff table test (v1/v2/v3-12id/v3-20id/v4 all load to the right class). Full suite (473 baseline) green before any training.

## Part 2 — Training recipe

**DAgger's role is now migration-only** (Phase A). It never mixes into PPO legs — that would be a second anchor toward teacher blind spots.

### Phase A — EI re-baseline on v4 (populate zero-init columns with dense gradients)

1. `migrate_v3_to_v4(ppo_best_m20legB.pt)` → `osv4_seed.pt`; sanity n=200 vs `model:ppo_best_m20legB:lucario` ≈ 0.50.
2. EI collection ~4,000 games, 12 workers, `--decks data/league/population.json`, opponents `self solver:lucario solver:<meta-lucario csv>`; overnight (~6–8 h; calibrate from the 25-game projection line).
3. Train `--init osv4_seed.pt`, mixing new v4 shards + recent v3 shards (pad shim).
4. **Gate A**: screen n=400 ≥0.46 → confirm n=800 2-seed pooled ≥0.47 vs `solver:lucario`; meta co-gate ≥0.48; floors (random:kyogre ≥0.90 n=200×2, rule:lucario ≥0.34). **Kill**: two attempts <0.46 → tier-2 cut, retry once; else re-scope, champion stands.

### Phase B — PPO legs (M20 shape; ~2–2.5 h each + ~3 h battery)

Common: 8 iters × 400 games, lr 3e-5, `--shaping dev --race-shaping 0.05`, reward unchanged, **defect-penalty 0** (M20 measured the anchor neutralizing it), `--eval-every 2 --eval-games 200` (promotion triggers only, never evidence), re-anchor each leg's `ref_model` at the previous leg's verified best.

**NEW CORE WORK — plan-head in PPO** (unblocked by ownership change; the direct lever for the gust failure):
- Collector: sample the turn plan at `--plan-tau` (existing `act_plan(tau, dirichlet_eps)`, `rl/policy.py:224`) and **record the plan decision as a PPO row** — plan logprob, plan candidate set (re-encodable plan matrix), value at the plan prompt.
- `rl/ppo.py`: extend `ppo_update` with a plan-policy term — same clipped-surrogate form over plan logits, joint with the option loss (shared trunk). GAE treats the plan decision as one more step in the trajectory (it already is a decision point; the reward stream is unchanged). Add `--plan-ppo` flag to enable, keep the anti-aliasing invariant note in mind: in PPO (unlike EI) the sampled plan IS the executed plan, so plan-as-input and plan-as-action are consistent by construction.
- Tests: gradient flows to `plan_head` under `--plan-ppo`; logprob/ratio parity on a recorded batch; off-flag behavior byte-identical to M20 path.

**KL/exploration schedule across legs — the anchor is scaffolding, removed progressively. Exploration is deliberately hotter than M20 (user directive: the agent must try combinations — gust plans, retreats — it currently never samples):**
- **Leg B1 "stabilize v4 under RL"**: KL 0.1, `--plan-tau 0.3` (mild — enough to sample gust/retreat plans, small enough to isolate the encoder change), lucario-centric mix. Gate: ≥ Phase A; target ≥0.50. Track gust-conversion + promote-strongest metrics.
- **Leg B2 "meta curriculum + plan-PPO"**: KL 0.05 (re-anchored at B1 best), meta mixture, `--plan-ppo` ON, `--plan-tau 0.7 --plan-dirichlet 0.15`, entropy anneal 0.003→0.0005 (higher start than M20's 0.001 — deliberate). Gate: mirror ≥ B1−1pp AND meta ≥0.52 AND gust-conversion rate strictly improved from forensic baseline.
- **Leg B3 (if B1+B2 promote)**: 10–12 iters × 600 games, `--kl-anneal-to 0` (linear within-leg 0.05→0) — by the end this is pure self-play PPO on prize+win reward, the user's Q2 realized. Watch rules: entropy *rising* = M2 diffusion = kill; ratio blowup = kill; verify < previous −2pp = kill leg, keep prior checkpoint.

## Part 3 — Opponent curriculum mechanism

Thin layer over existing machinery — `collect()` already accepts `pool=(specs, weights)` (`rl/collector.py:337,350`); `rl/ppo.py` just never passes it.

1. `rl/collector.py`: `parse_pool(items, checkpoint, learn_deck)` parsing `"spec=weight"` via `matchrunner.parse_spec`; special tokens `mirror=w` (current ckpt) and `past=w` (recent promoted selves via `default_pool` tail logic `:70-74`). CLI `--opponents`.
2. `rl/ppo.py train()`: `opponents` list + optional `--opponent-schedule` JSON (`[{"until_iter": 4, "opponents": [...]}]`); resolve per iteration, pass `pool=`; log drawn mix per iteration.
3. Meta decks by csv path work today (`resolve_deck`, `rl/matchrunner.py:55-66`) — e.g. `solver:data/kaggle/meta_v2/<deck>.csv=0.30`.

Mixes:
- **B1**: `mirror=0.20 solver:lucario=0.30 solver2:lucario=0.10 rule:lucario=0.10 generic:lucario=0.10 solver:iono=0.10 past=0.05 random:kyogre=0.05`
- **B2** (dampened meta weights so minority archetypes appear): `solver:<meta lucario+solrock>=0.30 solver:<kangaskhan>=0.12 solver:<drakloak>=0.08 solver:<zacian>=0.05 solver:lucario=0.15 mirror=0.10 past=0.10 rule:lucario=0.05 random:kyogre=0.05`

Forgetting check: per-leg battery adds n=200 spot-check vs previous leg's majority opponent; drop >3pp while focus rises = forgetting → replay weight 0.25, rerun leg.

## Part 4 — Verification (measure-agent conventions)

| Checkpoint | Gate | Kill |
|---|---|---|
| Phase 0 | full suite + new tests green; logs probe decided | unresolvable logs semantics → state-diff memory tier |
| `osv4_seed` | ≈0.50 vs champion model (n=200) | any deviation = migration bug, stop |
| Phase A | ≥0.47 n=800 2-seed + meta ≥0.48 + floors | 2×<0.46 → tier-2 cut → re-scope |
| Leg B1 | ≥ Phase A; floors | entropy rise / ratio blowup / −2pp |
| Leg B2 | mirror ≥ B1−1pp AND meta ≥0.52; floors + B1 spot-check | meta flat AND mirror down → weights revisit once, else stop at B1 |
| **Ship** | mirror pooled >0.50 (beats 0.488 pin outside noise) + meta ≥ champion−1pp + floors | nothing beats champion → no ship; v4 infrastructure still lands |

Ship via `./build_submission.sh --checkpoint <best> --deck lucario` + M18.1 deck-md5 ritual; ship call is Piotr's. Hermes Telegram updates at every stage transition + hourly heartbeat on runs >30 min + immediate on any failure/kill; diary observations incrementally in docs/M21.md.

## Risks

1. **Train/serve logs mismatch** (top): probe-first + fallbacks + memory-aware parity check.
2. **Feature blowup**: zero-init migration (start = champion), old-shard regularization, tier-2 cut, Gate A non-inferiority kill.
3. **PPO instability on fresh features**: Phase A trains columns first; B1 changes one variable; anchor bounds drift.
4. **Curriculum forgetting**: mixture + replay + gate-opponent floor weight + spot-checks.
5. **KL→0 degenerate self-play equilibria**: mixture (never pure mirror) + evidence-grade gates only.
6. **Sniff regressions**: `enc_ver` buffer, legacy branches untouched, sniff-table test.
7. **Plan-head-PPO instability** (new math, new risk): joint plan+option surrogate could destabilize the proven option-only update. Mitigation: `--plan-ppo` is a flag, B1 runs without it (encoder isolated first), B2 enables it re-anchored at a verified checkpoint, off-flag path byte-identical to M20 (tested), kill rules unchanged.

## Deferred to M22 (priority order)

(1) opponent deck/archetype inference (determinizer-derived priors; needs `rl/determinize.py` bundle-pure); (2) league-72-deck curriculum breadth + longer anchor chains; (3) deck surgery (top lever since M19, orthogonal); (4) tier-2 blocks if cut; (5) hidden-width growth if v4 saturates. (Plan-head-in-PPO moved INTO M21 — it is the direct fix for the never-gusts failure.)

## Standard milestone requirements (hard rules)

- **Plan doc**: at implementation start, store this plan as `docs/M21-plan.md`; describe M21 in `docs/MILESTONES.md`.
- **Diary**: record observations in the M21 diary (`docs/M21.md`) **incrementally, the moment they are produced** — forensic baselines, logs-probe outcome, collection stats, train metrics, every gate result including kills, anomalies. Never retrospectively. Kill results are as valuable as passes.
- **Telegram (Hermes)**: `hermes send -t telegram` at every pipeline stage transition (forensics done, probe outcome, Phase A collect start/end, Gate A result, each leg start/end, each verify result, ship); hourly heartbeat attached to any run longer than ~30 min (Phase A collection, PPO legs, batteries); immediate notification on any failure or kill-gate.
- **Commits**: commit only the milestone that ships to Kaggle; never commit `.env`/credentials; tests/lint green before any change is called done.

## Step ordering

0. Store plan as `docs/M21-plan.md` + MILESTONES.md entry + forensic replay analysis (gust/promote baselines + encoding-awareness check) + update `piotr-owns-rl-core` memory → 1. Logs probe → 2. `rl/memory.py` + tests → 3. encoder twins + goldens → 4. network/migration + parity → 5. wiring (collector → matchrunner → plan_iter → ppo → submission → shipping/build) → 6. suite green → 7. migrate + sanity → 8. Phase A (overnight) + Gate A → 9. curriculum/exploration flags + plan-head-PPO implementation + tests → 10. B1 → verify → 11. B2 → verify → 12. optional B3 → 13. ship decision + docs/M21.md diary wrap + pins.
