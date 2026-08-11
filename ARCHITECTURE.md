# RL Agent Architecture — Pokémon TCG AI Battle (Kaggle `cabt`)

How the agent is actually built, as of **M20 (2026-07-19)**.

> **Living doc.** This file describes the system as it *is*. What we tried, measured, and
> changed our minds about lives in the milestone diaries (`docs/M0.md` … `docs/M20.md`) and,
> in fast-index form, `docs/DECISIONS.md`. When a measurement contradicts this file, add a
> DECISIONS entry and update the relevant section here.
>
> **This document was rewritten at M20.** The original (M0-era, last touched at M7 planning)
> described a design we have since measured our way out of — see §2 for what changed and why.

---

## 1. Where the project stands

| | |
|---|---|
| **Champion** | `checkpoints/ppo_best_m20legB.pt` + `decks/lucario.csv` |
| **Live submission** | Kaggle **54836093** (M20, 2026-07-19) |
| **Mirror strength** | **0.488** pooled n=800 vs `solver:lucario` — the campaign's best neural result |
| **Meta co-gate** | ~0.494 (parity with the previous champion) |
| **Campaign bar** | ≥0.55 vs `solver:lucario` — **still unmet, 6.2pp short** |
| **Best settled live score** | M16 at 519.8, vs the pure-rules bundle's **548.6** |

The neural track has not yet beaten our own rules-based submission on the live leaderboard.
That is the central open problem.

**In flight — M21** (`docs/M21-plan.md`, branch `feature/m21`), which will change §5 and §7.4:

- **Encoder v4** — the v3 state ignores the entire `obs.logs[]` event stream (the observable
  proxy for the opponent's hidden hand), face-up prize ids, special-energy identity, and the
  select-prompt constraints. v4 is **append-only** (the v3 numeric block and context one-hot
  stay byte-identical; new slots go at the tail) and adds explicit versioning via an
  `enc_ver` buffer, because width-sniffing becomes ambiguous at v4.
- **Plan head in PPO.** The plan head has never received a PPO gradient, and plans are picked
  greedily at collection — so gust plans are never sampled. This is the standing hypothesis
  for why the shipped agent *never plays Boss's Orders*, including a game-winning gust.
- **Annealed KL** (→ 0 across legs) and an **interleaved opponent-mixture curriculum**, using
  the `collect(pool=...)` path. (Landed: `rl/ppo.py` has passed `pool=` from
  `--opponents`/`--opponent-schedule` since M21.)

M21 target was 0.50–0.53; the bar stays 0.55.

---

## 2. What changed from the original design

The M0 design proposed three learnable components — a play policy, a value function, and a
deck policy — trained by self-play PPO from a behavior-cloned warm start. Reality after 20
milestones:

| Original plan | What actually happened |
|---|---|
| **Play policy** — option-scoring net | ✅ Survived, and is now the whole agent. Grew from a 511-feature MLP to a 892k-param embedding network (`OptionScorerV3`). |
| **Value function** — PPO critic + progress metric | ⚖️ Split in two. A critic head still rides along for PPO, but the *useful* value net (`rl/setup_value.py`) is consumed **only at data-generation time** as a search leaf. Every attempt to consume a value signal live was killed (§15). |
| **Deck policy** — mutation bandit, then a learned generator | ⚖️ Built (`rl/deck_build.py`, `rl/deck_search.py`, `decks/gen/`) and then **deferred**. Deck quality is the biggest measured lever and the largest untouched one. |
| **Self-play PPO as the engine of improvement** | ❌ then ✅. PPO stalled M2–M4 (on a corrupted loss — see DECISIONS 2026-07-10), was ruled dead at M8.3, and was **revived at M20** on the V3 encoder where it produced the champion. |
| **BC warm start from the rule agent** | ❌ Superseded. Imitating the rule expert caps at ~55% fidelity because its reasoning is unobservable (§14). The teacher is now a **search process we control**, not a fixed agent. |
| **Deck/play learning kept separate** | ✅ Still true, and now enforced: a policy and its deck are **one artifact** (§11). |

The single largest architectural shift: **the improvement operator moved from "self-play RL"
to "a widened within-turn search that labels states"** (§7). Training is mostly supervised;
PPO is a fine-tuning stage on top.

---

## 3. The environment contract

From `cg/api.py` and the `cabt` Kaggle environment. Unchanged since M0 and still load-bearing:

- **Agent signature**: `def agent(obs_dict: dict) -> list[int]`
- **First call**: `obs["select"] is None` → return the **deck** (60 card IDs).
- **Every other call**: return option indices, `minCount <= len(ret) <= maxCount`, each
  `0 <= i < len(select.option)`, no duplicates.
- **Reward**: `+1` win, `-1` loss, `0` draw — a single terminal reward.
- **Budget**: `remainingOverageTime: 600` s for the whole episode. The policy must be fast.
- **Forward model**: `search_begin` / `search_step` let us simulate ahead. This is what the
  turn solver (§7) is built on.

**Key structural fact**: the action space is not fixed. Each step presents a variable-length
list of `Option` objects under a `SelectContext`. So the policy is a **pointer network** —
score each presented option, never a fixed softmax head. Action masking is implicit.

**Two engine traps that shape the code:**

1. **One global battle per process** (`cg/sim.py` → `Battle.battle_ptr`). Parallel self-play
   is `multiprocessing`, one battle per worker. Every collector is built around this.
2. **Rule agents keep module-level mutable state.** Two players must be two *separate module
   instances* or their globals interleave mid-game. `rl/teacher.py::load_teacher` execs a
   fresh isolated module per pilot for exactly this reason.

---

## 4. The policy network

`OptionScorerV3` — `rl/policy.py:131` (readable twin: `tcg/network.py`). This is what is
trained and what ships.

```
                    ┌──────── state trunk ────────┐
 state numeric (1297) ┐                            │
 context one-hot (64) ├─▶ concat ─▶ MLP(256) ──────┤
 plan vector    (27)  │                            │──▶ score(option_i)   ← policy head
 state ids 20 × 16    ┘                            │──▶ V(s)              ← value head (tanh)
                                                   └──▶ plan_logits       ← plan head
 option numeric (94) ┐
 option ids  2 × 16  ┴─▶ MLP(256) ─▶ scored against the trunk
```

- **892,483 parameters.** Card-id embedding table is `(1268, 16)`.
- `state_enc.0` input = `1297 + 64 + 27 + 20×16` = **1708**
- `option_enc.0` input = `94 + 2×16` = **126**

**The suffixes are constructor arguments, not separate classes.** A checkpoint carries no
metadata; every loader **sniffs the widths** from the state dict (`rl/policy.py:245
option_dim_of`, `rl/matchrunner.py:344`, `rl/plan_iter.py:726 _n_ids_of`).

| Name | Axis | Value | Landed |
|---|---|---|---|
| `v3` | `n_state_ids=12`, `option_dim=90` | board ids only, legacy options | M11 |
| `v3h` | `n_state_ids=20` | **h**and-aware (+8 hand-card id slots) | M15 |
| `v3o` | `option_dim=94` | **o**ption-identity | M16 |

Current nets (`osv3o_*`, `ppo_*_m20*`) are **both** — 20 state ids, 94-wide options.

**The warm-start invariant** governs every architecture change: a new net with its new
columns zeroed must be *exactly* the old net. `load_v2_into_v3`, `load_v3_into_v3h`,
`load_v3h_into_v3o` (all in `rl/plan_iter.py`) each preserve this, and it is unit-tested.
That is why the plan block sits *between* state_ctx and the id embeddings — so the plan
columns can be zero-initialized in place.

---

## 5. Encoders

`rl/encoders.py` — **numpy-only by design**, because this file ships inside the submission.

| Constant | Value | |
|---|---|---|
| `FEAT_DIM` | 36 | per-card feature vector |
| `STATE_DIM` / `STATE_V2_DIM` | 1217 / **1297** | v2 adds `_race_features` (8) + `_deck_pools` (72) |
| `OPTION_DIM` / `OPTION_V3_DIM` | 90 / **94** | v3 appends `N_OPTION_EXTRA` (4) |
| `N_CONTEXTS` | 64 | 49 defined today, head-room reserved |
| `N_CARD_IDS` / `EMBED_DIM` | 1268 / 16 | |
| `N_STATE_IDS` + `N_HAND_IDS` | 12 + 8 = **20** | board slots + hand slots |
| `N_OPTION_IDS` | 2 | acted card, target card |

- **`encode_state_v2(state, my_deck)`** → `(1297 float32, 12 ids)`. Needs the 60-card list —
  observations don't carry it, so the pilot closes over its own deck.
- **`encode_state_v3(state, my_deck)`** → same numerics, **20 ids** (12 board + 8 hand-card
  ids, sorted for permutation stability, zero-padded). From the M14 encoder audit: *the hand
  was a summed 36-dim pool that the id-embedding pathway never saw — invisible combos.*
- **`encode_option_v2(opt, obs)`** → `(94 float32, [acted_id, target_id])`. The first 90
  columns are byte-identical to v1; the appended 4 are type-multiplexed (option types are
  mutually exclusive and the type one-hot disambiguates): ATTACK → damage/cost/effective
  damage; ATTACH → energy/gap-to-charged/saturation (M19); RETREAT → damage fraction /
  prizes at risk / bench-ready (M19); NUMBER → `number/10`.

**The option-identity fix (M16) was the single biggest live gain of the campaign.** Before
it, PLAY options carried only a hand index — no `area`, no `cardId` — so the encoder never
resolved *which card* was being played: zero card features, embedding id 0. "Play Boss's
Orders" and "play Poké Pad" were byte-identical vectors decided by tie-break noise.
**431 of 1353 live decision states (~14/game, 32%) offered ≥2 trainers the net could not
tell apart.** This bug was present from v1 through M15.

`encode_option_v2_legacy` preserves the exact pre-M16 90-wide encoding so pinned baselines
(`osv2_*`, `osv3_plan0c`, `osv3h_plan1`) stay reproducible. It is selected automatically by
sniffed width — **not dead code**.

---

## 6. The turn plan

`rl/plan.py` — bundle-pure (numpy + `cg.api` + `rl.combat` only). `PLAN_DIM = 27`,
`MAX_PLAN_CANDS = 48`.

The plan is the **observable** version of the rule expert's hidden `AttackPlan`: an
`(attacker, target, attack, needs_attach)` commitment made at a turn's first MAIN prompt and
fed to the policy as *input features* for the rest of the turn.

Making the plan an explicit input is what breaks the hidden-plan imitation trap that killed
M1–M3 and M9: **labels stay consistent as a function of the inputs.** A teacher whose
reasoning is invisible produces contradictory labels for identical observations, and no
amount of capacity fixes that (§14).

**Serve-time protocol** — identical in `rl/matchrunner.py:356` and `submission/main.py`:

```
at each turn's FIRST own MAIN prompt:
    cands = enumerate_plans(obs)
    plan  = argmax(plan_logits(trunk with plan=zeros), cands)     # plans-as-options pointer
    cache plan, keyed (turn, yourIndex)
every other prompt this turn:  reuse the cached plan
no cached plan (off-MAIN, new game):  zeros(PLAN_DIM)  == "no plan"
```

Plan once and hold. Replanning every MAIN is off-distribution for the head *and* flip-flops
the plan mid-turn — measured at 0.278 vs 0.345 before the fix.

The chosen candidate's 27-vector is byte-identical to what conditions the policy, so there
is no train/serve mismatch in plan encoding.

---

## 7. The training recipe

Driver: `rl/plan_iter.py`. Canonical commands: `.claude/skills/train-ship/SKILL.md`.

```
  ┌─ 1. COLLECT ────────────────────────────────────────────────────────┐
  │  widened turn solver labels states during teacher self-play         │
  │  rl.plan_iter collect --mode expert  →  data/plan_m<N>/shard_*.npz  │
  └────────────────────────────┬────────────────────────────────────────┘
                               ▼
  ┌─ 2. TRAIN (supervised) ─────────────────────────────────────────────┐
  │  CE(policy) + 0.5·Huber(value) + plan_weight·CE(plan)               │
  │  rl.plan_iter train --data <dirs> --init <prev champion>            │
  └────────────────────────────┬────────────────────────────────────────┘
                               ▼
  ┌─ 3. MEASURE ────────────────────────────────────────────────────────┐
  │  mirror vs solver:lucario · meta co-gate · floors — all 2-seed      │
  └────────────────────────────┬────────────────────────────────────────┘
                               ▼
  ┌─ 4. PPO FINE-TUNE (M20+) ───────────────────────────────────────────┐
  │  rl.ppo --start <ckpt> --kl-coef 0.1 --defect-penalty 0.1           │
  └────────────────────────────┬────────────────────────────────────────┘
                               ▼
  ┌─ 5. SHIP ───────────────────────────────────────────────────────────┐
  │  ./build_submission.sh --checkpoint <ckpt>.pt --deck lucario        │
  └─────────────────────────────────────────────────────────────────────┘
```

### 7.1 The improvement operator

**The teacher is a search process, not an agent.** `rl/turn_solver.solve_turn_line` runs at
**5× the live deadline and 5× the node budget** with override gates bypassed
(`WIDE_NODES=4000`, `WIDE_DEPTH=10`), and its chosen line is both *executed* and *labeled*.

Why a within-turn search is cheap and sound: **within my own turn the opponent never acts**,
so the search needs no opponent determinization. Only my own draw order is hidden, and the
engine re-prompts after every action — so the solver recomputes from ground truth at each
prompt. There is no cached plan to invalidate.

`_teacher_step` (`rl/plan_iter.py:160`) picks a label source per turn:

1. **Kill line** from the widened solver, when it clears `MIN_OVERRIDE_SCORE`.
2. **SETUP plan** from the M13 setup-value net used as a search leaf, committing when the
   margin beats stand-pat by `VS_MARGIN=200`. *(Data-gen only — never live.)*
3. Otherwise the greedy `make_generic_pilot` labels.

> **The anti-aliasing invariant** (`rl/plan_iter.py` docstring): *a row's plan features always
> come from the same solve that produced that row's label.* A row carries a non-zero plan
> **iff** its label comes from a bar-clearing line. This is precisely what separates the M11
> recipe from the buddy-DAgger dead end (M9: 0.198).

### 7.2 Collection health

`collect()` prints these every run; they are the tripwires:

- plan coverage ≈ 0.98 · `SETUP`/game ≈ 3.5 · `vs_errors` = 0
- per-opponent teacher-seat win rate: `self` ~0.51 / `ext` ~0.13 / `rule` ~0.11

**Watch `vs_errors` and `SETUP=0`.** In M17, `leaf_value` hard-coded the 12-id encoder while
the value net was 20-id, so every call raised `RuntimeError` into a bare `except: pass`.
SETUP was 0 across all 10,000 games — **the value net was never evaluated once**, and the
milestone's conclusion ("value miscalibration") was retracted in M18. The fix added encoder
dispatch on `vnet.n_state_ids`, an error counter with traceback, margin percentiles, and a
loud `WARNING: SETUP=0`.

### 7.3 Training details

`BCDatasetV3` + `collate_v3`. Old plan-less shards load with shaped defaults (plans=zeros,
no plan rows), so **diverse older data keeps regularizing the plan=0 fallback policy** —
mixing `plan_m15` back in is what rescues the meta co-gate (dose-response: 0.378 → 0.487).

Policy CE is weight-column aware; value and plan CE stay unweighted so `val_acc`/`plan_acc`
remain comparable across milestones. Game-level 10% validation split, best-val checkpointing.

### 7.4 PPO's current role

`rl/ppo.py` — clipped surrogate + value loss + entropy, plus two M20 additions: a **KL anchor
toward the frozen starting policy** (`--kl-coef`) and a per-step over-attach **defect
penalty** (`--defect-penalty`). `_forward` duck-types on `plan_enc` so either twin's V3 class
works. Collection is `rl/collector.py` (opponent-pool self-play).

M20's result carries a caveat worth preserving: PPO delivered **+5.9pp of strength**
(0.4294 → 0.488) but **the milestone's stated objective failed** — over-attach was unmoved at
0.72/game, because the KL anchor that preserved strength also restrained the penalty. It
shipped as a strength result, not a behavior result.

---

## 8. Pilots and the spec grammar

A **pilot** is `agent(obs_dict) -> list[int]`. Everything — training opponents, evaluation,
league play — is built from a **spec tuple** through one factory: `make_pilot(spec, instance)`
in `rl/matchrunner.py:99`.

`rl/matchrunner.py` is *"the canonical home of the OpponentSpec vocabulary and the slot-fair
game loop"* — **one battle loop for every evaluator** (M7.2), replacing three implementations
that had drifted apart.

| Spec | Builds | Status |
|---|---|---|
| `model:<ckpt>:<deck>` | neural pilot; sniffs v3/v2/v1 from the state dict | **LIVE — the ship path** |
| `solver:<deck>` | `turn_solver.make_solver_pilot` | **LIVE — canonical eval opponent** |
| `generic:<deck>` | `generic_pilot.make_generic_pilot` | LIVE — teacher base + fallback |
| `rule:<agent>[:<deck>]` | competition sample agents via `rl/teacher.py` | LIVE — floor baseline |
| `random:<deck>` | uniform random legal moves | LIVE — floor |
| `ext:<path>:<deck>` | external Kaggle-style `main.py` (the "buddy" agent) | LIVE — collection opponent |
| `generic2/2a/2b`, `solver2/2a/2b` | M9 pilot-fix variants (a/b for attribution) | auxiliary |
| `rank:<ckpt>:<deck>` | solver + confident ranker override | **dead** (M12) |
| `vsolver:<ckpt>:<deck>` | solver with learned value as search leaf | **dead as a live consumer** (M13) |
| `mcts:<ckpt>:<deck>:<sims>` | `rl/mcts.py` | **dead** (M8.4) |
| `solver-dev:<deck>` | solver + M8.1 development tier | **dead** (M8.1) |

Three pilots do the real work:

- **The generic pilot** (`rl/generic_pilot.py`) — deck-agnostic **greedy one-action argmax**
  over hand-written tier scores. All the intelligence is in `score_option` dispatching to
  `score_attack` / `score_attach` / `score_retreat` / `score_play` / `score_card`.
- **The solver pilot** (`rl/turn_solver.py:423`) — *wraps* the generic pilot rather than
  hooking its scorers, so the pilot twins stay byte-identical and parity-pinned. On any
  exception or no line found, it falls through to greedy: the wrapper must never cost the
  crash gate.
- **The model pilot** (`rl/matchrunner.py:316`) — the offline twin of `submission/main.py`.
  Same width-sniffing, same plan state machine.

---

## 9. Teachers and decks

**Teacher packages** = a rule brain (`sample-agent*/main.py`) + a deck CSV, registered in
*both* `rl/teacher.py` and `tcg/teachers.py` (kept in parity by `tests/test_decks_teachers.py`).

| Teacher | Deck | Strength |
|---|---|---|
| `lucario` | `decks/lucario.csv` | the strongest sample agent; the shipped deck |
| `iono` | `decks/iono.csv` | weaker lightning deck |
| `tuned` | `decks/lucario.csv` | M5 parameterized Lucario (statistical tie with `lucario`) |
| `dragapult` | `decks/dragapult.csv` | **added 2026-07-19** — 0.495 vs `lucario`, 0.657 vs `iono` |

Dragapult is registered as a teacher, deck, and league anchor, but is **deliberately not yet
in the collection opponent pool** — changing the training distribution is a milestone-level
decision (§14, "you become what you train against").

**Decks.** `tcg/decks.py` is the registry, kept dependency-free on purpose because deck files
are read both by the torch training loops and by the engine-only teacher loader.
`decks/gen/` holds 62 generated candidates + `manifest.json` from `rl/deck_build.py`; they
feed the league, **not** the ship path. Deck selection for a submission is entirely manual.

Two hard-won data facts encoded in `rl/deck_build.py`:

1. **Evolution is by NAME, not card id.** `evolves_from_id` points at one specific card, but
   decks legally play any same-named card.
2. **`attackId` maps to cards by cumulative `n_attacks`** in card-id order — the parquets
   were written in one pass.

---

## 10. Measurement

Protocol: `.claude/skills/measure-agent/SKILL.md`. This is the part of the system most likely
to lie to you, so it has the most rules.

**The bar**: ≥0.55 vs `solver:lucario` at n≥800, zero crashes, sane latency — plus the
**meta co-gate** since M10: a weighted win rate against the frozen top-10 harvested-meta deck
pool, each deck solver-piloted (`rl.replay_bc meta-eval`, n=60/deck, 2 seeds).

**Mirror protocol**: screen n=400 on one seed → confirm on a **fresh** seed → pool.
*Promotion* is stricter than the bar: pooled **n≥1200 with both seeds >0.52**, plus floors
and the meta co-gate.

**Floors**: vs `random:kyogre` and `rule:lucario`, n=200 × 2 seeds. Since M18 these are
**comparative checks against the current champion, not absolute bars** — the shipped champion
failed its own pinned floors under equal measurement.

### The traps — every one of these has cost us a milestone

- **`results` code `0` = side-a WIN.** Naively summing the jsonl array gives the *loss* count.
  M15's 0.384 reads as 0.620 if you get this backwards.
- **Player 0 has a ~61% built-in advantage.** All head-to-heads slot-swap; `play_series` does
  this by default.
- **Single-seed gates are stale.** The champion's pinned 0.534 meta was seed-0 inflation; it
  measures ~0.465 on two seeds. **Every gate baseline must be 2-seed pooled.**
- **In-loop n=200 evals are promotion triggers only, never evidence.** Identical policies read
  53.0% and 46.5% in two M20 baselines (±6.5pp), and one "collapse" was pure phantom.
- **Never read strength from a 40-game flag sample.** M19's 40-game 0.475 became 0.331 at n=800.
- **Offline metrics never arbitrate.** M17 trained +13pp better offline and played 4.4pp worse.

**Forensics first.** Every milestone opens with `rl/postmortem.py` on the previous shipped
agent's replays. Since M19 the shipped agent emits one `NN|{json}` line per decision on
stderr (logits, plan commits), which `rl.kaggle_ingest agent-logs` joins back to the replay
by step — so we can see *what the net thought*, not just what it did.

---

## 11. Shipping

```
./build_submission.sh --checkpoint <ckpt>.pt --deck lucario --message "M20: ..."
   │
   ├─ [1/3] tcg.shipping export   → build submission/ from the checkpoint
   ├─ [2/3] tcg.shipping gate     → 4 gates, below
   ├─ [3/3] tar                   → dist/submission_<agent>_<stamp>.tar.gz
   │                                 + verify archive contents explicitly
   └─ [4/4] kaggle submit         → only when --message is given (opt-in)
```

**The bundle** (`submission/`, ~3.6 MB of weights):

```
main.py             numpy-only agent — no torch, no polars
deck.csv            the 60 cards this policy was trained on
policy_weights.npz  892,483 params
card_features.npy
cg/                 engine bindings — Kaggle does NOT provide these
rl/{__init__,combat,encoders,plan}.py      byte-identical copies of rl/ source
```

`submission/main.py` is a hand-written **numpy reimplementation of the forward pass**
(`_trunk_v3`, `score_options_v3`, `score_plans`), but it **imports the real encoder modules**
rather than re-deriving them — no drift-prone hand-copy. It **sniffs the architecture from
the weights themselves** (`_IS_V3 = "plan_enc.0.weight" in WEIGHTS`; id count back-computed
from tensor shapes) and hard-fails on a v1 export rather than mis-replaying it.

**The four gates**, all in `tcg/shipping.py`:

1. `parity_check` — numpy forward vs torch reference at `rtol/atol=1e-4`, plan head included
   (M20 build: max diff 8.34e-07).
2. `bundle_isolation_check` — subprocess with the project root stripped from `sys.path`,
   plays a full game, asserts `torch`/`polars` never got imported and the *bundled* `rl` won.
3. `deck_check` — legality.
4. `gate_game` — a full `kaggle_environments` self-play, both statuses `DONE`.

Plus a shell-level archive check the Python side structurally cannot do: it greps `tar -tzf`
output for every required path, because the local gate resolves imports from the project root
and so cannot catch a file missing from the *bundle*.

> ### ⚠️ A policy and its deck are ONE artifact
> A policy trained on Lucario plays Kyogre badly and vice versa. **`--deck lucario` is
> mandatory on every ship.** `tcg/shipping.py:41` still carries `DEFAULT_DECK = "kyogre"`, an
> M1-era fossil — and `decks/kyogre.csv` is actually a water deck built around 4× Mega
> Abomasnow ex, not Kyogre. M16 and M18 both shipped it by accident, which permanently
> confounds M16's 519.8 (a Lucario-trained net piloting Abomasnow zero-shot). Post-M18.1
> protocol: md5-verify `submission/deck.csv` against `decks/lucario.csv` **out of the built
> tarball** before submitting. See §16 — this is still an open defect.

---

## 12. `rl/` vs `tcg/` — the twin law

`tcg/` began as a readable refactor of `rl/`. The clean switchover never happened, and
authority is now **split down the middle of the build**:

| Layer | Authoritative |
|---|---|
| Build/ship orchestration, torch model classes, `FEAT` matrix, parity reference | **`tcg/`** |
| Modules physically copied into the bundle | **`rl/`** |
| All training loops (`plan_iter`, `collector`, `matchrunner`, `league`, `postmortem`) | **`rl/`** — many have no twin |
| Rules bundle (`generic_pilot`, `turn_solver`) | **`rl/`** |

So `tcg.shipping.export()` loads a checkpoint into a **`tcg`** class and writes an **`rl`**
encoder into the same tarball. **The parity tests are the only thing making that safe.**

**Twin pairs**: `rl/generic_pilot.py`↔`tcg/pilot.py`, `rl/encoders.py`↔`tcg/encoders.py`,
`rl/ppo.py`↔`tcg/ppo.py`, `rl/combat.py`↔`tcg/combat.py`, `rl/policy.py`↔`tcg/network.py`,
`rl/deck_search.py`↔`tcg/deck_search.py` (legality checker only — the tcg twin was trimmed to
`validate_deck` in the M25 cleanup; the gate needs it because the bundle shadows `rl` on
sys.path), `rl/bc.py`↔`tcg/behavior_cloning.py`.

**Enforcement:**
- `tests/test_parity.py` sweeps an exhaustive scenario grid and asserts equality —
  `assert count > 400` pins that the sweep really covered the branch grid.
- `tests/test_plan_iter.py::test_rl_tcg_v3_twins_are_identical` cross-loads state dicts.
- In-source `change BOTH` markers on every twinned edit.
- **Twins may drift deliberately**: a fix lands in `rl/` first and is mirrored to `tcg/`
  *only after it survives measurement*.

**The M7 no-twin rule**: `rl/matchrunner.py`, `rl/kaggle_ingest.py`, `rl/deck_build.py`,
`rl/league.py`, `rl/turn_solver.py`, `rl/plan.py`, `rl/plan_iter.py`, `rl/postmortem.py`,
`rl/collector.py` and `rl/rank.py` live in `rl/` **only** — pinned by `tests/test_imports.py`.

### Test tiers

**612 tests**, all offline against a fake engine (`tests/conftest.py` installs `tests/fake_cg.py`
into `sys.modules` before any `rl`/`tcg` import, because both build card tables at import time).

`[ENGINE]` and `[NET]` in docstrings are **prose annotations marking what a test deliberately
does not cover** — not pytest markers. `[ENGINE]` = verifiable only with the real (gitignored)
engine, deferred to a manual runbook step. `[NET]` = needs Kaggle network access.

The *executable* tiering is `tests/test_imports.py`, which uses `pytest.importorskip` to pin
dependency tiers. `RL_CG_ONLY_MODULES` is the **bundle-purity pin**: the shipped `rl/` modules
must never acquire a heavy dependency.

---

## 13. Repo layout

```
├── ARCHITECTURE.md          <- this file
├── CLAUDE.md                <- working agreement / house rules
├── build_submission.sh      <- the one-command ship pipeline (.ps1 twin for Windows)
├── cg/                      <- engine bindings — YOU provide these (gitignored)
├── rl/                      <- training + the bundled inference modules
│   ├── plan_iter.py           the training driver (collect / train / relabel)
│   ├── turn_solver.py         the improvement operator + eval opponent
│   ├── policy.py              OptionScorerV3 (+ V2/V1 for pinned baselines)
│   ├── encoders.py  plan.py  combat.py      ← these four ship in the bundle
│   ├── matchrunner.py         the one battle loop + spec grammar
│   ├── generic_pilot.py       deck-agnostic greedy pilot
│   ├── collector.py  ppo.py   self-play collection + PPO fine-tune
│   ├── setup_value.py         M13 value net — data-gen leaf only
│   ├── kaggle_ingest.py  postmortem.py  replay_bc.py   forensics + meta gate
│   ├── deck_build.py  deck_search.py  league.py        deck factory (deferred)
│   └── teacher.py             isolated rule-agent loader
├── tcg/                     <- readable twins + the authoritative ship pipeline
│   ├── shipping.py            export + gates
│   ├── network.py  pilot.py  encoders.py  combat.py  ppo.py    (parity-pinned)
│   └── decks.py  teachers.py  constants.py  library.py
├── submission/              <- the built bundle (numpy-only)
├── sample-agent{,-iono,-tuned,-dragapult}/   <- rule teachers
├── decks/                   <- lucario · kyogre · iono · dragapult + gen/
├── checkpoints/             <- trained nets
├── data/                    <- plan_m*/ training shards, kaggle/ replays, parquets
├── dist/                    <- built tarballs
├── docs/                    <- milestone diaries M0–M20, DECISIONS.md, MILESTONES.md
├── tests/                   <- 612 offline tests + fake engine
└── notebooks/               <- card EDA, model_monitor.ipynb (live score + deck audit)
```

---

## 14. The measured laws

These are earned, not assumed. Violating one has cost a milestone at least once.

1. **Agent strength ≈ teacher strength × imitation fidelity** — and fidelity *collapses*
   (~0.53–0.56 ceiling) on teachers whose reasoning is unobservable. *(M2)*
2. **Pick the strongest teacher whose reasoning is observable — and queryable.** "Observable"
   must mean **per-decision label consistency with respect to the observation**, not merely
   readable source code. *(M2 → M9 → M10)*
3. **Deck quality dominates.** Slot-fair round-robin: Lucario ≫ Iono ≫ Kyogre, with the
   *pilot* held constant. Deck surgery remains the biggest untouched lever. *(M2)*
4. **You become what you train against.** An opponent pool that was 50% mirror+random
   optimized the mirror and regressed against every rules pilot (0.44 → 0.145). *(M7.5)*
5. **Label quality dominates architecture.** M11's entire +9pp came from label quality with
   the architecture unchanged. *(M11)*
6. **A classifier can't rank siblings, but a ranker can.** "Value can't rank" was a
   loss-function problem, not an architecture problem — a pairwise head hits 0.873 held-out.
   *(M5 → M12)*
7. **Label-reweighting works on the behavior and monotonically costs strength**, in clean
   dose-response with the weighted row fraction (2.2% → −3pp; 10.4% → −10pp). *(M19)*
8. **Live gates arbitrate; offline metrics never do.** *(M17)*
9. **A policy and its deck are one artifact.** *(M1, re-learned painfully at M18.1)*
10. **Don't trust deck file names.** `decks/kyogre.csv` is a Mega Abomasnow deck. *(M1, M18.1)*

---

## 15. Measured dead ends — do not re-propose

| Approach | Killed by | Result |
|---|---|---|
| Rule-pilot teacher / BC from the sample agent | M1–M3 | fidelity caps ~55%; the clone plays *worse* (0.25) |
| Inference-time MCTS on the classifier value head | M8.4 | sims ladder flat: 0.515 / 0.490 / 0.495 at 16/32/64 |
| M8.1 development-tier solver | M8.1 | pooled 0.492, n=1600 |
| BC from harvested elite replays | M10 | elite fidelity saturates 0.527 — imitation transfers incoherence, not strength |
| DAgger from a hidden-plan teacher (buddy) | M9 | **0.198** (−14.7pp): plan-coupled labels are contradictory w.r.t. the obs |
| Override-style consumption of **any** eval signal on non-lethal turns | M8.1, M12, M13 | five measurements, all below the 0.500 null |
| Inference-time guards on rare prompt classes | M11 | three ATTACH_FROM guards, all reverted |
| Offline disagreement-weighted BC repair | M18 | screen **0.278** (−15pp) — actively harmful |
| Any improvement operator built on `score_leaf`'s non-lethal ranking | M12 | the bottleneck is `score_leaf` itself, not what consumes it |
| EI/DAgger rounds on a rarely-firing operator | M11 | the bar-honoring solver improves only ~7% of turns |

> **One amendment.** "More PPO iterations is dead" (M8.3) was measured on **v2 nets and does
> not transfer to V3**. M20 revived PPO on the V3 encoder and it produced the champion.
> `plan_iter --mode ei` and `plan_iter relabel` remain wired but are dead recipes — they exist
> so their results stay reproducible, not to be run again.

> **Epoch amendment (M41b, 2026-08-04).** The M37 audit found the prize-term inversion was
> systemic — 4 sites repo-wide, 1 known — and MILESTONES.md's EPOCH MARKER voids every
> solver-backed number recorded before 2026-07-31. Three entries above are **pre-epoch** and
> must not be read as verdicts on the post-fix substrate: the **M8.1 development-tier solver**
> (routes through `score_leaf`), **override-style consumption of any eval signal** (two of its
> three cells are solver-based), and **any improvement operator built on `score_leaf`'s
> non-lethal ranking** (explicitly `score_leaf`). M40 S3's post-fix measurement — wrapping the
> net in the solver = **+125 ELO [+90, +163]** — directly contradicts what those entries imply.
> **M8.4 (MCTS on the classifier value head)** is voided differently: it never used
> `score_leaf`, but it ran on the *classifier* head, which M12 found to be the wrong signal and
> the E0-validated outcome-trained value has since replaced — substrate-superseded, untested on
> the current one. The other entries (BC/DAgger/imitation rows) are post-epoch-valid and stand.
> Consumption discipline is unchanged: search output as **labels** is open (BACKLOG #10);
> override-style consumption on non-lethal turns stays dead until a new post-fix measurement
> says otherwise.

---

## 16. Known gaps

Real defects found while rewriting this document (M20). None are fixed.

1. **`DEFAULT_DECK = "kyogre"` in `tcg/shipping.py:41`** — the fossil that mis-shipped two
   submissions. Mitigation is currently *procedural* (verify the md5), not code.
2. **`rl/plan.py` is missing from `build_submission.sh`'s `REQUIRED` archive check**, despite
   being mandatory for every v3 bundle. It is also absent from `test_imports.py`'s bundle-purity
   tier — the least-protected file on the shipping path.
3. **`encoder_parity_check()` is unreachable on the v3 path** — it runs only in the v1 branch,
   and v1 can no longer be replayed. The isolation game covers it in practice.
4. **`tcg/__init__.py`'s docstring states the opposite of §12** — it still claims `rl/` is the
   live submission path and `tcg/` is pending switchover.
5. **`rl/ppo.py` and `rl/bc.py` docstrings are false** — both still say the core functions are
   unimplemented stubs.
6. **`CLAUDE.md` lists `rl/network.py`, which does not exist** (it is `rl/policy.py`), and says
   docs cover M0–M14.
7. **`README.md` describes the M0–M7 design** and points at superseded commands
   (`rl.gate`, `rl.export`, `build_submission.ps1` as primary).
8. **`.claude/skills/measure-agent/SKILL.md` still pins `osv3o_plan5` as champion** — one
   milestone behind `ppo_best_m20legB.pt`, and its baselines predate M20's numbers.

---

## 17. References

**The environment** — the official sample agents remain the best documentation of the engine
API that exists: `sample-agent/` (Mega Lucario), `sample-agent-iono/`,
`sample-agent-dragapult/`, and `reference/reinforcement-learning-and-mcts-sample-code.ipynb`
(the hosts' transformer + determinized-MCTS agent). The MCTS half of that sample is a
measured dead end here (§15), but its option-scoring decoder independently validates the
pointer-network shape, and its card-ID `EmbeddingBag` is what our embedding table became.

**Tooling**: [kaggle-environments](https://github.com/Kaggle/kaggle-environments) ·
[openskill.py](https://openskill.me/) · [uv](https://docs.astral.sh/uv/)

**RL background**, still worth reading in this order:
- [OpenAI Spinning Up](https://spinningup.openai.com/) — Intro to RL parts 1–3, then
  [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Karpathy, "Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/) — policy
  gradients from scratch
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  — read before touching `rl/ppo.py`
- [Andy Jones, "Debugging RL Without the Agonizing Pain"](https://andyljones.com/posts/rl-debugging.html)
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) — our reference, never a dependency
- Papers: [PPO](https://arxiv.org/abs/1707.06347) · [GAE](https://arxiv.org/abs/1506.02438)

**Why no RL framework**: our action space is a variable-length option list scored by a
pointer network, and the env is a callback (`agent(obs_dict) -> list[int]`) with nested-dict
observations. SB3/Tianshou/TorchRL/RLlib all assume fixed `Discrete`/`Box` spaces and a
Gym-style `env.step()`. Wrapping would cost more code than the PPO loop does. Self-play with
opponent pools isn't native to SB3, and the engine is a ctypes DLL with one global battle per
process, so there is no vectorization win available either.
