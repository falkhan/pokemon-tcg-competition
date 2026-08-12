# M43 plan — from imitation to game-play learning

Piotr's directive (2026-08-05): *"Using BC clones can only get us to a certain
point. Now we need our pilots to get trained on specific decks through games
against other pilots, and self-play to adjust the weights and learn how to
advantage the board."* Scope decision (same day): **Lane B first in execution,
Lane A as the headline question**; plan + code prerequisites land on this
branch, training executes on the box with the M38+ artifacts.

## Why now, in one paragraph

Imitation is measured-exhausted (M39: five arms, two corpora, all negative on
the 900+ panel; the pre-registered branch fired) and structurally capped
(M40 X5: BC retains 0.20 [0.07, 0.33] of demonstrator edge). Distilling the
search is dead (M41b Stage B: both teacher budgets grim-negative or
inconclusive; ~1M games/cell to detect a distilled edge at 0.20 retention —
BACKLOG #10 closed). What has never been run honestly is learning from
*played games*: PPO — whose M2–M4 "failure" was the multiplicative-loss bug
(DECISIONS.md M7.4b) and whose M20 run produced a then-champion — and
self-play corpus manufacture, whose tooling M40 S2 built, validated, and left
pointed at exactly this. M43 runs both as pre-registered lanes.

**Standing caveat threaded through every gate below:** three consecutive
ships inverted offline→live (retain_b −18pp being the worst). Offline deltas
are SCREENING, live decides, and X5 voids all bed absolutes — every verdict
is a same-battery arm-minus-control delta, never a band claim.

## Backlog disposition (the critical review this plan came from)

| item | disposition |
|---|---|
| #10 distil-the-search | **DEAD** (M41b Stage B) — struck in BACKLOG.md |
| #9 AZ/PSRO population loop | stays parked (compute; Stage B weakens its premise) |
| #13(c) PPO + shaped rewards | **picked up as Lane A**, falsification order kept verbatim |
| #2 offline best-response | **partially picked up as Lane B** (retention + α=0.25, the combination M39 never ran) |
| #6 value-net retrain | resolved into the Phase-0 E0 check (every BC net already trains a head; only measurement was missing) |
| deck question | resolved in practice by M41's ogerpon arm; the residual — pilot strength — is Lane B |
| #4, #11, #12, rl/↔tcg/ dedup | stay deferred, orthogonal to the declared block |

## Phase 0 — forensics on the M41b ships (mandatory, first)

Two live subs (2026-08-05, commit 4094166), zero reads at plan time. G-10
read floor: **n ≥ 45 games per sub**. Wrapper: `scripts/m43_forensics.sh`
(E0 + ingest + family split + Hermes notifications); postmortems and the
replay review with Piotr are manual follow-ups.

| step | what | when |
|---|---|---|
| 0.1 | `kaggle_ingest refresh --subs 55265099 55265105` + `agent-logs` on losses | after accrual (1–2 days) |
| 0.2 | `kaggle_ingest forensics --sub <id>` + `rl.postmortem` on ≥3 losses/sub, replay review with Piotr | after 0.1 |
| 0.3 | **E0 on `m41b_wide_prod.pt`'s value head** (`scripts/m40_e0_value.py`) | day 1, no accrual needed |

**Pre-registered consequences (written before the numbers):**

- **0.a — wide-ship live read.** Holds band relative to its predecessors →
  the width-143 substrate is live-validated; both lanes proceed on it. A
  **fourth consecutive offline→live inversion** → all M43 offline gate
  language stays screening-only, M43 ships at most once, and the inversion
  is diaried as the campaign's top open instrument defect.
- **0.b — ogerpon loss modes.** Live losses reproducing the benched-out /
  `empty-bench` mode measured offline (M41b) → Lane B's target is
  live-confirmed. A new dominant loss family → it joins both lanes'
  opponent pools/beds before any run starts.
- **0.c — E0 verdict on the wide head** (bars frozen from M40 E0: pass
  ≥ 0.62 matched-pair, kill < 0.58): PASS → the wide net's own head serves
  as GAE critic and as Φ. KILL → warm-start the critic from
  `m39_retain_b.pt` (the only E0-passing head, 0.642) via `--value-ckpt`,
  after a width-compat check — `warm_start_value` loads `value_head.*` with
  `strict=False` and the lineages differ in width; if the keys don't
  transfer cleanly, the Φ arm is dropped and Lane A runs as plain
  outcome-reward PPO (the falsification order's step 2 then reports "no
  valid Φ exists yet"). BETWEEN → GAE yes (GAE tolerates a mediocre V),
  Φ no.

## Lane B — self-play corpus manufacture for ogerpon (runs first, ~1–2 days)

**Target, measured (M41b, n=1000/cell):** the ogerpon deck beats the grim
beds at **0.940 under the generic rule pilot** and **0.993 under its own
trained net**, but **0.562 under the alakazam-trained wide candidate** —
a pilot gap, not a deck gap, with benched-out/`empty-bench` as the loss
mode. And the width-143 columns are corpus-size-bound: **+2.78pp z=4.02 at
~23.1k rows, −0.34pp z=−0.47 at ~10.1k** (M41b data-hunger A/B). Ogerpon's
corpus is **10,468 rows**; live accrual cannot close a ~13k-row gap.
Manufacture it: the net plays its own deck, winning-seat rows carry full
weight, loser rows α=0.25.

- **B0 — smoke (half a day, kills cheap).** Arm =
  `model-pz:checkpoints/m41_ogerpon.pt:decks/ogerpon.csv` (the ship's serve
  config; gustsnipe died by its own G-11 probe) through
  `scripts/m40_s2_collect.py`. Both defaults must be overridden (`--arm`
  AND `--deck` default to alakazam). One 50-game chunk; confirm rows/game
  and that shard option width feeds the `--init-wide` retrain's
  `option_dim_of` sniffing.
- **B1 — calibration.** ~200 games vs the pool (grim_d1–d3, topgrim,
  wall_d1–d3, ogerpon mirror) at `--tau 0.6`, with the collector's frozen
  kill: `--kill-above 0.90 --kill-after 20000`. Agreement > 0.90 → one
  retry at τ=0.8; still > 0.90 → **the net has nothing to learn from its
  own play at this band; Lane B dies in half a day** and the diary records
  that ogerpon's gap needs off-policy rows (harvested Grass demonstrators —
  BACKLOG deck-question section).
- **B2 — harvest.** ~2,000 games across beds + seeds (≈2× the row gap at
  the measured ~70–79 rows/game; minutes of engine time, resumable,
  interrupt freely).
- **B3 — retrain.** `plan_iter train` from `m41_ogerpon` with
  `--init-wide`, data = the re-encoded live ogerpon corpus + B2 shards +
  **retention shards** (M39's actual result, +1.85pp z=4.01 — retention is
  the default now, not an arm), `--outcome-weight 0.25` (α=0.25 beat
  winners-only by +2.28pp), 14 epochs (the dose law is substrate-bound —
  M41b). Output: `checkpoints/m43b_oger_wide.pt`.
- **B4 — gate.** Pre-registered hashed spec **`docs/specs/m43_laneB.json`
  (hash `9da112f64aaa77f8`)** via `scripts/gate_spec.py run/decode`:
  control = `m41_ogerpon` same battery, 19 beds (grim/topgrim/wall d1–d3 +
  loss families + `rule:tuned`), n=1000/cell, **bars: pass ≥ +2.0pp pooled
  delta, kill ≤ 0.0pp** (two-proportion z vs same-battery control, G-9).

**Decision branches:** PASS → ship-candidate path below. (0, +2pp) → ONE
retry with a doubled B2 corpus (the data-hunger law is roughly linear in
this range); no second retry. KILL → the corpus-size explanation of the
ogerpon width flat-line is **falsified** (it was one of three untested
explanations in M41b) — log it, close the lane.

## Lane A — honest PPO on m41b_wide_prod (the headline, rest of the week)

"Honest" means: correct losses (the M7.4b fix), a modern substrate, a full-
roster gate, and pre-registered arms. `rl/ppo.py` is complete (GAE, clipped
surrogate, KL anchor to the frozen start, entropy 0.001 post-M23,
`--plan-coef`); `rl/collector.py` gives `spec=weight` opponent mixtures with
`mirror=`/`past=` tokens and hard-refuses dragapult in any training pool
(the doubly-clean out-of-loop evaluator — it remains legal in *gate* beds).

- **A0 — prerequisites (landed on this branch, this commit):**
  1. **The PPO gate path**: `docs/specs/m43_laneA_base.json` (hash
     `c21d3351e44de0e9`) and `docs/specs/m43_laneA_phi.json` (hash
     `ad5930925ddf024b`) — arm `model-c-pkgz:checkpoints/
     ppo_best_m43a_{base,phi}.pt:alakazam_v2_h4` vs control
     `model-c-pkgz:checkpoints/m41b_wide_prod.pt:alakazam_v2_h4`, the full
     25-bed m40 roster, n=400/cell, bars pass ≥ +2.0pp / kill ≤ 0.0pp.
     Full roster is non-negotiable: exploiter overfit is measured
     (0.90→0.54, arXiv 2404.16689) — never gate on the target bed alone.
  2. **`--shaping value`** in `rl/collector.py` + `rl/ppo.py`: Φ = V(s)
     from the **frozen start** checkpoint (never the moving learner — same
     anchor discipline as the KL term), reusing the existing potential
     shaping site (`F = coef·(Φ′−Φ)` telescopes out of the return).
     Arch-compat is hard-checked; `phi_ckpt` must be explicit. Tests:
     `tests/test_value_shaping.py`; smoked end-to-end on a real 2-game
     collect.
  3. On the box before the first leg: a 10-game width-143 collect smoke
     from `m41b_wide_prod` (the collector width-sniffs via `option_dim_of`;
     must be verified against the real checkpoint, which this branch
     cannot).
- **A1 — arms, per BACKLOG #13(c)'s pre-registered falsification order,
  kept verbatim:**
  > (1) `advantage_by_type` on a fresh post-epoch PPO shard to confirm the
  > gap still exists; (2) `Phi = V(s)` potential shaping as the control arm
  > — free, principled, no new signals; (3) hand-crafted potentials only if
  > (2) leaves something on the table, and only genuine state functions;
  > (4) never the event bonuses without a potential reformulation.

  Concretely: **A-base** (plain outcome reward + prize shaping,
  KL-anchored) and **A-phi** (`--shaping value --race-shaping <coef>`).
  Step (1) runs on A-base's first shard before A-phi launches. Step (3)
  unlocks only if A-phi > A-base at z ≥ 2 AND the supporter-credit hole
  persists. **Step (4) — event bonuses (cards-taken / avoided-attacks /
  survived-pokemon) — is banned: cards-taken pays the agent to deck out, a
  measured live loss family.**
- **A2 — run recipe (per leg):**
  ```
  uv run python -m rl.ppo --start m41b_wide_prod.pt \
    --learn-deck alakazam_v2_h4 --eval-deck alakazam_v2_h4 \
    --workers 8 --games-per-iter 400 --iterations 50 --eval-every 5 \
    --kl-coef <M20's value — look up in docs/M20.md on the box> \
    --opponents "model:checkpoints/m39_bc_grim.pt:grim_live=0.15" \
                "model:checkpoints/m39_bc_grim_b.pt:grim_live=0.10" \
                "model:checkpoints/m39_bc_topgrim.pt:grim_live=0.10" \
                "model:checkpoints/m38_bc_wall.pt:greattusk_wall=0.10" \
                "model:checkpoints/m40_bed_garchomp_d1.pt:data/kaggle/garchomp_c7b3253f_deck.csv=0.08" \
                "model:checkpoints/m40_bed_rocket_d1.pt:data/kaggle/rocket_59e27a5e_deck.csv=0.07" \
                "mirror=0.25" "past=0.15" \
    --tag m43a_base       # or m43a_phi, + --shaping value --race-shaping <coef>
  ```
  Grim ≈0.35 of the mix (the ladder is 49.5% grim, down-weighted for bed
  flattery), wall + loss families ≈0.25, mirror 0.25, past 0.15. Dragapult
  is auto-refused by the collector. Hermes per CLAUDE.md:
  `scripts/hermes_heartbeat.sh m43a runs/m43a_<arm>.log '^iter '` in the
  background, plus stage-transition and failure sends.
- **A3 — inner kill bars (per run, before any gate is spent):** entropy
  must not RISE over the leg (the M23 diffusion signature); KL to the
  frozen start bounded; the `vs_teacher` promotion curve must beat the
  warm-start baseline within 15 iterations — else stop the leg and diary
  the curves.

**Decision branches:** both arms ≤ 0 on their gates → **PPO-at-this-scale
is honestly negative for the first time** — recorded with the same standing
as M39's ceiling result, and the next milestone moves to population /
bed-building (#13(a), the double-oracle lane). A-phi > A-base at z ≥ 2 →
shaping is real; step (3) unlocks. Either arm PASSES → ship-candidate path.

## Costs (measured throughput, 8-worker cap)

| item | games | wall clock |
|---|---|---|
| Phase 0 (E0 + ingest + postmortem) | — | ~1 h active; 1–2 days ladder accrual |
| B0 + B1 | ~250 | < 1 h incl. reads |
| B2 harvest | ~2,000 | ~1 h with chunking overhead |
| B3 retrain (14 ep, ~40k rows) | — | ~1–2 h CPU |
| B4 gate (19 beds × n=1000, × arm+control) | ~38k | ~1.5 h |
| A2 one PPO leg (50 × 400 + evals) | ~22k | ~4–6 h |
| A gates (25 beds × n=400, × arm+control, × 2 arms) | ~40k | ~1 h |
| QC battery + replays per ship candidate | ~30 | ~30 min + review |

Two A legs + one re-run ≈ 1.5–2 days. The whole milestone fits one week
with margin for one retry per lane. The lanes do NOT share collection
(different decks; PPO shards carry logprobs/values, s2 shards feed
`plan_iter train`) and cannot truly run concurrently under the worker cap —
sequential is the honest schedule.

## Scripts: reuse vs written

**Reused unchanged:** `scripts/m40_s2_collect.py`, `rl/plan_iter.py train`
(multi-`--data` retention, `--outcome-weight`, `--init-wide`),
`scripts/m40_e0_value.py`, `scripts/gate_spec.py`, `rl/ppo.py` +
`rl/collector.py` (plus the small addition below), `rl/kaggle_ingest.py`,
`rl/postmortem.py`, `scripts/watch_games.py`, `scripts/qc_battery.py`,
`scripts/ship_verify.py`, `scripts/hermes_heartbeat.sh`.

**Written this commit:** `docs/specs/m43_laneA_base.json`,
`docs/specs/m43_laneA_phi.json`, `docs/specs/m43_laneB.json`,
`scripts/m43_forensics.sh`, `--shaping value` + `--phi-ckpt`
(`rl/collector.py`, `rl/ppo.py`), `tests/test_value_shaping.py`.

## Ship criteria (all of them; no exceptions this time)

1. Hashed-spec gate PASS vs the same-battery control — and the verdict text
   states that offline deltas are screening only.
2. `scripts/ship_verify.py` + the full multi-deck QC battery
   (`scripts/qc_battery.py --prefix m43_qc_<arm>`), including the
   previous-ship-tarball mirror leg and a bespoke leg exercising the
   changed behavior (Lane B: a grim bed at n large enough to see the
   benched-out mode; Lane A: the heaviest training-pool family).
3. **Piotr's manual replay review and explicit go — not waived.** (M41b
   shipped with the review waived; M43 does not compound that precedent.)
4. `MODELS` dict in `notebooks/model_monitor.ipynb` updated in the ship
   commit (standing rule).
5. Hermes notifications at every stage transition; heartbeat on any run
   > 30 min.

Which slot ships if both lanes pass (ogerpon, alakazam, or both) is
**Piotr's call at ship time**, per the standing deck-confirmation rule.

## Non-goals (pre-registered; do not reopen inside M43)

- No BACKLOG #9 (AZ/PSRO population loop) — parked on compute.
- No #10 (distil-the-search) — killed by M41b Stage B.
- No event-bonus rewards of any kind — non-potential, measured deck-out
  hazard.
- No serving search at inference — the stop-invest line stands; the solver
  remains a bed/label instrument.
- No deck switch — the deck question stays where BACKLOG.md left it.

## Open items carried to execution time (on the box)

- Lane A `--kl-coef`: start from M20's value (in the M20 diary).
- A-phi's `--race-shaping` coefficient: start at the historical race-shaping
  scale; it is a screening knob, not a pre-registered bar.
- Both live subs stay up through Phase 0 accrual (default yes, until the
  45-game reads complete); a Lane B ship replaces 55265105 only after its
  read.

## Amendment — 2026-08-12: box migration; spec r1 re-registration

The project moved to a new machine (i7-11800H, RTX 3070; the old box is
unavailable). The migrated clone carries the full git history, the raw
Kaggle cache (5,807 episodes through 2026-08-03), checkpoints through ~M33
and the tracked ship bundle — but none of the M38+ artifacts this plan
assumed: `m41b_wide_prod.pt`, `m41_ogerpon.pt`, `m39_retain_b.pt`, all 21
gate-bed checkpoints, the three harvested loss-family deck CSVs, the ogerpon
live corpus and the retention shards. Piotr's call (2026-08-12): **rebuild
locally from the documented recipes** rather than treat the plan as blocked.

What that changes, and what it does not:

1. **`m41_ogerpon.pt` is recovered EXACTLY, not rebuilt** — sub 55265105 is
   a re-ship of the tracked M41 bundle (`submission/policy_weights.npz`,
   commit `3db93a8`; `submission/deck.csv` md5-equals `decks/ogerpon.csv`).
   Lane B's init and control are therefore the live weights, bit-for-bit.
2. **`m41b_wide_prod.pt` and every bed are NEW G-13 DRAWS** rebuilt from the
   recipes in `docs/M38.md`/`M39.md`/`M41b.md` and
   `scripts/m39_build_panels.sh` / `m40_build_lossfam_beds.sh` /
   `m40_build_topgrim_panel.sh` on the refreshed cache. Bed absolutes were
   already void (X5); the same-battery arm-minus-control design is
   unaffected. The Lane A control is the rebuilt wide net — the same net the
   arm trains from — so the delta remains a clean single-variable read. The
   caveat: it is NOT bit-identical to what sub 55265099 serves; Phase 0's
   live read (consequence 0.a) speaks about the live net, the gates about
   the rebuilt one.
3. **Spec hashes**: the r0 hashes (`9da112f64aaa77f8` laneB,
   `c21d3351e44de0e9` laneA_base, `ad5930925ddf024b` laneA_phi) were
   registered against the old box's weights and are **void**. The specs are
   re-registered as `*_r1` (same arms, beds, n, seed, bars — only the name
   and a non-hashed `note` changed): laneB_r1 `249a36090018c78f`,
   laneA_base_r1 `b01885e1111300ed`, laneA_phi_r1 `b7ab4a2513acf803`.
   Registered BEFORE any bed was rebuilt or any gate run.
4. **"Retention shards" in B3** is read as the lineage's own live corpus
   (`data/bc_m43_oger_w143`), which is already the first `--data` term —
   M39's retention arms mixed the lineage's own training corpus into the
   fine-tune, and for `m41_ogerpon` that corpus IS the live ogerpon corpus.
   The alternative reading (cross-deck alakazam shards) has no precedent and
   risks deck-context contamination. Flagged for Piotr's review.
5. **`m39_bc_top` family is not rebuilt** — it appears in no M43 spec and no
   Lane B pool. `m39_retain_b.pt` is rebuilt only if E0 actually KILLS.
6. **Deferred fixes landed with this amendment** (commit on `feature/m43`):
   the invariant shaping form (BACKLOG #5 — `F = γΦ(s′) − Φ(s)`,
   Φ(terminal)=0, closing the `coef·(Φ_last − Φ_first)` residual BEFORE any
   Lane A collection), `--shaping-coef` alias (#6), `tcg.shipping export
   --fixes` + `ship_verify --gate-arm` set-equality (#4), and `--device`
   GPU plumbing for the two trainers (collection/eval/serving stay CPU;
   checkpoints save CPU tensors).
