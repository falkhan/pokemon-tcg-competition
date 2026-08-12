# M44 — the Kaggle sim: competitive PPO league across (pilot, deck) pairs — DRAFT r2 for Piotr's review

**Pre-registered:** DRAFT 2026-08-12 — bars and specs to be hashed at
execution start, after Piotr's review of this draft. Nothing below is a
registered bar yet.

**REVISION r2 (2026-08-12): pipeline-verified draft.** Every CLI token,
tool and artifact named by r1 was checked against the working tree. The
D2 command survives verbatim (all twelve flags exist in `rl/ppo.py`;
`model-pz:ckpt:deck=w`, `mirror=`, `past=` all parse in
`rl/collector.py parse_pool`; cross-deck opponents are fully supported;
dragapult is hard-refused in any training pool at
`rl/collector.py:218-246`). Four r1 claims did NOT survive and are
corrected in place:

1. **D3's instrument claim was wrong on the record.** M43's gate was
   `scripts/gate_spec.py` — an UNWEIGHTED pooled two-proportion z over
   the hashed 25-bed spec (`docs/M43.md:331`: 0.7506 vs 0.7135, +3.72pp,
   z=5.93). The meta-weighted decoder is a different tool
   (`scripts/m40_decide.py`) with an incompatible runs layout, a 22-bed
   pool (topgrim excluded), and a missing input
   (`data/m39_live_mix.json`, regenerable). D3 now selects AND gates on
   the gate_spec instrument; meta-weighting is demoted to an optional
   diagnostic.
2. **The round-robin league grid assumed a tool that does not exist.**
   `rl/rank.py` is the M12 value-as-ranker (collect/train), not a
   ranking harness; `rl/league.py` is the stale M7 openskill table. The
   grid is now a 3-cell `matchrunner play` diagnostic — no new tooling.
3. **D2's "inner kills" are OPERATOR procedure, not code.** `rl/ppo.py`
   has no entropy guard, no KL bound, no auto-stop, no revert — the bars
   are watched on TensorBoard and enforced by killing the process,
   exactly as M43 ran them (`docs/M43-plan.md:186-190`). D2 now says so
   and states the revert semantics the pipeline actually provides.
4. **D4 cited the E0 KILL bar as the pass bar.** Pass is matched-pair
   ≥ 0.62 AND within-game variance share ≥ 0.10; 0.58 is the kill
   (`scripts/m40_e0_value.py:57-58`). Under the one-round compression
   D4 drops transplant experiments entirely — M43 already measured the
   transplant (wide body + retain_b head → 0.559 FAIL, "the head is
   body-specific"), and no transplant harness exists in the repo.

Plus one new hard prerequisite: **Phase R (artifact preflight)**. The
repo tracks no checkpoints (`checkpoints/**` gitignored); a fresh
checkout — like the one this revision was verified on — holds NONE of
the seed nets, none of the 25 gate beds, no E0 corpora, no
`dist/qc_beds/`. M44 must inventory artifacts on the execution box
before any game is spent. See Phase R for recovery routes and the
descope ladder.

**AMENDMENT (2026-08-12, Piotr): the competition ends 2026-08-16 — M44
compresses to ONE round** (3 legs + league + gate), champion ships
2026-08-13 evening at the latest so it accrues 2+ days of ladder games.
D2's "2 rounds pre-registered" is superseded: round 2 runs only if the
round-1 champion ships early and time clearly allows. The second part of
the amendment is DONE: `m43b_oger_wide` shipped 2026-08-12 as sub
55464395 (ogerpon deck, QC 10W-2L, Piotr's go on record, monitor MODELS
dict updated in commit 1ade100) — the Lane B live A/B is running on the
ladder now, not pending.

## The directive

Piotr (2026-08-12): *"explore the possibility of 2 pilots playing each
other to simulate the kaggle environment — but with PPO learning. We will
then select the agent+deck with the highest winrate."*

Read as a design: a small **population of (pilot, deck) pairs that train
BY PLAYING EACH OTHER** — the live ladder in miniature — with PPO doing
the learning, and a final selection of the strongest pair as the ship
candidate. This picks up **BACKLOG #13(b)** (clones vs clones — the
double oracle) and the compute-feasible core of **#9** (population loop,
parked since M38 on compute that the new box + GPU + OMP pinning now
provide: a full 50-it PPO leg is ~2.5h, a hashed 25-bed battery ~11 min
at the pinned ~31 games/s).

## Why now

- M43 proved the substrate: honest PPO improves the strongest BC start
  (+3.72pp z=5.93) with stable inner dynamics. One arm, one deck.
- The M43 A-base leg trained against FROZEN opponents (beds + past
  selves). The live ladder is not frozen — it is other adapting agents.
  A co-training league is the cheapest honest approximation of that.
- The M43.1 rebuild produced trained/clonable starts for several
  archetypes **on the M43 box** (manifest `docs/M43.md:382-402`; the
  repo itself carries none of them — Phase R): the alakazam PPO
  champion, the exact ogerpon net, and BC clones of grim (90k-row
  corpus — the ladder's biggest family), plus thinner
  wall/arch/garchomp/rocket corpora.
- Lane B's carry-forward said ogerpon needs opponents with headroom, not
  more corpus — a league where its opponents ALSO improve is exactly that.

## Phase 0 — forensics first (standing rule)

- `uv run python -m rl.kaggle_ingest refresh` for subs 55464234 and
  55464395 — the local `data/kaggle/episodes.parquet` (9,054 rows) tops
  out at sub 55450580; NEITHER new ship is ingested yet.
- Live read of 55464234 once n≥45 (G-10; the only datum on record is the
  hour-1 public score 883.4, not a claim). Read 55464395 vs incumbent
  55265105 the same way — that A/B is the ladder deciding what the
  ceilinged offline battery could not. Diary both, incl. replay
  forensics of losses.

## Phase R — artifact preflight (hard prerequisite, before D1)

Verify on the EXECUTION box, ~5 min if it is the M43 box (`ls` + md5
against the M43.1 manifest), else a rebuild measured in hours:

| artifact | role | recovery if missing |
|---|---|---|
| `checkpoints/ppo_best_m43a_base.pt` | pair K seed AND the D3 control | **no repo route** — the 55464234 npz was never committed (the tracked bundle went M41 ogerpon → m43b ogerpon, commits 3db93a8 → 5e59a9f). Exists only in `dist/submission_neural_20260812_183326.tar.gz` + `checkpoints/` on the M43 box, else Kaggle re-download. Single point of failure — open item 6. |
| `checkpoints/m41_ogerpon.pt` | pair O seed | exact recovery from git: commit 3db93a8's tracked npz (the documented M43 route; md5 `4eae59d0…`, `docs/M43.md:66-68`) |
| `checkpoints/m39_bc_grim.pt` | pair G seed | rebuild per `scripts/m43_rebuild.sh` recipes (grim d1: 90,867-decision corpus `3121746f@700`, init m28_winners, val .642) |
| 21 panel beds + `m28_winners.pt` | the 25-bed gate roster | `scripts/m43_rebuild.sh` after `kaggle_ingest refresh` + harvest; m28_winners recoverable from commit 2946a56's tracked npz if absent |
| `dist/qc_beds/`, previous ship tarballs | qc_battery legs | copy from M43 box / re-export |

**Descope ladder if Phase R forces a rebuild:** K > G > O > D. K is the
champion continuation and the control — non-negotiable; G is the novelty
seat; O costs a ~1.5h leg the compressed window may not have; D was
default OFF already.

## Design

### D0 — form (settled in draft, challengeable)

**Iterated best response, NOT simultaneous both-seat learning.** Each
leg, ONE pair trains while the others sit frozen in its opponent pool.
Rationale: (a) zero collector surgery — the pool mechanism
(`--opponents "model-pz:ckpt:deck=w"`) already does this;
(b) simultaneous two-seat updates make both policies non-stationary
learning targets — the known cycling failure of naive co-evolution; the
collector also only records the learning seat, and that is a feature.
True simultaneous self-play (both seats recorded, one net) remains what
`mirror=` already provides WITHIN each leg. v2 territory if M44 pays.

### D1 — the roster (pilot, deck) seeds

| pair | start checkpoint | deck | fix token (pool + gate + ship) | standing |
|---|---|---|---|---|
| **K** (kazam) | `ppo_best_m43a_base.pt` | alakazam_v2_h4 | `model-c-pkgz` (conserve,racemode2,racemode4,planzero) | live champion, sub 55464234 |
| **O** (oger) | `m41_ogerpon.pt` | ogerpon | `model-pz` (planzero) | exact M41 live net |
| **G** (grim) | `m39_bc_grim.pt` (val .642, 90k-row corpus) | grim_live | `model-pz` (provisional) | NEW learner — the ladder's biggest family (~49.5%) gets a pilot of ours for the first time |
| D (stretch, default OFF) | `m40_bed_garchomp_d1.pt` | garchomp c7b3253f | `model-pz` | only if K/O/G come in under budget |

Fix tokens are pre-registered per pair so pool opponents == gate arms ==
ship artifact (the M43-review finding-#4 discipline; `ship_verify.py
--gate-arm` enforces it at export).

Pair D mechanics, should it ever activate: the deck is NOT a `decks/`
name — the csv lives at `data/kaggle/garchomp_c7b3253f_deck.csv`, which
works in opponent specs (path-tolerant `resolve_deck`) but NOT as
`--learn-deck` (bare-name-only, `rl/collector.py:54`); training pair D
requires copying the csv to `decks/garchomp_c7b3253f.csv` first. Do not
confuse it with `decks/cynthia_garchomp.csv` — a different deck.

Dragapult stays the held-out evaluator everywhere (the collector
hard-refuses it in training pools — `_assert_dragapult_held_out`,
`ValueError` unless `ALLOW_DRAGAPULT_TRAINING=1`; it remains legal in
gate beds).

**Seed floor (kill; NEW bar, first registered here — no precedent):**
every seed must score ≥0.25 vs each rule anchor (tuned, iono; n=400 per
anchor, ~±4.9pp CI half-width) before round 1, via plain
`matchrunner play` — a pair too weak to threaten anyone teaches nothing
and pollutes the league. K passes trivially; run it anyway for the
record.

### D2 — the round (ONE, per the amendment)

Per pair P, sequential under the worker cap (fresh tag per leg — a
reused tag inherits the old `ppo_best_<tag>.pt` and its baseline bar,
`rl/ppo.py:489`):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m rl.ppo \
  --start <P.seed>.pt              # resolved relative to checkpoints/
  --learn-deck <P.deck> --eval-deck <P.deck> \
  --iterations 25 --games-per-iter 400 --workers 8 \
  --eval-every 5 --eval-games 200 \
  --kl-coef 0.1 --device auto --tag m44_<P>_r1 \
  --opponents "<Q1.fix>:checkpoints/<Q1.net>.pt:<Q1.deck>=0.225" \
              "<Q2.fix>:checkpoints/<Q2.net>.pt:<Q2.deck>=0.225" \
              "mirror=0.25" "past=0.10" \
              "rule:tuned:lucario=0.10" "rule:iono=0.10"
```

Mechanics, verified against the working tree:

- The rule anchors MUST be spelled in full. Bare `rule:tuned` defaults
  the deck slot to "tuned" and crashes on the deliberately absent
  `decks/tuned.csv` — the M23 first-smoke failure. `rule:iono` is fine
  (decks/iono.csv exists).
- `past=` is tag-scoped to `ppo_m44_<P>_r1_it*.pt` (the M43-review fix)
  and silently inert until the leg's first promotion — expected, not a
  bug. `mirror=` re-resolves to the live work checkpoint every
  iteration; both ride P's own deck by construction.
- 25 iterations per leg — the M43 curve banked its best at it34 of 50
  and faded after (76.0 → 68.0 by it49); one compressed round gets the
  productive half.
- KL 0.1 is to the frozen `--start`, so the anchor is per-pair
  automatically.
- The promotion metric is `vs_teacher` = `rule tuned/lucario` REGARDLESS
  of `--eval-deck` (`load_teacher()` called with defaults,
  `rl/ppo.py:518`) — the same instrument M43 promoted on. Each leg
  measures its own pre-loop baseline; `vs_solver` is logged, never
  gates.
- No `--value-ckpt`, no `--phi-ckpt` anywhere (D4). Default race shaping
  at coef 0.0 is a parser-legal no-op.

**Inner kills — OPERATOR procedure on TensorBoard, not code** (there is
no entropy guard, KL bound, auto-stop or revert in `rl/ppo.py`; M43
reference values in parentheses):

- entropy (`train/entropy`) must not rise over the leg (M43:
  0.823 → 0.769, monotone-ish down);
- KL to the frozen start stays bounded (M43 peak 0.146);
- `vs_teacher` must beat the leg's own baseline within 15 iterations
  (M43: beaten at it9).
- Breach → operator kills the process. "Revert" is non-adoption: the
  pair's league net is `ppo_best_m44_<P>_r1.pt` if the leg ever
  promoted, else the pair keeps its seed. Nothing needs restoring.
- Ops: M43 measured ~2.5h at 50 it ⇒ ~1.25–1.5h per 25-it leg; Hermes
  stage-transition updates + hourly heartbeat per standing rule; watch
  for the intermittent workers-8 pool wedge (stall-detect and bounce,
  `docs/M43.md:257-264`).

**Round 2** (only if the round-1 champion ships early and time clearly
allows): re-run D2 with every pair's pool refreshed to the round-1 nets.
A pair whose D3 delta came in ≤ 0 is FROZEN (stays as opponent, stops
training). No round 3 without a new plan entry.

### D3 — selection + ship gate (one instrument, one battery)

Selection and the ship gate collapse into the SAME instrument — the one
M43 actually used, `scripts/gate_spec.py`, hash-disciplined:

- Three hashed specs `docs/specs/m44_<P>.json` (hashed at execution
  start): arm = pair P's league net under P's registered fix token;
  control = **the SHIPPED m43 net** `ppo_best_m43a_base`
  (`model-c-pkgz`, alakazam_v2_h4) for ALL three; beds = the 25-bed
  roster verbatim from `docs/specs/m43_laneA_base.json`; n=400/cell,
  seed 1, bars pass ≥ +2.0pp pooled / kill ≤ 0.0.
- `gate_spec.py hash → run → decode` per spec; ~20k games ≈ 11 min per
  spec at the pinned rate, ~35 min for all three. The decode refuses
  mismatched hashes/n — the pre-registration teeth.
- **Champion = highest pooled delta vs the shared control.** The ship
  gate is the champion's own PASS. One number answers both questions:
  "who won the league" and "did the league add anything on top of M43"
  — for every pair including K, whose spec is automatically
  "K-evolved vs K-as-shipped".
- The selection is deliberately NOT raw head-to-head — "agent+deck with
  the highest winrate" must not reward a deck-matchup lottery (K beats
  O ≠ K is the better ship). The head-to-head grid is a DIAGNOSTIC:
  3 cells (K–O, K–G, O–G) × n=800 via `matchrunner play` (~2,400 games,
  ~2 min). No ranking tool exists (`rl/rank.py` is not one) and none is
  built.
- Optional diagnostic, only if slack: regenerate `data/m39_live_mix.json`
  (`scripts/m39_live_mix.py`; episodes.parquet is present) and read the
  live-mix family weights against per-bed rates. It cannot be a second
  verdict: `m40_decide.py` reads a different runs layout than
  `gate_spec.py` writes, and its pool excludes the topgrim beds. THE
  VERDICT IS THE POOLED gate_spec NUMBER.

### D4 — critics (the E0 law, compressed)

Correction from r1: the E0 **pass** bar is matched-pair ≥ 0.62 AND
within-game variance share ≥ 0.10; **0.58 is the kill bar**
(`scripts/m40_e0_value.py:57-58`). r1's transplant branch is CUT for
M44: no transplant harness exists (M43 merged checkpoints by hand), the
four E0 corpora are box-local, and M43 already measured the answer —
wide body + retain_b head → E0 0.559, FAIL, "the head is body-specific;
transplanting it loses the ranking."

Therefore: **K trains on its own head** (it just trained under GAE);
**O and G run plain outcome-reward GAE from their own (weak) BC heads**,
diaried as such. No Φ anywhere: "no valid Φ exists yet" stands until a
candidate's OWN head passes E0 — at the pass bar, not the kill bar.

## What this is measuring (the honest question)

Does competitive pressure from co-trained opponents produce a stronger
ship than M43's frozen-pool PPO — on the same roster, same bars, same
control? The league is also the first true test of pair G: whether OUR
pilot on the ladder's dominant deck can beat the ladder's own grim
pilots (the beds).

## Kill criteria / stop rules

- Phase R kill: a seed that cannot be recovered removes its pair
  (descope ladder K > G > O > D); if K's seed is unrecoverable, M44
  stops and escalates to Piotr — there is no league without the
  champion and no gate without the control.
- Seed floor kill (D1) before any training is spent.
- Per-leg inner kills (D2) — operator-enforced; a killed leg means
  non-adoption, the pair keeps its seed.
- Gate kill ≤ 0.0pp vs the m43 ship = the league added nothing; diary,
  no ship, carry the roster to BACKLOG.
- Standing law: offline = screening, live decides. One ship max.

## Cost (M43-measured, new box, OMP-pinned ~31 games/s on batteries)

| item | games | wall |
|---|---|---|
| Phase 0: ingest + live reads | — | ~15 min |
| Phase R preflight (M43 box / rebuild) | — | ~5 min / **hours → descope** |
| seed floor (3 pairs × 2 anchors × 400) | 2.4k | ~10 min |
| one PPO leg (25 it × 400 + 5 evals × 400) | ~12k | ~1.25–1.5h |
| the round (3 legs, sequential) | ~36k | ~4–4.5h |
| selection + gate (3 specs × 25 beds × 400 × 2) | 60k | ~35 min |
| head-to-head diagnostic (3 × 800) | 2.4k | ~2 min |
| export + ship_verify + qc_battery + Piotr review | — | ~1h + review |
| **total** | **~100k** | **fits 2026-08-13 evening IFF Phase R is clean** |

## Non-goals (pre-registered)

- No simultaneous both-seat gradient updates (v1 is iterated BR).
- No new decks / deck search — the roster is built from nets we hold.
- No dragapult pair (held-out evaluator law).
- No event-bonus rewards; no Φ without an E0-passing own head.
- No AZ/PSRO machinery — 3 pairs × 1 round is a pilot of the idea, not
  a population algorithm.
- No new league/grid tooling — the diagnostic is a 3-cell
  `matchrunner play` loop; no meta-weighted verdict this milestone.

## Ship ritual (unchanged, binding)

Offline gate PASS → export → `scripts/ship_verify.py` (with
`--gate-arm` = the champion's spec token) → `scripts/qc_battery.py
--prefix m44_qc_<pair>` (all working sample agents + previous ship
tarball mirror leg) → **STOP for Piotr's manual replay review, explicit
go, and deck confirmation** → ship commit includes the `MODELS` dict
entry in `notebooks/model_monitor.ipynb` (+ `DECK_META` if the champion
deck is new to the monitor — grim_live would be).

## Open items for Piotr's review

1. Roster: is pair G (our first grim pilot) in? Highest-upside,
   highest-novelty seat. Pair D default OFF — confirm.
2. Selection metric: r2 RECOMMENDS the unweighted pooled gate_spec delta
   vs the m43-ship control (the actual M43 instrument, hash-disciplined,
   selection and gate in one battery). This SUPERSEDES r1's
   meta-weighted recommendation, which named a decoder whose input file
   is missing and whose layout is incompatible with the gate runner.
   Confirm.
3. 25 it × 1 round stands per the amendment; round 2 only on early ship
   (r1's "2 rounds" item is moot).
4. Does the ogerpon seat earn its ~1.5h of the compressed window? The
   league gives it exactly the headroom opponents Lane B's carry-forward
   asked for — but it is also the first pair the descope ladder drops.
   Keep or drop.
5. Phase 0 = live reads of BOTH 55464234 and 55464395 once n≥45
   (standing rule: forensics of the previous ship starts every
   milestone; the ingest gap means neither is readable locally yet).
6. NEW — champion weight custody: 55464234's weights were never
   committed (the tracked bundle now carries 55464395's). They exist
   only on the M43 box and on Kaggle. Recommend committing the m43a
   export npz (or an md5-pinned `ppo_best_m43a_base.pt`) BEFORE M44
   touches anything — it is the live champion, pair K's seed, and the
   D3 control.
7. NEW — confirm M44 executes on the M43 box. A fresh checkout (this
   one) cannot run Phase R → D3 as-is: no seeds, no beds, no E0
   corpora, no qc_beds.
