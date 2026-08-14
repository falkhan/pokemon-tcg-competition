# M44 — competitive PPO league across (pilot, deck) pairs — EXECUTION PLAN r4

**r4 (2026-08-13, Piotr's directive):** the league grows to FOUR pairs
(mega lucario ex joins the roster), selection moves from the gate delta
to an **ELO ranking computed from pilot-vs-pilot play**, and the **TWO
highest-ELO pairs ship** (not one). The pipeline must periodically print
a ranking table. Training must demonstrably address all three loss
modes — deck-out, bench-out, prizes — with per-cause telemetry (new code,
Step 3g). r3's pausability requirement and math audit stand; the r1/r2
review narrative lives in git history (commit 9279aea).

**Deadline — SHIP TODAY (Piotr, 2026-08-13):** two new submissions go to
Kaggle TODAY. Motivation: M43 fell short of the campaign's 1000-ELO goal
— the wide alakazam ship sits ~823 live, ogerpon ~650, and 55464234's
only datum (hour-1 public 883.4) is not on a 1000 trajectory. The
competition ends 2026-08-16, so today's ships are the last that accrue
2+ days of ladder games. Every step below is scheduled against a
same-day ship, net of pause time; the compression rule (Step 5) cuts
iterations before it cuts legs, and a cut leg only un-trains a pair — it
stays in the ranking on its seed. The ship-today commitment does NOT
waive QC or Piotr's review — conflicts escalate (Kill criterion 4),
never resolve silently.

## Settled decisions (pre-registered; hash specs at execution start)

| decision | value |
|---|---|
| form | iterated best response — ONE pair trains per leg, the other three frozen in its pool (no simultaneous both-seat updates; `mirror=` provides within-leg self-play) |
| roster | **K** `ppo_best_m43a_base.pt` / alakazam_v2_h4 / `model-c-pkgz` · **L** (NEW) best-of-3 seed pick / **lucario** (the mega lucario ex deck, = the tuned sample agent's and the 0.899-weight meta deck) / `model-pz` · **O** `m41_ogerpon.pt` / ogerpon / `model-pz` · **G** `m39_bc_grim.pt` / grim_live / `model-pz` — pair D (garchomp) stays OFF |
| reward | outcome ±1 ONLY — it already penalizes ALL three loss modes (deck-out, bench-out, prizes) symmetrically, which is exactly "learn to prevent losing" without reward-hacking surface. Per-cause penalties/bonuses stay a NON-GOAL (flip only on Piotr's explicit call). What r4 adds is per-cause MEASUREMENT (Step 3g) so each leg's diary shows whether every loss mode actually declines |
| selection | **ELO (Bradley-Terry) fitted on the league round-robin** — 4 pairs + 2 frozen rule anchors (tuned/lucario, iono) as calibration players, tuned FIXED at R=1000 to pin the scale across ranking points. Ranking table printed + Hermes'd at every ranking point (baseline, after every leg, final) |
| ship | **the 2 highest-ELO pairs at the final ranking, each with its own deck** — two exports, two QC batteries, two MODELS entries, Piotr's review/go/deck-confirm for BOTH |
| gate role | `scripts/gate_spec.py` vs the m43a-ship control is DEMOTED to a diagnostic, run on the two ship candidates only — it informs Piotr's go (is the candidate better than the incumbent on the 25-bed ladder proxy?), it no longer selects. Bars pass ≥ +0.02 / kill ≤ 0.0 are read advisorily; a kill triggers escalation, not silent no-ship |
| promotion metric (in-leg) | `vs_teacher` (rule tuned/lucario, `load_teacher()` defaults) — each leg beats its own pre-loop baseline; `vs_solver` logged, never gates |
| critics | K trains on its own head (just trained under GAE); L, O, G run plain outcome-reward GAE from their own heads, diaried as such. No Φ anywhere — "no valid Φ exists" stands until a candidate's OWN head passes E0 at the PASS bar (matched-pair ≥ 0.62 AND within-game variance share ≥ 0.10; 0.58 is the KILL bar, `scripts/m40_e0_value.py:57-58`) |
| held-out | dragapult trains nowhere (collector hard-refuses it in pools); legal in gate beds and QC |
| non-goals | no per-cause reward terms, no new decks beyond the roster, no dragapult pair, no AZ/PSRO machinery, no simultaneous both-seat gradients, no meta-weighted verdict |

## Audit (2026-08-13) — math + RL best practices (r3, extended in r4)

Verified against the working tree; **no formula corrections were
required** in the trainer:

- `compute_gae` (`rl/ppo.py:195-221`) is textbook GAE(γ=0.99, λ=0.95):
  δ_t = r_t + γV(s_{t+1}) − V(s_t) with V(s_T)=0, A_t = δ_t + γλA_{t+1}
  backward, returns = A + V. Correct.
- `ppo_update` (`rl/ppo.py:246-366`) is the standard clipped surrogate:
  additive loss (policy + 0.5·huber value − entropy·coef), ratio clip
  0.2, grad-norm clip 0.5, invalid options masked before softmax and
  excluded from entropy, KL(π_new ‖ π_frozen-start) anchor matching its
  docstring. Correct.
- Accepted (registered, not changed): global-once advantage
  normalization; huber value loss without value clipping; no LR anneal.
  All match what M43's positive was measured on — do not change
  mid-campaign.
- **ELO model (new in r4):** P(i beats j) = 1 / (1 + 10^((R_j−R_i)/400)).
  Fit by Bradley-Terry MLE on the round-robin grid via the Zermelo/MM
  iteration on π_i = 10^(R_i/400):
  π_i ← W_i / Σ_{j≠i} [ n_ij / (π_i + π_j) ],
  draws counted as half a win to each side (the `rl/league.py`
  convention), then rescale so tuned = 1000. Report ±SE from the
  observed Fisher information, I_ii = (ln10/400)² Σ_j n_ij p̂_ij(1−p̂_ij).
- **Statistical registrations:**
  - Ranking cells n=200 ⇒ per-cell CI ±6.9pp; each pair's rating pools 5
    opponents × 200. Intermediate rankings are progress prints, not
    verdicts; the FINAL ranking runs n=800/cell (per-cell ±3.5pp) and is
    the selection instrument.
  - Winner's curse: "top-2 of 4 by fitted ELO" inflates the winners'
    ratings by selection; diary final ELOs with this caveat and never
    quote them as unbiased strength.
  - The league ELO is a CLOSED-population measure (pilot-vs-pilot +
    2 anchors); it is NOT a ladder predictor ([[no-validated-live-
    predictor]] stands). The gate diagnostic on the top-2 (25-bed
    battery, SE(Δ) ≈ 0.71pp pooled, z≈2.83 at the +2.0pp advisory bar)
    is the ladder-proxy read that informs Piotr's go. Offline =
    screening, live decides.
  - Promotion evals n=200 ⇒ ±6.9pp: in-leg promotion is a screen.
  - Seed floor n=400 ⇒ CI ≤ ±4.9pp (±4.2pp at the 0.25 bar).

## Pausability map (Piotr will pause/resume on this box)

| stage | pausable? | mechanism |
|---|---|---|
| PPO legs (Step 5) | **after Step 3 lands** | sentinel-file pause + `--resume` (state saved every iteration; SIGKILL loses at most the in-flight iteration, ~4 min) |
| ranking grid (Steps 5/6) | already, by design | cells cached keyed by (net-md5 pair, decks, n, seed); a rerun replays only missing/stale cells |
| gate diagnostic (Step 6) | already | `rl.matchrunner.run_pairs` header-checked per-cell resume — kill anytime, rerun the SAME command |
| seed floor, QC | atomic | minutes each; rerun if killed |
| `plan_iter collect` | NO resume | **not used in M44** |

Never run two PPO legs concurrently (`data/ppo/` shards are shared).
Pausing leg X and launching leg Y, then resuming X later, is safe —
every iteration wipes and recollects shards.

---

## Step 0 — forensics first (standing rule)

- [ ] `uv run python -m rl.kaggle_ingest refresh` — local
      `data/kaggle/episodes.parquet` tops out at sub 55450580; neither new
      ship (55464234 alakazam, 55464395 ogerpon) is ingested.
- [ ] Live read of 55464234 once n≥45; read 55464395 vs incumbent
      55265105 the same way. Diary both, including replay forensics of
      losses — specifically tag each observed live loss by cause
      (deck-out / bench-out / prizes): that breakdown seeds the Step 3g
      telemetry's first hypotheses.

## Step 1 — artifact preflight

**Verified on THIS box 2026-08-13** — it IS the M43 box: seeds
(`ppo_best_m43a_base.pt`, `m41_ogerpon.pt`, `m39_bc_grim.pt`), bed
checkpoints, `docs/specs/m43_laneA_base.json` (25 beds), E0 corpora,
both 2026-08-12 ship tarballs, `decks/grim_live.csv` and
`decks/lucario.csv` all present. Remaining actions:

- [ ] Record md5 of all seeds (incl. pair L's, once picked in Step 4) in
      the diary.
- [ ] Rebuild `dist/qc_beds/` (missing): `uv run python
      scripts/m39_build_qc_beds.py` — qc_battery silently skips its
      loss-family legs without it.
- [ ] `data/m39_live_mix.json` absent — regenerate only if the optional
      live-mix diagnostic is run; never a verdict.
- Contingency: a seed that fails md5/`_load_model` un-trains its pair
  (leg cut; the pair stays ranked on whatever seed is recoverable). K's
  seed unrecoverable → recover from
  `dist/submission_neural_20260812_183326.tar.gz`, else STOP and
  escalate.

## Step 2 — champion weight custody (before anything touches checkpoints/)

- [ ] Commit an md5-pinned copy of 55464234's weights (the m43a export
      npz or `ppo_best_m43a_base.pt`) — never committed, exists only on
      this box and Kaggle, and it is pair K's seed AND the gate-diagnostic
      control. m43a WAS shipped, so this respects the ship-only commit
      rule.

## Step 3 — CODE CHANGES (mandatory before any leg)

Change `train()` and the collector's game-end accounting ONLY —
`compute_gae`/`ppo_update` stay untouched (the `tcg/ppo.py` parity twin
and its tests must not move).

**3a–3f: pausable PPO** (unchanged from r3):

- [ ] **3a. State persistence.** `_save_train_state(path, model, opt,
      next_it, best_wr, meta)` → `checkpoints/ppo_state_<tag>.pt` holding
      `{model: cpu_sd, opt: opt.state_dict(), next_it, best_wr, meta:
      {start, iterations, learn_deck, opponents}}`. Write `<path>.tmp`
      then `os.replace` (atomic). Call at the END of every iteration,
      after the promotion block.
- [ ] **3b. `--resume` flag** (store_true, requires `--tag`). Refuse if
      the state file is missing or stored `meta` disagrees with the CLI
      (`start`/`iterations`/`learn_deck`). Restore model sd →
      `model.to(dev)` → build AdamW → `opt.load_state_dict` (PyTorch
      re-homes optimizer state). Loop `range(next_it, iterations)`.
      **Skip the baseline re-measure and use stored `best_wr`** — no bar
      drift. Skip the `start → ppo_best` copy. KL `ref_model` rebuilds
      from `--start` (frozen); anneals key off `it` and continue
      correctly.
- [ ] **3c. Pause sentinel.** Top of each iteration: if
      `checkpoints/ppo_pause_<tag>` exists → save state, print `PAUSED
      before iter N`, exit 0. Pause = `touch
      checkpoints/ppo_pause_m44_<P>_r1`; `--resume` auto-deletes the
      sentinel at startup.
- [ ] **3d. Register:** RNG state not persisted — resumed runs are not
      bit-identical (cuda already isn't). `past=`/`mirror=` survive
      resume by construction.
- [ ] **3e. Test.** `tests/test_ppo_resume.py`: round-trip a tiny
      OptionScorer + AdamW (one step taken) through save/load — exact
      tensor equality, `next_it`, `best_wr`, meta-mismatch refusal. Run
      with the parity suite:
      `uv run pytest tests/test_ppo_resume.py tests/test_ppo.py -v`.
- [ ] **3f. Acceptance smoke** (~5 min): `--iterations 2
      --games-per-iter 8 --workers 2 --eval-every 1 --eval-games 4 --tag
      m44_resume_smoke` → sentinel-pause after iter 0 → `--resume` →
      confirm start at iter 1, NO baseline re-measure line. Delete smoke
      artifacts.

**3g: loss-cause telemetry** (NEW — the "prevent losing by deck-out /
bench-out / prizes" instrument):

- [ ] Classify every finished collection game's terminal state in the
      collector worker: loser's cause ∈ {deckout, benchout, prizes}
      (deck empty at forced draw / no Pokemon in play / opponent took
      all prizes — exact engine fields confirmed at implementation
      against the wrapper; the `me.prize` remaining-prizes convention is
      `rl/collector.py:532-536`). Return per-game cause with the
      existing per-game stats.
- [ ] Aggregate per iteration in `train()`: the learner's losses AND
      wins split by cause. TB scalars `loss_cause/{deckout,benchout,
      prizes}` + `win_cause/...`, one stdout line per iteration.
      (Win-by-deckout is a strategy — the deck-out race — worth seeing
      rise or fall too.)
- [ ] Registered read (per leg, diaried): no loss cause may RISE over a
      leg while overall win rate improves — a falling total hiding a
      rising deck-out share is exactly what outcome-only reward can
      mask, and it's the trigger for Piotr to reconsider the per-cause
      penalty non-goal in v2.
- [ ] Test: unit-test the classifier on 3 synthetic terminal states (one
      per cause) in `tests/test_ppo_resume.py` or a sibling.

**3h: league ranking script** (NEW — `scripts/m44_league.py`):

- [ ] Round-robin driver over a roster file (`docs/specs/m44_roster.json`:
      pair → {net md5-pinned, deck, fix token} + the 2 anchors): every
      unordered player pair = one cell, both seats, played via
      `rl/matchrunner` primitives (workers ≤ 8). `rl/league.py` was
      evaluated and rejected — M7-era PlackettLuce over a stale anchor
      field; only its draws-as-half-wins convention is kept.
- [ ] Cell cache keyed by (md5(netA), md5(netB), deckA, deckB, n, seed)
      under `data/m44_league/` — reruns replay only missing/stale cells;
      this is also the pause story (kill anytime, rerun).
- [ ] Bradley-Terry fit + table print per the Audit formulas: rank,
      pair, deck, ELO ±SE, W-L-D, Δ since previous ranking point. Table
      goes to stdout, the diary, and Hermes at every ranking point.
- [ ] Loss-cause columns in the table (from the same games): losses by
      deckout/benchout/prizes per pair — the league-level view of 3g.
- [ ] Test: BT fit on a synthetic 3-player grid with known rates
      recovers the constructed rating order; tuned-anchor pinning = 1000
      exact.

## Step 4 — seeds: pair L pick + seed floor (kill bar, before any training)

- [ ] **Pair L seed pick** (~15 min): no modern lucario pilot exists
      (`bc_lucario*.pt` are July v1-era). Candidates, all piloting
      `lucario`: `ppo_best_m43a_base.pt`, `m41b_wide_prod.pt`,
      `m28_winners.pt`. n=400 each vs the tuned anchor; highest wins,
      md5 recorded. (KL anchors to the frozen start per-pair, so a
      cross-deck start is legitimate; the seed floor below still
      applies.)
- [ ] **Seed floor:** every seed (K, L, O, G) scores ≥ 0.25 vs each rule
      anchor (tuned/lucario, iono), n=400 per anchor, via `matchrunner
      play` (decode per the measure-agent skill). A failing seed
      un-trains its pair (leg cut) — the pair still enters the ranking
      on its seed unless the seed also fails to load.
- [ ] **Baseline ranking** (ranking point 0): `m44_league.py` over all 4
      seeds + anchors, n=200/cell — the league's starting table, printed
      + Hermes'd.

## Step 5 — the round (4 legs, sequential, workers ≤ 8)

Leg order **K → L → G → O** (champion continuation first, Piotr's new
seat second). Per pair P, fresh tag `m44_<P>_r1`; opponents = the other
three pairs' CURRENT league nets (updated as legs complete):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m rl.ppo \
  --start <P.seed>.pt \
  --learn-deck <P.deck> --eval-deck <P.deck> \
  --iterations 25 --games-per-iter 400 --workers 8 \
  --eval-every 5 --eval-games 200 \
  --kl-coef 0.1 --device auto --tag m44_<P>_r1 \
  --opponents "<Q1.fix>:checkpoints/<Q1.net>.pt:<Q1.deck>=0.15" \
              "<Q2.fix>:checkpoints/<Q2.net>.pt:<Q2.deck>=0.15" \
              "<Q3.fix>:checkpoints/<Q3.net>.pt:<Q3.deck>=0.15" \
              "mirror=0.25" "past=0.10" \
              "rule:tuned:lucario=0.10" "rule:iono=0.10"
```

(Weights sum to 1.00. Rule anchors spelled IN FULL — bare `rule:tuned`
crashes on the absent `decks/tuned.csv`. `past=` is tag-scoped, inert
until first promotion; `mirror=` re-resolves each iteration; both ride
P's own deck. KL 0.1 anchors to the frozen `--start` ⇒ per-pair
automatically.)

- **After each leg:** the pair's league net becomes
  `ppo_best_m44_<P>_r1.pt` if the leg ever promoted, else the seed
  (non-adoption = revert). Then **ranking point**: `m44_league.py`
  n=200/cell (cache makes this only the updated pair's row, ~1 min) —
  table printed + Hermes'd. Subsequent legs pool the UPDATED net.
- **Inner kills — OPERATOR procedure on TensorBoard** (no guard code;
  M43 references): entropy must not rise (0.823 → 0.769); KL to frozen
  start bounded (peak 0.146); `vs_teacher` beats the leg's own baseline
  within 15 iterations (it9). Plus 3g: no loss cause rising while win
  rate improves. Breach → kill the process (exact PID); the pair keeps
  its seed.
- **Compression rule (pre-registered):** if wall-clock jeopardizes the
  same-day ship, FIRST cut remaining legs to `--iterations 15` (M43 beat
  its baseline by it9; ~55 min/leg), THEN cut legs in order O, G. A cut
  leg's pair stays in the ranking on its seed. K's and L's legs are
  never cut.
- Ops: ~1.25–1.5h per 25-it leg net of pauses; pause/resume per Step 3;
  Hermes update at every leg start/end + hourly heartbeat + immediate
  failure notification; watch the workers-8 pool wedge (stall-detect and
  bounce, `docs/M43.md:257-264`); diary metrics as they land, never
  retrospectively.

## Step 6 — final ranking, selection, ship diagnostic

- [ ] **Final ranking** (the selection instrument): `m44_league.py` over
      the 4 league nets + anchors at **n=800/cell** (cache upgrades the
      n=200 cells; ~12k games ≈ 7 min). Table printed + Hermes'd +
      diaried with the winner's-curse caveat.
- [ ] **Selection: the 2 highest-ELO pairs ship**, each with its own
      deck. No tie math beyond SE overlap: if #2 and #3 overlap within
      1 SE, escalate the choice to Piotr with the table.
- [ ] **Gate diagnostic on both ship candidates:**
      `docs/specs/m44_<P>.json` per candidate — arm = league net under
      P's fix token, control =
      `model-c-pkgz:checkpoints/ppo_best_m43a_base.pt:alakazam_v2_h4`,
      beds verbatim from `m43_laneA_base.json`, n=400/cell, seed 1, bars
      `{pass: 0.02, kill: 0.0}` read ADVISORILY. `gate_spec.py hash →
      run → decode` (~11 min each; resumable). Results go to Piotr with
      the QC replays — a kill (≤ 0.0 vs the m43a incumbent) triggers
      Kill criterion 4 escalation, not silent no-ship.

## Step 7 — optional diagnostics (never verdicts)

- [ ] Only with slack: regenerate `data/m39_live_mix.json`
      (`scripts/m39_live_mix.py`) and read live-mix family weights
      against the final table (which pairs' decks the live meta actually
      contains). The r3 head-to-head grid is superseded — the ranking IS
      the head-to-head, with rating math on top.

## Step 8 — ship ritual (binding, per candidate — run TWICE)

For EACH of the two selected pairs: export → `scripts/ship_verify.py`
with `--gate-arm` = that pair's spec token → `uv run python
scripts/qc_battery.py --prefix m44_qc_<pair>` (all working sample agents
+ qc_beds legs + previous ship tarball
`submission_neural_20260812_184453.tar.gz` as mirror leg; add a bespoke
`play_games` leg for behavior the pair specifically changed) → **STOP
for Piotr's manual replay review, explicit go, AND deck confirmation for
BOTH candidates** → each ship commit includes its `MODELS` dict entry in
`notebooks/model_monitor.ipynb` (+ `DECK_META` for decks new to the
monitor — grim_live and lucario would be).

## Kill criteria / stop rules

1. Preflight kill: unrecoverable seed → its leg is cut, pair ranks on
   whatever loads; K unrecoverable → STOP, escalate.
2. Seed floor kill (Step 4): a sub-0.25 seed un-trains its pair.
3. Per-leg inner kills (Step 5) — non-adoption, pair keeps its seed.
4. Ship-conflict escalation: a ship candidate whose gate diagnostic
   kills (≤ 0.0 vs the m43a incumbent), or #2/#3 ELO within 1 SE,
   collides with the ship-today directive → escalate to Piotr
   immediately with the numbers (his options: ship anyway on his
   explicit call, substitute #3, or hold an incumbent). Never resolve
   silently in either direction.
5. Standing law: offline = screening, live decides. Two ships max.

## Cost (M43-measured; net of pause time)

| item | games | wall |
|---|---|---|
| Step 0 ingest + live reads | — | ~15 min |
| Step 1 preflight + qc_beds rebuild | — | ~20 min |
| Step 2 custody commit | — | ~5 min |
| Step 3 code changes + tests + smoke (3a–3h) | — | ~1.5–2h |
| Step 4 L seed pick + seed floor + baseline ranking | ~6.2k | ~25 min |
| Step 5 round (4 × ~12k) + 4 ranking points | ~52k | ~5–6h (25 it) / ~3.5–4h (15 it) |
| Step 6 final ranking + 2 gate diagnostics | ~52k | ~30 min |
| Step 8 2× (export + verify + QC) + review | — | ~1.5h + review |
| **total** | **~110k** | **same-day evening ONLY with a morning start or the 15-it compression; Steps 3–4 must begin immediately** |
