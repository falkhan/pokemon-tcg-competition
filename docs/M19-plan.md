# M19 — plan stub (to refine after 54817441 replay forensics)

**Status:** STUB + forensic round 1 RESULTS (see §below, 2026-07-19 ~08:00).
Refine ship decision after 24-48h more settle data.
**Live context at stub time:** M16 (`osv3o_plan1`) settled **519.8**; rules
bundle historical best 548.6. plan5's first target: beat 519.8.

---

## Lessons learnt in M18 (carry into everything)

1. **Silent exception swallows hide milestone-killing bugs.** M17's entire
   regression traced to one bare `except: pass` around a width-mismatched
   encoder call — 10k games collected with a dead teacher and no signal.
   Rule now enforced in code: count errors, print the first traceback, and
   print a loud `WARNING` when a supplied component produces zero effect
   (`vs_errors` / `SETUP=0` pattern in `rl/plan_iter.py`). Apply the same
   pattern to any future optional-component wiring.
2. **Single-seed pinned gates are landmines.** The champion's meta pin 0.534
   was seed-0 luck (seed 1: 0.395); the pinned floors were stale too (the
   champion itself failed them). ALL gate baselines are now 2-seed pooled
   (re-pinned in `.claude/skills/measure-agent`); floors are comparative
   (candidate vs current champion, same seeds), not absolute bars. Never
   compare a candidate's fresh number to a historical single-seed pin.
3. **Offline disagreement-weighted DAgger is a measured dead end** (plan4:
   0.278 vs 0.4275 champion screen, −15pp). W=10 upweighting on the
   teacher's own state distribution amplifies label inconsistency instead of
   fixing the student. Residue NOT yet dead: smaller W, and ei-mode
   collection weights (student-distribution states; infra landed —
   `EI_DISAGREE_WEIGHT` column is written automatically in ei mode).
4. **The value teacher works and matters** (first positive transfer evidence
   for the M13 value arc): fixed 20-id teacher → SETUP 2843/800 games →
   +6pp mirror (plan3 0.484, best neural ever). But **SETUP-heavy data
   trades meta for mirror** (plan3 meta 0.378) — the two axes fight.
5. **plan_m15 is the meta regularizer.** Every meta collapse (M17, plan3)
   lacked it; adding it back recovered +11pp (plan5 0.487). Dose-response is
   non-monotonic (plan6 = m18+m15 only → 0.463) — keep BOTH m16 and m15 in
   future mixes until something better is measured.
6. **Offline metrics still anti-correlate with live** (3rd confirmation):
   plan4's weighted objective looked fine offline; plan3's val_acc 0.784 was
   the best honest number and its meta tanked. Only live series arbitrate.

## 🎯 FOCUS POINT (Piotr, 07-19): deck swap × leaderboard score

**Primary M19 question (Piotr): does the deck change move the leaderboard
score?** Comparison anchors: M16 settled 519.8, M15 422.7, rules 548.6.
Split live win rate by opponent archetype (water vs lucario matchups may
flip). The model_monitor notebook now has a deck-grouped chart (chart 4,
replay-audited deck identity + ace sprites) tracking exactly this.

**RESOLVED (2026-07-19, replay audit — one live replay per submission,
`notebooks/model_monitor.ipynb` cell 2):** every ship M7.4a→M15 played
LUCARIO live (13x Fighting Energy, 4x Mega Lucario ex); **M16 (54801291,
ep 86653767) ALREADY played the water deck live** (35x Water Energy, 4x
Mega Abomasnow ex), same as M18 (ep 86776210). Outcome B holds:
- The **M15→M16 live jump (422.7 → 519.8) is deck-confounded** — deck swap
  and option-identity encoder shipped in the same submission.
- **M16 → M18 is the clean like-for-like A/B**: same deck (abomasnow),
  different net (plan1 vs plan5) — plan5's settle vs 519.8 is a pure NET
  signal.
- The deck variable's own effect needs a deliberate experiment (pairing
  matrix below), since no submission pair isolates it going forward.

## ⚠️ Lesson 0 (post-ship discovery, Piotr 07-19): we ship the ABOMASNOW deck

Since M16, neural bundles ship `decks/kyogre.csv` — actually a water deck
with **4x Mega Abomasnow ex** as its ace — NOT lucario. Cause chain:
`DEFAULT_DECK="kyogre"` fossil in `tcg/shipping.py` (M1/M2 champion pairing,
never updated) + `build_submission.sh` only passes `--deck` when given +
the `--deck lucario` flag dropped during M16's dry-build firefight + the
omission codified into the train-ship skill command. Verified via dist/
tarball audit: M11/M14/M15 = lucario; M16 + M18 = kyogre/abomasnow.
Full trace in docs/m18.md.

**Consequences to resolve in M19 (decide with Piotr before any code/flag
change):**
- M16's 519.8 live jump is CONFOUNDED (option-identity fix vs deck swap on
  the same submission). M15 (lucario) 422.7 → M16 (abomasnow) 519.8.
- All offline gates measure the `:lucario` pairing — a different artifact
  than what ships. Either measure the shipped pairing
  (`model:<ckpt>:kyogre`) or ship the measured one (`--deck lucario`).
- The net has always TRAINED on lucario states; it pilots abomasnow
  zero-shot live. A deliberate experiment matrix: {plan5, plan1} ×
  {lucario, kyogre} offline + the live scores we already have.
- Update train-ship skill §3 + DEFAULT_DECK once the pairing decision is
  made (do NOT silently "fix" to lucario — abomasnow may be the better
  live deck; the 519.8 says it might be).

## 🔬 Replay post-mortem round 0 (07-19, Piotr-triggered — first 7 plan5 episodes)

Piotr's read from watching replays: deck is weak, Mega ex not evolved when
available, too many losses from having no basic to field. Quantified against
the first 7 plan5 episodes (+ last 18 plan1 episodes as the same-deck
baseline; scratchpad script `m18_replay_postmortem.py` pattern, replays
cached in data/kaggle/raw/):

1. **Record so far: plan5 3W-4L** (plan1's last 18: 9W-9L). n=7 — direction
   only, refine with tomorrow's episode volume.
2. **Bench-out is THE loss mode — CONFIRMED, and it's structural.**
   **4 of 4 plan5 losses ended with an empty bench, having fielded ≤2
   pokemon in the entire game** (losses at turns 2-7: lose the active →
   nothing to promote → game over). plan1 baseline: 7 of 18 losses look the
   same. Root cause is deck construction, not (only) piloting: `kyogre.csv`
   runs **6 basics / 60 cards** (4 Snover + 2 Kyogre; P(no basic among any
   5 draws) ≈ 58%) and **35/60 = 58% energy** — the second basic simply
   never arrives. No mulligan log lines and no no-basic opening hands in
   the sample: the engine setup appears to guarantee a starter; the
   fragility is entirely about the SECOND basic.
3. **Mega Abomasnow ex under-use — CONFIRMED, with a refined mechanism.**
   plan5 fielded the Mega in 4/7 games (plan1: 13/18). plan5 spent **220
   steps with Mega ex dead in hand, and in only 62 of those was a Snover on
   board** — i.e. the dominant blocker is "no Snover on board to evolve
   onto" (not drawn / not played / KO'd before evolving), and in 1 of 5
   games WITH the opportunity plan5 still never evolved. 3 of plan5's 4
   losses never fielded the Mega at all — games it fields the Mega it
   mostly wins (3 of 4).
4. **Net×deck mismatch compounds everything:** plan5 trained on 100%
   lucario-deck states; every water-card option-identity embedding is
   effectively untrained; it pilots this deck zero-shot.

**Levers this opens (rate in the morning with settle data):**
- **Deck engineering — a never-touched lever.** We have NEVER designed a
  deck; we inherited `kyogre.csv` (M1-era sample) and `lucario.csv`. Fixes
  as obvious as "fewer energies, more basics" (e.g. +Abomasnow line /
  +basics, −10 energy) are untested. Needs offline support: matchrunner
  already accepts arbitrary deck csvs.
- **Ship the measured pairing** (plan5 + lucario, one `--deck lucario`
  flag) — reverts the accidental deck swap; the 2-seed offline battery
  already measured exactly this artifact.
- **Train on the water deck** (self-play collection with kyogre.csv) if the
  deck itself is worth keeping — decide AFTER the leaderboard settle
  verdict on whether abomasnow-zero-shot ≥ lucario-measured.

## Forensic round 1 (the first task tomorrow — start here)

Download 54817441 replays (`rl.kaggle_ingest`, monitor notebook lists
episodes) and check the M18 watch-list **knowing the agent plays
kyogre/abomasnow live** (re-read watch-list items 3-4 in that light —
solrock-mirror expectations don't apply as written):

1. **Score trajectory vs 519.8** — is plan5 actually better live? (Settles
   over ~a day; fresh subs start at μ=600, not a result.)
2. **Random-floor failure mode** — plan5's one measured regression (0.815 vs
   0.895 vs random:kyogre, ~3sd on s1). Find live games vs weak/erratic
   opponents and identify HOW it loses winnable positions (energy waste?
   wrong attacker? passing on lethal?). This is the most concrete defect
   lead we own.
3. **SETUP-plan behavior** — first live agent trained on working
   value-teacher labels. Do early-game board developments look coherent
   (bench building, energy placement before turn 32)?
4. **Solrock mirror matchups** — meta says +3.3pp vs the 0.90-weight
   archetype; confirm in live episodes vs lucario-family opponents.

## 📊 Forensic round 1 RESULTS (2026-07-19 ~08:00, 73 fresh episodes)

Data: `rl.kaggle_ingest refresh` for subs **54817441** (M18, plan5+abomasnow,
n=37) and **54817813** (M18.1, plan5+lucario, n=36); baselines re-profiled the
same way — **54801291** (M16, plan1+abomasnow, n=44) and **54793851** (M15,
osv3h+lucario, n=31). Aggregation over `rl/postmortem.py` `classify_end` +
`audit_flags` (scratchpad script `m19_replay_forensics.py`; parquet snapshot
alongside it). Caveat: both plan5 subs have only ~37 games (started 07-18
~23:00, still settling from μ=600); loss-MODE findings are robust, W/L and
score deltas are directional.

### 1. Scoreboard (watch item 1): plan5 beats 519.8 on NEITHER deck

| sub | pairing | record | score @ n≈37 | trend |
|---|---|---|---|---|
| 54817441 M18 | plan5 + abomasnow | 14W-23L (.38) | **456** | falling (last 10: 480→456) |
| 54817813 M18.1 | plan5 + lucario | 16W-20L (.44) | **493** | hovering 485-503 |
| 54801291 M16 | plan1 + abomasnow | 20W-24L (.45) cached sample | **519.8** settled | anchor |
| 54793851 M15 | osv3h + lucario | 12W-19L (.39) cached sample | 422.7 settled | anchor |

**FOCUS POINT answered — the deck moves the score AND interacts with the
net.** Same net (plan5): lucario 493 vs abomasnow 456 (~+40 for the trained
deck). Same deck (abomasnow): plan1 519.8 vs plan5 456. plan5 is *better on
its training distribution* (493 ≫ M15's 422.7 on lucario) and *worse
zero-shot* (456 < plan1's 519.8 on abomasnow) — consistent with the
lucario-specialized M18 training mix (value-teacher SETUP + m15/m16 data)
trading generality for on-distribution strength. Honesty note: the M16↔M18
same-deck W/L gap (.45 vs .38, n≈40 each) is within seed noise — the SCORE
gap is large but not settled; treat "plan5 < plan1 live" as probable, not
proven, until settle.

### 2. Abomasnow axis: bench-out is the DECK, not the net (round 0 upgraded)

M18: **21/23 losses BENCHED-OUT** (15/23 fielded ≤2 pokemon all game; median
loss turn 7; 7 losses ended with all 6 of the opponent's prizes untaken).
M16 baseline: **22/24 losses benched-out** (18/24 fielded ≤2). Identical loss
anatomy across both nets ⇒ structural: 6 basics / 35 energy in `kyogre.csv`.
The flip side is new: **excluding bench-outs, the deck is near-unbeatable —
M16 20W-2L, M18 14W-2L in games where a second basic showed up.** Fixing the
second-basic drought (deck surgery, not training) converts a coin-flip deck
into the strongest thing we've fielded.

### 3. Lucario axis (M18.1): deck functions, loses the PRIZE RACE

Bench-out essentially gone (1/20 losses; 0 losses with ≤2 fielded). Losses
are combat losses: we end holding 3-6 prizes while the opponent is at 1-3;
the 12 "unclassified" ends are final multi-prize KOs on our Mega (3-prize
swing from the last observed state). Median loss turn 14-15 (vs 7 on
abomasnow). Plus **3 DECK-OUT losses** in long games (M15 also had 2 —
chronic on this deck). Live mirror vs Mega Lucario opponents: **1W-4L**
(watch item 4: the meta-gate +3.3pp does NOT show up live; n=5 only).
Random-floor leak (watch item 2) is real but minor: 3 losses per sub to
opponents rated <480.

### 4. Chronic pilot defects — 3 net generations, same flags (pipeline gap)

Per-game `audit_flags` rates (M18.1-lucario / M15-lucario / M16-abom):

| flag | M18.1 | M15 | M16 | example |
|---|---|---|---|---|
| attach-off-racer | 1.11 | 0.94 | 0.16 | energy to active while Mega Lucario ex unloaded |
| hand-discard (Carmine) | 0.50 | 0.39 | — | ep 86782442 t7: Carmine threw away **2x Mega Lucario ex** |
| fetch-dead-evolution | 0.36 | 0.19 | 0.41 | fetched Hariyama, no Makuhita in play or hand |
| trainer-hoarded | 0.25 | 0.58 | 0.20 | Cyrano playable 20 turns, never played (abom) |
| evolve-left | 0.14 | 0.06 | 0.02 | t5/t7 ended with Mega Lucario ex evolution offered, unplayed (eps 86789498, 86796760) |

These rates are flat-to-worse across generations ⇒ the collect→BC pipeline
never targets them: the expert either shares them or they live in rare states
BC averages away, and no offline gate measures them. Piotr's round-0
"Mega not evolved when available" is confirmed on lucario too (`evolve-left` +
Carmine discarding the Mega). Refined mega stats: Mega dead-in-hand ~40
steps/game on both decks; in ~26% of those steps the base WAS on board
(370/1399 abom, 495/1483 lucario) and we still didn't evolve that step.

### Implication ranking for M19 (pre-settle, re-rate with tomorrow's data)

1. **Deck engineering is the biggest untouched lever** (§2): +basics/−energy
   surgery on kyogre.csv, or a designed deck; offline support exists
   (matchrunner takes arbitrary csvs). Attacks 50-57% of all abomasnow losses.
2. **Prize-race combat on lucario** (§3) is a net problem → targeted
   collection / defect-class oversampling (direction a) aimed at the flag
   classes in §4 (Carmine-with-evolutions-in-hand, attach target choice,
   evolve-now-vs-wait states).
3. **Ship-pairing decision** waits for settle: if M18.1 stalls <519.8, the
   live champion remains plan1+abomasnow — consider re-shipping plan1 while
   M19 trains.

## 🛠 New instrumentation (2026-07-19): net-internals agent logging

Kaggle stores per-step `{duration, stdout, stderr}` for OUR OWN agent
(private to the team; opponent's is 403), fetchable via
`KaggleApi.competition_episode_agent_logs`. The replay JSON carries the full
option menu + chosen action but NONE of the net's scores — so from the next
ship onward the agent explains itself through this side channel:

- `submission/main.py` now writes one `NN|{json}` line per decision to
  stderr (~80 bytes, ~5 KB/game): obs step `s` (the replay join key), turn,
  context, chosen indices `a`, per-option logits `sc`, and on each turn's
  plan commit the plan index `p` + plan logits `psc`. Opt-out:
  `PKM_AGENT_LOG=0`. Failures self-report as `NN|ERR|` lines (no silent
  swallows — M18 lesson 1).
- `rl.kaggle_ingest agent-logs --episode <id>` fetches + caches logs to
  `data/kaggle/logs/` (seat autodetected from episodes.parquet);
  `fetch_agent_logs()` is the API.
- `rl/postmortem.py report` now shows a "net log" coverage line and, with
  `--decisions`, prints each option's logit + the plan-commit line —
  directly answering "why did the net pick this card"; `parse_net_log()`
  is the parser for aggregate studies (underplayed cards, margin-vs-error
  correlations on the §4 flag classes).
- NOTE: logs exist only for episodes played AFTER the next submission;
  M18/M18.1 episodes have empty stderr. Gates/parity/isolation all pass
  with logging on (verified; 64-decision local game parses 64/64 recs,
  11 plan commits).

## ✅ M19 SCOPE DECIDED (Piotr, 07-19): defect repair, deck engineering DEFERRED

Piotr's replay observations (confirmed by forensics + code trace):
(1) >1 energy attached to Solrock (1-cost attacker) — wasted attachments;
(2) no "save the active" logic — a damaged Mega never retreats into an
energized bench. Root causes and fixes (all landed, 469 tests green,
diary docs/M19.md):

| layer | over-attach | no-retreat |
|---|---|---|
| teacher (greedy) | saturated tier flat 600 → now 600+bonus−150×surplus (`tcg/constants.py`, both pilot twins) | escape tier needed lethal-on-board → new tier 2b: 2+-prize active at ≤0.4 hp with ready bench → 1450 |
| solver | never overrides these rows (greedy labels them) — unchanged | leaf has no HP term — unchanged (follow-up if labels stay retreat-free) |
| net visibility | ATTACH option = printed features only → extra block now carries [target energy, gap, saturated] | RETREAT = bare one-hot → extra block now [damage frac, prizes at risk, bench-ready] |
| measurement | new `[over-attach]` postmortem flag: live M18.1 = 0.75/game | retreat offered 21×/g, taken 0.11×/g live vs teacher 0.75/g — 7× imitation gap |

**Key measured insight:** teacher A/B (40+40 mirror games) shows the teacher
was already mostly clean — the live defects are the STUDENT failing to
imitate on information it could not see. The encoder features (width-94
unchanged, ATTACK-only extra slots reused per option type) are the primary
lever; recollect+retrain required to exploit them.

Pipeline (started ~10:20): collect 800 expert games seed 3 → train
plan_m19+m16+m15 warm from plan5 → gates vs plan5 pins (2-seed; do NOT
re-run plan5 baselines under the new encoders — contaminated).

## Candidate directions (pre-rating, refine after forensics)

| # | Direction | Why | Cost |
|---|---|---|---|
| a | Targeted collection oversampling the forensic defect class (M14-loop, still never closed) | attacks the actual live losses | 1 collect+train |
| b | ei-mode collection with student-distribution disagreement weights (infra ready) | the un-dead half of the DAgger idea | 1 collect+train |
| c | Meta-axis data: collect vs meta_v2 archetype opponents (kangaskhan/drakloak as `--opponents` specs) | meta is the leaderboard axis; mixed-opponent infra exists | 1 collect+train |
| d | `--vs-margin` sweep (e.g. 500/1000) to modulate SETUP density | tune the mirror↔meta trade found in M18 | cheap re-collects |
| e | 10k-scale re-collect with the FIXED teacher (M18h revisited) | volume was never tested with a live teacher | >2h — needs sign-off |
| f | **Deck engineering** (post-forensics R1): fix kyogre.csv second-basic drought (+basic lines, −energy), gate offline with matchrunner deck csvs | 21-22 of ~24 abomasnow losses are bench-outs on BOTH nets; non-bricked games go 34W-4L pooled | deck csv + offline battery, no training |
| g | Defect-class option features / oversampling for §4 flags (Carmine-discard, attach-off-racer, evolve-left) | 3 net generations, flat error rates — pipeline blind spot | encoder/collect work |

## Constraints / process (unchanged + new)

- Gates: beat plan5's 2-seed pins (mirror 0.4294 pooled n=800; meta ~0.498
  2-seed; floors comparative). Campaign bar ≥0.55 still stands.
- Hermes Telegram updates every stage + heartbeat >30min runs
  (`runs/m18_reporter.sh <tag> <log> <proc_pattern> <done_regex>`).
- Incremental diary (docs/M19.md once the milestone starts); kills are
  first-class results.
- Runs >2h and rl/ppo.py changes need Piotr sign-off. No mid-run edits; no
  broad pkill (a self-matching `pgrep | xargs kill` bit us today — always
  exclude your own shell or use exact PIDs).
- jsonl decoding: 0 = side-a WIN.
