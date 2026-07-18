# M19 — plan stub (to refine after 54817441 replay forensics)

**Status:** STUB, written 2026-07-19 at M18 close. Refine once submission
54817441 (`osv3o_plan5`) has live episodes to analyze.
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

## Candidate directions (pre-rating, refine after forensics)

| # | Direction | Why | Cost |
|---|---|---|---|
| a | Targeted collection oversampling the forensic defect class (M14-loop, still never closed) | attacks the actual live losses | 1 collect+train |
| b | ei-mode collection with student-distribution disagreement weights (infra ready) | the un-dead half of the DAgger idea | 1 collect+train |
| c | Meta-axis data: collect vs meta_v2 archetype opponents (kangaskhan/drakloak as `--opponents` specs) | meta is the leaderboard axis; mixed-opponent infra exists | 1 collect+train |
| d | `--vs-margin` sweep (e.g. 500/1000) to modulate SETUP density | tune the mirror↔meta trade found in M18 | cheap re-collects |
| e | 10k-scale re-collect with the FIXED teacher (M18h revisited) | volume was never tested with a live teacher | >2h — needs sign-off |

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
