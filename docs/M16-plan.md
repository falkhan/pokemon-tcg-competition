# M16 plan: replay forensics → option-identity encoding

Status: **active 2026-07-18; diary [M16.md](M16.md).** Forensics ran against
this stub's questions; the replays indicted something bigger than any focus
point: **option-identity blindness** (PLAY options don't encode which card
they play, ATTACK options don't encode which attack — see the diary). M16 =
fix the encoding (v3o), re-collect, retrain, measure, ship. The focus
points below were triaged: starter defect largely fixed by M15 (small
lever); trainer/ability exploration is DOWNSTREAM of the blindness (the net
can't prefer what it can't distinguish); the 10k retrain question stands.

## Session opening moves

1. Download the 54793851 replay set (`rl.kaggle_ingest` episode listing +
   forensics; `notebooks/model_monitor.ipynb` for score/win-rate curves).
2. **Loss investigation** — extend the M14 forensic protocol (behavioral
   diff, scratchpad forensic script) to answer the M15 watch-list
   (docs/M15.md): did attaches-to-saturated drop from plan2's 0.63? Do
   committed setup lines correlate with held cards now? Then classify the
   remaining loss causes and rank by frequency × fixability.
3. Apply findings to the next iteration (labels / encoder / teacher —
   whatever the replays actually indict).

## Points of focus (user-set, 07-18)

### 1. Retrain with the v3 encoder at volume?

Open question: full retrain on the new encoder with a **10k-game
mixed-opponent batch** (self / buddy `ext:` / rule / solver), rather than
M15's 800-game warm-start increment. Sub-questions for the plan:
- Hand-aware VALUE head first? `osv3_setupval2` is still 12-id; the
  collector is already v3-default, so a 10k safari (`m13_collect.sh
  --target 10000`) regenerates the value on hand-aware states — the queued
  overnight job. Value → teacher → policy ordering matters.
- 10k games is a >2h batch → **Piotr sign-off** (standing rule).
- Does the 800-game increment vs full retrain difference show up in the
  replays (i.e. is the hand embedding under-trained — hand ids were
  zero-init and only saw 30k decision states)?

### 2. Explore/exploit: trainer cards, abilities, attack variety

The net plays what the teacher labeled; rarely-used trainer cards /
abilities / non-default attacks get few labels → never explored → never
learned. Candidate levers to evaluate (measure, don't assume):
- **Plan-level exploration with greedy execution** (the M11 law — τ /
  dirichlet exist in `plan_iter collect`; exploration below plan level was
  a measured dead end).
- **Oversampling rare prompt classes** in collection (the ATTACH_FROM
  oversample precedent) — same idea for trainer-card and ability prompts.
- **Usage-rate instrumentation first**: per-card-class action rates from
  replays + self-play (trainer play rate, ability activation rate, attack
  distribution vs board-best) so "starts using trainers efficiently" has a
  number and a gate.
- Card-fact gaps: abilities/attacks the combat table under-describes can't
  be scored by teacher OR net (the Solrock/Lunatone CONDITIONAL_ATTACKS
  precedent) — audit coverage before blaming exploration.
- Respect the dead-end list: no override-style consumers, no inference-time
  guards on mostly-right prompt classes; changes go through labels/teacher.

### 3. Starter selection: Solrock-first even with Riolu + Mega Lucario ex in hand

User-observed live defect (07-18): the agent ALWAYS leads Solrock, even when
the opening hand holds Riolu + Mega Lucario ex — the stronger starter line
(energy-attach target that evolves into an attacker that trades up vs weak
actives). Investigate why, and bias toward the stronger setup — it also
gives the cleaner reward signal to learn from. Hypotheses to test against
replays + labels, in rough order:
- **Teacher preference**: does the teacher itself label Solrock-first on the
  opening placement prompt? If so, it's a promote/positioning-scoring defect
  (ties into the wrong-attacker audit below) and no amount of student
  training fixes it — fix the teacher, regenerate labels.
- **Blind starter decision**: the opening active choice is exactly a
  hand-conditional decision. The policy encoder sees the hand only since
  M15 (and the hand embedding is barely trained); the VALUE head is still
  12-id hand-blind — a hand-aware value (focus point 1) may be the actual
  fix. Check whether the shipped net's starter choice varies with the hand
  AT ALL (replay forensics: starter distribution conditioned on opening
  hand contents).
- **Label scarcity**: one starter prompt per game → ~800 labels total in
  plan_m15; if the teacher is even mildly Solrock-biased the net has
  essentially no counter-examples (connects to focus point 2's
  rare-prompt oversampling).
Instrument first: starter-choice distribution + win rate by starter from
the replay set, same from self-play, teacher's own choice on the same hands.

## Carried-over candidates (not yet scoped)

- Wrong-attacker positioning: teacher promote/retreat scoring audit (logged
  M14, still open).
- Plan-vocabulary gap: setup lines map to null plans — extend the Plan
  tuple with setup descriptors (develop/attach targets), the M14
  architecture item.
- Clustering bug (open since M7.5).

## Constraints (standing)

Observation-run ship policy (technical gates enforced, win-rate
informational); campaign bar 0.55 vs `solver:lucario` + meta co-gate against
the RE-PINNED baselines (plan0c 0.357, osv3h_plan1 0.384, meta 0.419); >2h
runs and RL-core changes need Piotr; measurement protocol + result-decoding
in `.claude/skills/measure-agent`, pipeline commands in
`.claude/skills/train-ship`.
