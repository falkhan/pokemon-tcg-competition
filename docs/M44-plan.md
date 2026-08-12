# M44 — the Kaggle sim: competitive PPO league across (pilot, deck) pairs — DRAFT for Piotr's review

**Pre-registered:** DRAFT 2026-08-12 — bars and specs to be hashed at
execution start, after Piotr's review of this draft. Nothing below is a
registered bar yet.

**AMENDMENT (2026-08-12, Piotr): the competition ends 2026-08-16 — M44
compresses to ONE round** (3 legs ~6h + league + gate), champion ships
2026-08-13 evening at the latest so it accrues 2+ days of ladder games.
D2's "2 rounds pre-registered" is superseded: round 2 runs only if the
round-1 champion ships early and time clearly allows. Also shipped
alongside (Piotr's call, same date): `m43b_oger_wide` on one of the day's
spare slots — the Lane B live A/B the ceilinged offline battery couldn't
resolve.

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
provide: a full PPO leg is ~2.5h, a 38k-game battery ~25 min).

## Why now

- M43 proved the substrate: honest PPO improves the strongest BC start
  (+3.72pp z=5.93) with stable inner dynamics. One arm, one deck.
- The M43 A-base leg trained against FROZEN opponents (beds + past
  selves). The live ladder is not frozen — it is other adapting agents.
  A co-training league is the cheapest honest approximation of that.
- We now hold trained/clonable starts for several archetypes (the M43.1
  rebuild): the alakazam PPO champion, the exact ogerpon net, and BC
  clones of grim (90k-row corpus — the ladder's biggest family), plus
  thinner wall/arch/garchomp/rocket corpora.
- Lane B's carry-forward said ogerpon needs opponents with headroom, not
  more corpus — a league where its opponents ALSO improve is exactly that.

## Design

### D0 — form (settled in draft, challengeable)

**Iterated best response, NOT simultaneous both-seat learning.** Each
round, ONE pair trains while the others sit frozen in its opponent pool;
pairs take turns. Rationale: (a) zero collector surgery — the pool
mechanism (`--opponents "model-pz:ckpt:deck=w"`) already does this;
(b) simultaneous two-seat updates make both policies non-stationary
learning targets — the known cycling failure of naive co-evolution; the
collector also only records the learning seat, and that is a feature.
True simultaneous self-play (both seats recorded, one net) remains what
`mirror=` already provides WITHIN each leg. v2 territory if M44 pays.

### D1 — the roster (pilot, deck) seeds

| pair | start checkpoint | deck | standing |
|---|---|---|---|
| **K** (kazam) | `ppo_best_m43a_base.pt` | alakazam_v2_h4 | live champion, sub 55464234 |
| **O** (oger) | `m41_ogerpon.pt` | ogerpon | exact live net (sub 55265105) |
| **G** (grim) | `m39_bc_grim.pt` (val .642, 90k-row corpus) | grim_live | NEW learner — the ladder's biggest family (~49.5%) gets a pilot of ours for the first time |
| D (stretch, default OFF) | `m40_bed_garchomp_d1.pt` | garchomp c7b3253f | only if K/O/G legs come in under budget |

Dragapult stays the held-out evaluator everywhere (collector already
hard-refuses it in training pools; it remains legal in gate beds).

**Seed floor (kill):** every seed must score ≥0.25 vs the roster's rule
anchors (tuned + iono, n=400) before round 1 — a pair too weak to
threaten anyone teaches nothing and pollutes the league.

### D2 — the rounds

Round r (sequential under the worker cap, OMP-pinned, GPU updates):

```
for pair P in K, O, G:                      # ~2.5h each
  rl.ppo --start <P.current> --learn-deck <P.deck> --eval-deck <P.deck>
    --iterations 25 --games-per-iter 400 --workers 8 --eval-every 5
    --kl-coef 0.1 --value-ckpt <per-pair critic, see D4> --device auto
    --tag m44_<P>_r<r>
    --opponents  <the OTHER two pairs' current nets+decks, ~0.45 total>
                 "mirror=0.25"  "past=0.10"
                 <rule anchors tuned/iono ~0.20>
```

- 25 iterations per leg (half of M43's 50 — the M43 curve peaked at
  it34 of 50; with two rounds each pair gets 50 total with a re-anchor
  between).
- KL anchor + `past=` are per-pair, tag-scoped (the M43-review fix).
- Inner kills per leg, same as M43: entropy must not rise; KL bounded;
  the promotion metric must beat the leg's own baseline within 15 iters
  — else the leg stops and the pair keeps its previous net.
- **2 rounds pre-registered.** A pair whose league rating fails to
  improve in a round is FROZEN (stays as opponent, stops training). No
  round 3 without a new plan entry.

### D3 — the league read + selection

After each round: round-robin grid, all pairs + anchors, n=800/cell
(pinned ≈ 25 min), decoded to standings. THE SELECTION METRIC IS NOT raw
head-to-head — "agent+deck with the highest winrate" must not reward a
deck matchup lottery (K beats O ≠ K is the better ship). Champion =
highest **meta-weighted win rate on the M40-style 25-bed roster** (the
same instrument every ship since M40 has used), with the head-to-head
grid reported as a diagnostic beside it.

**Ship gate (hashed at execution):** `m44_champion.json` — the league
champion vs control `ppo_best_m43a_base` (the live best) on the 25-bed
roster, n=400/cell, pass ≥ +2.0pp pooled / kill ≤ 0.0. If the champion IS
pair K's evolved net, the control stays the SHIPPED m43 net — the gate
then reads "did the league add anything on top of M43".

### D4 — critics (the E0 law)

E0 gates every critic, per M43's finding that value heads are
body-specific: K inherits its own trained value head (it just trained
under GAE); O and G warm-start from `m39_retain_b` (E0 0.723) ONLY if
key-shapes transfer AND a transplant E0 on their own bodies clears 0.58 —
else that pair runs plain outcome-reward GAE from its own (weak) head,
diaried. No Φ anywhere: still "no valid Φ exists yet" until a candidate's
OWN head passes E0.

## What this is measuring (the honest question)

Does competitive pressure from co-trained opponents produce a stronger
ship than M43's frozen-pool PPO — on the same roster, same bars? The
league is also the first true test of pair G: whether OUR pilot on the
ladder's dominant deck can beat the ladder's own grim pilots (the beds).

## Kill criteria / stop rules

- Seed floor kill (D1) before any training is spent.
- Per-leg inner kills (D2) — a killed leg reverts the pair.
- League regression rule (D2) — frozen after one flat round.
- Gate kill ≤ 0.0pp vs the m43 ship = the league added nothing; diary,
  no ship, carry the roster to BACKLOG.
- Standing law: offline = screening, live decides. One ship max.

## Cost (new-box numbers, measured in M43)

| item | games | wall |
|---|---|---|
| seed floor checks | 3×800 | ~15 min |
| one PPO leg (25 it × 400 + evals) | ~11k | ~1.5-2h |
| one round (3 legs) | ~33k | ~5-6h |
| league grid per round | ~10k | ~25 min |
| 2 rounds + gates + QC | — | **~2 days** |

## Non-goals (pre-registered)

- No simultaneous both-seat gradient updates (v1 is iterated BR).
- No new decks / deck search — the roster is built from nets we hold.
- No dragapult pair (held-out evaluator law).
- No event-bonus rewards; no Φ without an E0-passing own head.
- No AZ/PSRO machinery — 3 pairs × 2 rounds is a pilot of the idea, not
  a population algorithm.

## Open items for Piotr's review

1. Roster: is pair G (our first grim pilot) in? It is the highest-upside,
   highest-novelty seat. Pair D (garchomp) default OFF — confirm.
2. Selection metric: meta-weighted roster win rate (recommended) vs pure
   round-robin head-to-head — confirm the recommended.
3. 25 iterations/leg × 2 rounds vs 50×1 — the draft prefers 2 shorter
   rounds so opponents refresh mid-campaign. Challenge welcome.
4. Does a Lane-B-style ogerpon seat earn its slot, given Lane B's
   inconclusive close? Draft says yes (the league gives it exactly the
   headroom opponents the carry-forward asked for) — cheap to drop.
5. Phase 0 of M44 = live read of 55464234 once n≥45 (standing rule:
   forensics of the previous ship starts every milestone).
