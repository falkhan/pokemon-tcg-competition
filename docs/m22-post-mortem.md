# M22 post-mortem — how the shipped agent plays, and what it teaches M23

_Written 2026-07-21 morning, after the first ~39 live games of the corrected M22c-RL ship._

## The ships

| sub | what | status |
|---|---|---|
| **54864190** | M22c-RL (`ppo_current_m22cRL.pt`, lucario) — teacher upgraded solver→sample-agent | **live, the subject of this post-mortem** |
| 54864089 | same net, **kyogre deck shipped by mistake** (M18.1 repeated) | dead arm — ignore its score |
| 54846434 / 54849475 | B2 / B3 (M21) | live, accruing |
| 54836093 | M20 champion | live context |

The deck accident and its durable fix (`build_submission.sh` now hard-errors without `--deck`)
are recorded in `M22.md` (00:45 entry) and `ship-deck-verify` memory. Not re-litigated here.

## Live state (2026-07-21 ~08:00, n=39)

```
arm            n     W-L    liveWR   95% CI          score
54864190      39   18-20    0.474  [0.317, 0.631]    548.0
54846434 B2   44   19-25    0.432  [0.284, 0.580]    531.5
54849475 B3   44   17-27    0.386  [0.239, 0.534]    541.9
54836093 ch   45   22-23    0.489  [0.343, 0.635]    579.3
```

Every CI overlaps every other. M22c-RL vs B3 is +8.8pp against an MDE of 30.8pp at this n —
`live_monitor compare` refuses the read (needs ~507/arm), and that refusal is correct.
**No live ranking claim is made anywhere in this document.**

Agent health: **zero agent errors in 39 games** (the one ERROR row is the opponent, 54830134,
crashing at step 5 — we were awarded the win).

## How the agent plays — live forensics (54864190)

### Per-archetype record

| opponent archetype | W–L | WR | mean steps |
|---|---|---|---|
| mega_lucario_ex+solrock (mirror) | 14–12 | 0.54 | 111 |
| mega_kangaskhan_ex+crustle | 3–7 | 0.30 | 151 |
| drakloak+dreepy | 1–1 | — | 130 |

### The end-state anatomy: two different games are being played

Final prize counts over all 39 games (from each game's last observed state):

- **Mirror losses (n=12): 7 blowouts (≥5 of our prizes untaken), 3 outright shutouts.**
  Mirror **wins** (n=14) are symmetric: 7 opponent-blowouts, 3 opponent-shutouts.
  The mirror is a setup-tempo coin flip — whoever establishes first snowballs. Consistent with
  M22's diagnostic ("vs lucario we lose less often but more decisively — a setup/tempo failure").
- **Kangaskhan losses (n=7): 3 shutouts, prizes_left = [6,6,6,4,4,3,3]. Kangaskhan wins (n=3)
  are all narrow grinds (opponent down to 1–2 prizes).** The asymmetry — narrow wins, crushing
  losses — is what a structurally losing matchup looks like, not variance.

### Anatomy of the kangaskhan shutouts (postmortems 87201455, 87168640)

Both quick losses have the same shape, and it is **Gap A live on the ladder**:

1. Mega Lucario ex promoted into the active with 1–2 energy and never able to attack;
   opponent's 80hp Crustle chips it 120/turn (440→320→200→80→KO, 3 prizes cashed at once).
2. We promote a **second** Mega and it dies the same way. Game ends 0–6 on prizes.
3. Hand is **empty from ~turn 4** — top-decking one card a turn — while in the other shutout
   three copies of Carmine (draw supporter) sat playable for 5 turns and were never played.
4. `attach-off-racer`: energy attached to Lunatone while the active Mega sat unloaded;
   Carmine (when played, t2) discarded the Hariyama evolution line.

Caveat, per M22b discipline: `trainer-hoarded` and `attach-off-racer` are **not** loss-enriched
overall (see flag table below) — these are descriptive anatomies of specific shutouts, not
statistically established causes. What IS solid: the shutout shape itself (never took a prize,
Mega ground down while unable to attack) against the tank archetype.

### Behavioral counters (corrected M22b metrics, 1350 MAIN states)

| metric | 54864190 | B3 (for shape, not ranking) |
|---|---|---|
| gust strict | 3/40 = 0.075 | 3/43 = 0.070 |
| retreat strict | 2/28 = 0.071 | 2/50 = 0.040 |

Behaviorally the new agent is **indistinguishable from B3** — expected: 10 PPO iters from the B3
checkpoint with a swapped teacher, and its out-of-loop number didn't move (+1.68pp, inside
5.48pp MDE). The teacher swap changed the training signal, not (yet) the policy.

### Win/loss flag contrast (18W vs 20L, Bonferroni α=0.0083)

No flag is enriched in losses; `trainer-hoarded` is nominally win-associated (0.67 vs 0.25,
p=0.021, does not survive correction). Same picture as B2/B3/champ: the loss-only taxonomy
would have named six "failure modes"; the contrast kills all of them. The whole-game confound
caveat applies (winning boards have more turns/energy).

### Powered offline read on the weak cell — tail meta-eval

The live kangaskhan number (0.30, n=10) is unreadable alone; the M22b retargeted tail gate is
its powered twin. 2-seed run on `ppo_current_m22cRL.pt` (n=120/deck/seed):

| cell | seed 0 | seed 1 | pooled n=240 |
|---|---|---|---|
| mega_kangaskhan_ex+crustle | 0.783 [0.694, 0.873] | 0.833 [0.744, 0.923] | **0.808** |
| drakloak+dreepy | 0.625 [0.536, 0.714] | 0.708 [0.619, 0.798] | **0.667** |

(logs: `runs/m22cRL_meta_tail_s{0,1}.log`; tail_score 0.753/0.810 — tail-only, never a
whole-field number.)

### 🔴 Finding: the tail gate does not measure the live matchup it names

Offline kangaskhan **0.808** (pooled, n=240) vs live kangaskhan **0.30** (n=10):
Fisher exact **p = 0.00097** despite the tiny live n. Honest caveats: the cell was selected
*because* it looked bad live (selection on the test data), and live n=10 is hypothesis-grade.
But the direction and size are exactly what the M22 endogeneity analysis predicts: the offline
cell is the kangaskhan *deck* piloted by **our own solver**; the live cell is the same deck
piloted by **other teams' trained agents**. Same story one rung down: drakloak+dreepy
(solver-piloted) 0.667 offline, while `rule:dragapult` — a related line piloted by the Pokémon
Company's agent — sits at 0.19.

**Pilot quality dominates archetype, in every cell, not just the mirror.** The tail gate is a
training-distribution readout; it is not evidence about any live matchup. The only
live-credible instruments remain `rule:dragapult` and live accrual itself. (Also note the
B2/B3 live kangaskhan cells are 4–3 each — the "M22c-RL is bad vs kangaskhan specifically"
reading is n=10-grade; the offline↔live chasm is the durable part.)

## What M22 established (condensed from `M22.md` — the durable spine)

1. **The ~30pp out-of-loop collapse.** Every pilot in our lineage — neural, neural+search,
   rule+search — sits at 0.17–0.22 vs `rule:dragapult`; the Pokémon Company sample agent sits
   at ~0.50 **on our exact deck**. The deficit is the pilot, not the deck, not the net
   specifically, not within-turn search.
2. **Four generations of mirror gains bought zero out-of-loop strength** (BC 0.204 → B3 0.172
   vs dragapult, flat within MDE, while mirror rose 0.429→0.509). The training loop is anchored
   to weak endogenous opponents: `solver:lucario` (45% of B3's training, double-counted in the
   mixture spec) is beaten 2:1 by the sample agent.
3. **No validated offline→live predictor** (mirror→live ρ collapses to −0.037 controlling ship
   order). Mirror is contaminated (in-training) and endogenous; the meta gate was 90% the mirror
   gate by deck identity; `rule:dragapult` is the only doubly-clean instrument and **must stay
   out of every training pool**.
4. **Cause A confirmed (p<0.0001): threat blindness.** Our opponent model is `opp_ttk` + two
   booleans about ACTIVES; dragapult's bench spread lands exactly where we never look (median
   loss: ⅓ of damage on our bench, vs literally 0.000 median in the lucario matchup).
5. **Two levers cleanly nulled:** C1 within-turn search (+2.25pp unresolved, zero headroom —
   6.7× compute changed 0 actions) and C2 whole-board threat tiebreak (p=0.548, kill-switch
   honored). **One lever under-tested:** the strong teacher — one 10-iter leg, still climbing
   vs the teacher (15%→18%, 0.365 at n=200), not converged. That is M23 Phase 1.
6. **The instruments held.** MDE refusal gates, the become-what-you-train-against self-check,
   and the dragapult floor each did their job; nothing shipped as a "result" that wasn't one.

## What the live forensics ADD to the M23 picture

1. **Gap A is not a dragapult-only story.** The Mega-over-exposure / can't-power-an-attacker
   failure produces live shutouts against kangaskhan-crustle, a tank archetype now ~15% of the
   field (post-drift). Fixing Gap A pays on the ladder, not just on the out-of-loop floor.
2. **The mirror is a tempo coin flip, and the live field is ~76% mirror.** Blowouts dominate
   both sides of our mirror record. Setup consistency (draw-engine usage, not stranding
   supporters, benching a backup attacker early) plausibly moves live WR more than any
   mirror-gate delta we can measure offline. The empty-hand-by-turn-4 pattern in the shutouts
   is the concrete thread to pull.
3. **The M22c-RL teacher swap did not change behavior** (gust/retreat/flags ≈ B3). Ten iters
   was signal-starved; convergence (M23 Phase 1) remains the live question.
4. **Kangaskhan cell is the one live matchup with a structural-loss shape** — watch it in the
   tail gate as Phase 1's secondary readout (it is exactly the cell the retargeted meta-eval
   now powers).

## Process notes (kept honest)

- The deck-ship accident recurred (M18.1 → M22c) before the hard-error guard landed. Cost: one
  dead live arm. The guard + md5 ritual are now in `build_submission.sh` and memory.
- Seat handling bit again during THIS analysis: deriving our seat from `submission_id_0` order
  gave the opponent's postmortem; `episodes.parquet.our_seat` is the only trustworthy source
  (already the recorded lesson in M22.md — it held).
- The bench-damage-share diagnostic from M22 EOD was ad-hoc and not saved as a script; if M23
  needs it again it must be rebuilt (violates the save-pipelines rule — flagging, not hiding).
