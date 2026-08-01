# M38 post-mortem — sub 55146658 (m38_w9294_cont3 + gacfr3 + alakazam_v2_h4)

Forensic, 2026-08-01. Live cache refreshed (`kaggle_ingest refresh --subs
55146658`): 55 episodes, 1 self-validation excluded → 54 ladder games.
Tools: `scripts/live_postmortem.py`, `scripts/pm_extra.py`,
`scripts/m37_pm_probe.py` (band/racemode/adjudication probes reused as-is).

## Verdict: the offline +6.1pp DID NOT TRANSFER — implied ELO 659 (vs M37's 736), a live regression. The ELO ceiling was not touched. The winners fine-tune fixed its own family (mirror 3-5 → 7-2) but the ceiling is set by the wall/stall/grim loss mass, where the ship was flat (wall 1-5), actively harmed by the rule stack we knowingly kept (G5), or measured on a stale bed that lied (grim 6-2 → 1-6).

| sub | milestone | n | W-L | WR | avg_opp | implied ELO |
|---|---|---|---|---|---|---|
| 55011605 | M35 (gacf, alakazam_v2) | 55 | 32-23 | 0.582 | 759.5 | 817 |
| 55030954 | M36 (gacf, alakazam_v2_h4) | 52 | 26-26 | 0.500 | 608.2 | 609 |
| 55065484 | M37 (gacfr3, m28_winners) | 63→94 | 33-30 | 0.524 | 715.8 | 736 → **~719 now** |
| 55146658 | **M38 (gacfr3, w9294_cont3)** | 54 | 26-28 | **0.481** | **672.2** | **659** |

**The pool-drift control:** the M37 sub kept playing in the same window and
sits at **~719** (n=94, converged) while M38 ended at **662 and still
sliding**. This is not ladder deflation — the M38 agent live-performs
~50–60 ELO below its own parent in the same pool. Trajectory: climbed to a
peak of ~708 by game ~38, then a **2W-8L tail** (grim 738/766/838,
dragapult 853, fan-rotom stall, archaludon) dragged it back to 662.
WR by band: 600–699 **15-14 (0.52)**, 700–799 **6-8 (0.43)**, 800+ **0-3**.
M37 at the same 700-band was 18-19 (0.49) — we are not better where it
counts, and 800+ remains untouched (0-6 across both ships).

## The transfer audit (what the gate claimed vs what live delivered)

| bed | offline cont3 vs control | live family result | verdict |
|---|---|---|---|
| mirror | .579 vs .518 (+6pp) | **7-2** (M37: 3-5) | ✅ transferred — the corpus's own family |
| m28-lineage | .604 vs .525 (+8pp) | (same family as mirror) | ✅ |
| iono | .4375 vs .2988 (**+14pp**) | 0-1 (n=1) | ⚪ untestable — ~2% of live games |
| dragapult | .4512 vs .2800 (**+17pp**) | 2-1 | ⚪ n=3; the two loss-mass beds that drove the pooled +6.1 are ~7% of the live mix |
| grim | .731 vs .689 (+4pp, z+1.9) | **1-6** (M37: 6-2) | ❌ **bed lied** — M26-era 600-band clone vs live 738–838 pilots |
| wall | .320 vs .332 (flat) | **1-5** (M37: 3-6) | ✅ honestly predicted flat — and we shipped anyway with a stack G5 measured at −7pp here |
| rocket | .556 vs .616 (−6pp, watch) | 0 games | ⚪ moot — rocket absent at this band |
| tuned | .688 vs .693 (flat) | lucario 6-5 | ✅ flat as predicted |

**The pooled +6.1pp was real but mis-weighted.** It was dominated by
iono/dragapult (+14/+17pp), which are ~7% of live games at this band,
while the beds matching the actual live loss mass (wall, grim, stall) were
flat, stale, or absent. Re-weighted by the live opponent mix, the offline
delta was ≈0 — and the grim bed's sign was wrong. This is the campaign's
third bed-fidelity transfer failure (garchomp M35, wall-solver M37, grim
clone M38), now with a corollary: **an honest pooled gate on the wrong bed
mix is still a wrong gate.**

## Loss anatomy (28 ladder losses)

| mode | count | share | detail |
|---|---|---|---|
| **Deck-race lost** | **8** | **29%** | 5 classified deck-outs + 3 next-draw (deck≤1 at end). Wall ×4, stall ×2 (Hop's Trevenant, Fan Rotom), garchomp ×1, mirror ×1. **In 5 of 8 we were ahead, even, or one prize from winning**: ep89251935 (mirror) decked out at 5 prizes taken vs 1; ep89220709 (wall) ahead 3-0; ep89207539 (wall) ahead 2-0 at deck 1v1. Pure economy failure in winning positions ≈ 18% of all losses. |
| **Swept** (took ≤1 prize) | 12 | 43% | grim ×5 (all its losses), wall ×1, froslass-starmie, ogerpon, dud-fez-budew, lucario, dragapult-853, grim-738. Up from M37's 30% — the 700+ band sets up faster and closes before we develop. Policy-quality territory. |
| **Contested race lost** | 8 | 29% | lucario ×3, archaludon ×3, iono, mirror. The era's chronic mode (M35 39%, M36 35%, M37 30%). |

## Matchup ledger (54 ladder games, M37 in parens)

| family | W-L | note |
|---|---|---|
| lucario | 6-5 (8-5) | biggest volume; holds |
| mirror | **7-2** (3-5) | the winners fine-tune's home matchup — the one clear live win of the ship. 1 deck-out loss remains (ep89251935, lost at 5 prizes taken) |
| grim/marnie | **1-6** (6-2) | the flip that erased the milestone. 5 of 6 losses swept ≤1 prize; three at 738/815/838. M37 called it "self-resolved" — it was a band artifact, and the M26-era clone bed can't see it |
| crustle/tusk wall | **1-5** (3-6) | unchanged mode: 4 deck races lost by 12–20 cards, racemode3 firing on 100% of prompts (0 dud/fez uses) — same as M37, the leak is engine burn, not draw abilities |
| archaludon | 2-3 (4-4) | contested losses at 672–756 |
| stall (hop trevenant / fan rotom) | **0-2** (0-1) | both deck-outs. **Trigger blindspot confirmed in code**: `_RACEMODE_WALL_IDS` = {Crustle, Dwebble, Great Tusk, Terrakion} only; Hop's Trevenant sits in the pressure set whose margin-gated branch (racemode/racemode2) never shipped, Fan Rotom is in NO set ([plan.py:400](../rl/plan.py:400)) |
| dragapult | 2-1 (1-3) | the +17pp bed; the loss is an 853 sweep |
| starmie | 2-1 (2-0) | the loss a froslass-starmie sweep |
| garchomp | 1-1 (1-1) | the loss a deck-out |
| other (ogerpon, dud-fez, kyogre…) | 4-2 | ogerpon + dud-fez sweeps |

## Root causes, ranked (with pilot-component attribution)

1. **Engine deck-burn is structural and no shipped mechanism addresses it**
   (→ 29% of losses, 5 of them from winning positions). The BC policy's
   engine usage — Alakazam evolution digs, Enriching Energy (−4/attach),
   Rare Candy, Poké Pad, Dawn — burns ~2.5 cards/turn vs disciplined
   opponents' ~1.5 (M37 burn audit, unchanged). racemode3 demotes only
   dud/fez *draw abilities*, which were already unused (0 uses across both
   ships). The M37 rec #2 (racemode4: demote the real burn sources) was
   never built — M38 spent its cycle on the teacher-fix lane instead.
   Components: `rl/plan.py` O-rule layer (dig-only demote), the policy net
   itself (burn-heavy line preferences learned from replays).
2. **We shipped a rule stack G5 had just measured as actively harmful on
   the wall bed** (plain .323 vs gacfr3 .254, ~−7pp) for minimal-diff
   continuity. Live wall 1-5. The single-variable strip submit was already
   queued in BACKLOG ("Rule-stack strip") — the bill arrived before the
   follow-up. Component: `gacfr3` economy rules in `rl/plan.py`, tuned on
   the old net's behavior, now interacting badly with the new net.
3. **Stale beds gated the ship** (→ the grim 1-6 flip, 5 sweeps). The grim
   clone is an M26-era 600-band artifact; live grim/marnie pilots at
   738–838 are a different opponent. The drift law (re-pin every era) was
   applied to *numbers* but not to *bed construction* — the clone itself
   is two metas old. Same class of failure as M37's solver:wall strawman.
   Components: gate battery bed roster (`scripts/m38_gate.sh`), QC battery
   (tuned/iono/dragapult + mirror tarball — zero coverage of wall, grim,
   or stall, i.e. the actual loss mass).
4. **The winners corpus teaches in-family piloting, not the losing
   matchups.** 800–1058-band alakazam winner seats demonstrate how winners
   fly *this* deck — mirror 7-2 proves the vehicle works — but carry no
   anti-wall economy or anti-grim setup lines (wall bed flat offline,
   honestly). Fine-tuning on it could never move the loss mass it never
   saw. Component: corpus construction (`bc_m38_w9294`, winners-only,
   same-deck filter).
5. **Setup speed vs the 700+ band's aggro is the growing mode** (43%
   swept, up from 30%). Chronic under-play persists: supporters 10.7% per
   offer (unchanged since M35), END-with-playable-Poké-Pad 10.9%, telepath
   attach 56.9%. The one positive drift: ≥1-supporter turns 64.5% (M37:
   58.4%). Component: the BC net's ceiling — no rules fix this; it needs
   stronger/denser demonstrations or a non-imitation objective.
6. **Trigger blindspots on stall variants** (2 deck-out losses): Fan Rotom
   and Hop's Trevenant stall got zero conserve behavior. The BACKLOG's
   "Kangaskhan-only wall blindspot" generalized exactly as predicted.
   Component: `_RACEMODE_WALL_IDS` / the unshipped margin-gated pressure
   branch in `rl/plan.py`.

## Recent-change validation ledger

| change | verdict | evidence |
|---|---|---|
| w9294_cont3 fine-tune (M38) | ✅ in-family / ❌ ceiling | mirror 3-5 → 7-2; but implied ELO 659 vs parent's ~719 in-window — the gains live where the losses aren't |
| gacfr3 kept (Piotr's minimal-diff call, G5 harm known) | ❌ | wall 1-5, 4 deck races lost; G5's −7pp wall prediction is consistent with live |
| racemode3 (M37) | ✅ mechanical / ❌ strategic (2nd confirmation) | 100% trigger-true prompts in all 6 wall games, 0/240 demoted-draw uses — and races still lost by 12–20 cards |
| P2 `rl/plan.py` wins/concedes fix | ✅ no live anomaly | 0 adjudication surprises beyond the known next-draw deck-outs |
| h4 hammer (M36) | ⚪ | EH played 19% of offers; mirror recovered anyway — dead-card worry not supported |
| deckguard/conserve (M30) | ✅ holds | dud 0/85, fez 0/27 at deck≤6 |

## Watch items closed (pre-registered at ship)

1. **Rocket −6pp regression: MOOT** — zero rocket-family games at this
   band. The offline watch item measured a bed the live meta no longer
   serves.
2. **Wall unchanged (rules-lane territory): CONFIRMED** — 1-5, all the
   old modes. The m38_bc_wall clone bed (G4) remains the milestone's
   most valuable surviving instrument.
3. **Monitor row on submit: done** (MODELS dict, ship commit 2c54b50).

## BACKLOG cross-check (which parked ideas the live data now funds)

| BACKLOG item | live evidence this sample | verdict for M39 |
|---|---|---|
| **Rule-stack strip** (deferred from M38 by design) | wall 1-5 after shipping a stack measured −7pp on the wall bed | **PICK UP — highest ROI.** The cheap single-variable follow-up submit (cont3 plain) was pre-planned for exactly this moment; G5 data already exists |
| **Kanga-only wall blindspot** (M37 audit) | Fan Rotom + Hop's Trevenant stall: 0-2, both deck-outs, zero trigger coverage | **PICK UP, generalized** — widen wall/stall coverage or ship the margin-gated pressure branch (racemode2 code exists, unshipped) |
| **racemode4 burn-source demote** (M37 rec #2, never scheduled) | burn still ~2.5/turn; 5 deck-race losses from winning positions | **PICK UP** — now measurable on the real m38_bc_wall bed, which didn't exist when the idea was parked |
| **Universal margin raceconserve** (M36 parked / M37 rec #3) | mirror recovered 7-2 via the fine-tune; 1 residual mirror deck-out | Deprioritize as a mirror fix; its *mechanism* (margin gate on any opponent) is subsumed by the pressure-branch item above |
| **Band gate refresh** (M37 post-mortem) | `band_decode.py` weights now two metas stale | Cheap ride-along; do it with the next ship commit |
| **Gen-2 collect + value-net retrain** (deferred from M38) | 43% swept losses = policy-quality ceiling | Still the right long lane, but M38 proved the vehicle matters more than the labels — see rec 5 |
| **W_COUNTER / 2-ply prize exchange** | no over-greed signature live (Boss at opp_pz≤1: 1 play, a win) | STAYS PARKED — E0b's call stands |
| **setup_plans_late re-eval** | deck-out-adjacent behavior still visible | Stays behind gen-2 |
| **Stop-investing list** (grim deck-tech was ON it) | grim 1-6 | **REVOKE the grim entry** — "self-resolved" was a band artifact. The stop-list logic (no new evidence → don't resurrect) worked as designed; this is new evidence |

## Where the next lever is (M39 recommendations, ranked by loss mass × tractability)

1. **Harvest-refresh the bed roster BEFORE any measurement (P0, blocking).**
   The snowball harvest demonstrably works and this sample hands us the
   exact sub ids: grim/marnie winner seats at 738–838 (CoCoSh, yujinki,
   Hamachi…), fan-rotom and hop-stall seats, plus more 800+ mirror seats.
   Build fresh BC clones: **grim @ 700–850** (replaces the M26 relic),
   **stall bed** (fan-rotom/trevenant), keep m38_bc_wall. Then re-pin the
   champion and cont3 on the new roster. Also: **weight the pooled gate by
   the live opponent mix** (this sample: lucario 11, mirror 9, grim 7,
   wall 6, archaludon 5, stall 2…) so a +14pp cell on a 2%-share bed can
   never again carry a ship decision. Add wall+grim+stall legs to
   `scripts/qc_battery.py` — the QC battery currently exercises none of
   the top three loss families.
2. **Single-variable strip submit (P1, cheap, pre-funded by G5):**
   `m38_w9294_cont3` PLAIN (no gacfr3). G5 already measured plain best on
   the wall bed (.323 vs .254) with no significant loss elsewhere; live
   wall+stall is 1-7 this sample (~15% of games). This is the BACKLOG's
   own "pure-strip follow-up ship", now with live justification. Gate it
   on the refreshed roster from (1) first.
3. **Anti-deck-out package on the real beds (P2):** (a) widen stall
   coverage — add Fan Rotom, Hop's Phantump/Trevenant (+ Kanga variants)
   to blanket-or-margin conserve; racemode2's split logic already exists
   in `rl/plan.py`; (b) racemode4 burn-source demote (Enriching Energy,
   Poké Pad, surplus Dawn/Hilda once set up, Sacred Ash before deck≤2).
   Ceiling ~5–6 cards/game — flips the ≤9-card-margin races: 5 of 8
   deck-race losses were within that or from winning positions.
4. **Winners-harvest round 3, cross-family (P3):** the mirror 7-2 result
   is proof the vehicle (low-dose fine-tune on live winner seats) moves
   live outcomes. Extend the corpus to what beats us: winner seats of
   grim 738+ and wall/stall pilots give (a) the beds in (1) for free and
   (b) an *opponent-diverse* demonstration pool — the current corpus's
   same-deck filter is why the gains stayed in-family. Also chase the
   903–1058 mirror band for depth on the deck we fly.
5. **The 43% sweep mass is the real ceiling and imitation-on-our-own-family
   won't dent it (P4, design decision for Piotr):** candidates, in order
   of evidence: AWR on the harvest corpus (`--outcome-weight` hook
   validated in M37, never launched — distinct from the killed
   solver-teacher distill), opponent-conditional fine-tune arms (per-family
   heads or opponent-deck features), or accepting the BC ceiling and
   pushing the O-rule/econ lane harder. This is the M39 scoping question.

## What M38 leaves behind (assets, not just scars)

The corrected solver teacher + semantic bar + `prize_semantics_probe`
(standing regression), the harvest snowball pipeline with working auth,
`bc_m38_w9294` (231 seats @800+) and `bc_m38_wall` corpora, the
m38_bc_wall bed (first live-faithful wall instrument), the dose law
(~1 epoch fine-tune, 10 regresses), the fresh G2 pins, and the negative
result that solver-teacher distillation cannot beat a winners-lineage
champion at any dose — which redirects all future training investment
toward live demonstrations.
