# M35 plan — bench-floor rule + rules matrix (attack the pilot brick, not the deck)

Piotr's call after M34 closed no-ship (docs/M34-plan.md): the consistency deck-tech
doesn't convert, and the M34 Tier-1 data redirects at the **pilot**. The strongest
signal in the campaign right now:

- **The pilot plays a basic at bench≤1 only ~39% of the time** (`offline_behavior.py`
  vs rule:lucario: v2 201/510 offers). It DECLINES a playable basic ~60% of the time
  when it has almost no board — and 6/17 live losses (35%) are exactly that
  zero-prize sweep (docs/m33-post-mortem.md).
- Hard-bricks (no out at all) are only 1.2% (Tier-0), so the residual is the pilot not
  taking the out it has — a deterministic-rule target, not a deck one.

## The rule — `benchfloor` (O10; O9 was the M31 precedence note)

Copy the O7 `poffinfloor` shape (`rl/plan.py` `apply_play_overrides`) but promote a
**basic-Pokémon PLAY** instead of a Poffin:

- **Trigger:** bench-alive ≤ 1 AND a basic-Pokémon PLAY is legal in the menu.
- **Action:** promote the highest-*ranked* basic-Pokémon PLAY to the front (the model's
  own ordering breaks ties among basics; the rule only overrides the decision to NOT
  bench at all).
- **No deck floor** — benching a card from hand costs 0 deck, so it can never fight
  `deckguard`/`ash`/`conserve` (which gate on low deck). This is why it needs no
  `_..._DECK_AT` guard, unlike O7/O8.
- **Precedence: `ash > benchfloor > poffinfloor > drawfloor > tempo`.** Ranking
  benchfloor ABOVE poffinfloor encodes "play the basic you're holding before digging
  for one" — and this **fixes the M31 regression for free** (O7 at bench 0 displaced
  directly-playable basics; missed-bench went 0→12 prompts, M31 post-mortem).
- Card-fact set: `_IS_BASIC_POKEMON = {c.cardId for c in all_card_data() if
  c.cardType == CardType.POKEMON and c.basic}` — the exact mirror of `_IS_ENERGY`
  (`rl/plan.py:26`). Verified: the deck's Abra/Dunsparce/Fez/Shaymin ids are all in it.
  This is a deterministic card-fact reorder → the strength-free override class
  (M26 law), NOT the killed value-override class.

### Pre-registered design decision — opponent-at-1-prize veto: **NONE (by default).**
Piotr flagged the counter-example: benching fodder that the opponent Bosses up and KOs
for their last prize. **Decision: no veto.** Rationale: (1) at bench≤1 you have almost
no board — not developing is near-certain loss regardless of opp prize count; (2) the
rule only reorders an ALREADY-legal, model-ranked basic play (minimal intervention);
(3) benching more *reduces* bench-out risk; the only downside is a rare gust-snipe that
needs opp to hold Boss + energy + exact lethal on a freshly-benched basic; (4) every
extra gate is a parameter to tune and a place for bugs. **QC watch (mandatory):** does
a benched basic get gusted and KO'd for a prize? If it appears in QC replays, add the
`opp prizeCount <= 1` veto then — pre-registered, not now. **Second watch:** does it
force benching a fragile ex (Fez/Shaymin) that becomes a prize liability? If the battery
shows non-inferiority failure, refine to prefer non-ex basics.

## The rules matrix — bundle the untested revert probe (post-mortem option 2)

M30's `gac` reached 817; M31 added `poffinfloor` (→`gacb`) and it was
neutral-to-negative live (M31 post-mortem). M33 ships `gacb` + v2. So test both the new
rule AND whether `poffinfloor` is still earning its slot, in one battery — **4 arms, all
on the shipped v2 deck (`decks/alakazam_v2.csv`), pilot = m28_winners:**

| arm | fixes | tests |
|---|---|---|
| **gacb** (control) | telepath,deckguard,ash,conserve,poffinfloor | current M33 ship rules |
| **gac** | telepath,deckguard,ash,conserve | is poffinfloor dead weight? |
| **gacb+bf** | …,poffinfloor,benchfloor | new rule on top of ship |
| **gac+bf** | …,conserve,benchfloor | benchfloor *replaces* poffinfloor's role |

## Gates (pre-registered)

- **PRIMARY (behavior, the rule's job) — `offline_behavior.py` vs rule:lucario, 2 seeds:**
  - basic-PLAY-at-bench≤1 rate RISES from ~39% toward ~100% (the mechanical check the
    rule engages);
  - bench-0-after-t3 LOSSES fall vs the control arm;
  - bench≤1 prompt share falls (board established sooner);
  - the M31 missed-bench-at-0 regression does not reappear.
- **NON-INFERIORITY (strength) — full battery `scripts/m34_battery.sh` frame, n=400/bed,
  2-prop z vs the gacb control:** hold rocket + grim (the M33 deck-out wins) and be
  non-inferior on lucario/mirror/arch/kyo. A rule that mechanically engages but costs
  WR is a KILL (M30 b1ga law: engaged ≠ converts).
- Ship ONLY on: behavior primary passes AND non-inferiority holds AND QC clean +
  Piotr's explicit go. Best arm by (behavior gain × no regression) wins.

## Note on the offline instrument limit (carried from M34)
rule:lucario under-reproduces the live brick-blowout (only ~14% of its losses are
bench-0 vs 35% live). So offline can under-STATE the rule's live value — the behavior
metric (does the rule make the pilot bench?) is the trustworthy part; the WR gate is
non-inferiority only, not the place we expect to see the win. The real test is live.
See [[no-validated-live-predictor]].

## Do NOT re-propose
- poffinfloor/drawfloor threshold-fishing (M31: the supporter/deck-gated rules are spent).
- Live value/override consumption (M8.1/M12/M13 law) — benchfloor is a card-fact reorder,
  which is the allowed class (M26).

## Mechanics
- Rule is OFF by default; `main.py` default string stays `gacb` until a ship decision.
  Add `benchfloor` to the enable list only via the test spec / a future ship-default edit.
- `feature/m35` branch. Tests: extend `tests/test_plan.py` (or test_plan_overrides) with a
  bench≤1 promote case + a precedence case (benchfloor beats poffinfloor). Green before battery.
- Hermes updates per stage; heartbeat on the battery.

## Execution log (2026-07-26)

**Implementation (rl/plan.py + submission/rl/plan.py twin, byte-identical):**
- `_IS_BASIC_POKEMON = {c.cardId for c in all_card_data() if c.cardType ==
  CardType.POKEMON and getattr(c, "basic", False)}` (getattr tolerates fake_cg at
  test import; the real set is 595 basics, deck's Abra/Dun/Fez/Shaymin all in it).
- `PLAY_FIX_BENCHFLOOR = "benchfloor"`, `_BENCHFLOOR_BENCH_AT = 1`. Rule block in
  `apply_play_overrides` between `ash` and `poffinfloor` (no deck floor). Docstring +
  main.py fix-name list updated.
- matchrunner `_MODEL_FIX_KINDS`: added `modelt-gacf` (gac+benchfloor) and
  `modelt-gacbf` (gacb+benchfloor) for the 4-arm matrix.
- Tests: 4 new cases in `tests/test_attach_override.py` (thin-bench promote incl.
  no-deck-floor; guards; benchfloor > poffinfloor; ash > benchfloor). **67 tests green**
  (override + plan + plan_iter); twin md5 verified equal.

**Behavior gate RESULT (offline_behavior.py vs rule:lucario, pooled 2 seeds n=400) —
PASSES:**

| arm | WR | basic-play @bench≤1 | bench≤1 share | bench-0-t3 losses |
|---|---|---|---|---|
| gacb (control) | 0.718 | 35.9% | 13.4% | 30/113 (27%) |
| gac | 0.700 | 37.1% | 15.4% | 34/120 (28%) |
| gacf (gac+bf) | 0.750 | **98.0%** | 12.3% | **15/100 (15%)** |
| gacbf (gacb+bf) | 0.680 | 98.8% | **10.8%** | 22/128 (17%) |

- benchfloor ENGAGES (basic-play 36%→98%) and **cuts the bench-0-after-t3 sweep-loss
  share from 27% (control) to 15% (gacf) / 17% (gacbf)** — the target defect moves.
- WR at this single bed is noisy (n=400 SE≈0.023): gacf leads (0.750), gacbf trails
  (0.680) — the full 6-bed battery resolves non-inferiority + best arm. Early hint that
  dropping poffinfloor (gacf) may be the stronger config (consistent with M31's
  poffinfloor-neutral-to-negative live finding).
- rule:lucario under-reproduces the live sweep (still the instrument caveat); the
  behavior deltas are the trustworthy read, WR is non-inferiority only.

**Battery running:** `scripts/m35_battery.sh` (4 arms × 6 beds × 2 seeds),
`runs/m35_battery_full.log`, heartbeat attached. Decode vs gacb control on completion.

## Battery RESULT (48/48, 2026-07-26) — gacf is the clean pick; gacbf killed; poffinfloor confirmed droppable

Pooled n=400/cell, 2-prop z vs gacb control (M33 ship rules):

| bed | gacb (ctrl) | gac | gacf (gac+bf) | gacbf (gacb+bf) |
|---|---|---|---|---|
| **rocket** | 0.615 | 0.595 (z−0.6) | **0.632 (z+0.5)** | **0.532 (z−2.4)** |
| **grim** | 0.688 | 0.708 (z+0.6) | 0.667 (z−0.6) | 0.723 (z+1.1) |
| lucario | 0.685 | 0.703 (z+0.5) | 0.728 (z+1.3) | 0.725 (z+1.2) |
| mirror | 0.495 | 0.520 (z+0.7) | 0.530 (z+1.0) | 0.573 (z+2.2) |
| arch | 0.932 | 0.927 (z−0.3) | 0.932 (z+0.0) | 0.975 (z+2.9) |
| kyo | 0.965 | 0.955 (z−0.7) | 0.973 (z+0.6) | 0.970 (z+0.4) |

- **gacbf KILLED** — rocket 0.532 (z−2.4, resolved regression). Stacking benchfloor ON
  TOP of poffinfloor is too many rules firing at bench≤1 — the M26/M30 composition law
  (composed arms weaker than the best single). It buys mirror/arch but loses the
  must-hold rocket deck-out bed → out.
- **gacf = THE PICK** — holds every bed (rocket +0.5, grim −0.6 unresolved, lucario
  +1.3, mirror +1.0, arch flat, kyo +0.6), AND carries the resolved behavior win
  (basic-play-at-bench≤1 36%→98%, bench-0-after-t3 losses 27%→15% — the best of any
  arm). First arm to resolvedly move the #1 live loss mode with no bed regression.
- **gac (plain poffinfloor revert) also holds** — so **dropping poffinfloor is
  safe/good** regardless (vindicates M31's neutral-to-negative live read of poffinfloor).
  gacf beats gac where it counts (the bench behavior); the WR difference is a wash.
- Caveat (unchanged): WR gains are directional not resolved at n=400 (rule:lucario
  under-reproduces the live sweep); the resolved part is the BEHAVIOR. Real test is live.

**Verdict: ship candidate = `gacf` (drop poffinfloor, add benchfloor) on m28_winners +
alakazam_v2.** vs M33 live the ONLY change is the rule set (gacb→gacf) → clean live A/B.
Next: Piotr's go → build+QC (watch the gust-snipe case) → explicit submit go.

## SHIPPED — sub 55011605 (2026-07-26)

**SHIPPED: sub 55011605 = m28_winners + gacf + alakazam_v2**, Piotr's "run the shipping
pipeline". Bundle `dist/submission_neural_20260726_223604.tar.gz`. Pre-submit verified
from the tarball itself: deck md5 `7b1c123b` == alakazam_v2 (≠ base clone `8e8cf124`),
60 cards; bundle main.py default = `telepath,deckguard,ash,conserve,benchfloor` (gacf —
poffinfloor dropped, benchfloor added); plan.py twin md5-matches. QC 3W-0L vs
`sample-agent-tuned` (tuned Mega-Lucario; the base `sample-agent` crashes as a kaggle
opponent — use tuned/iono/dragapult), replays `replays/m35_qc_gacf_tuned_00{0,1,2}.html`.
Status PENDING. Clean live A/B vs M33 54997669 (settled 736.7; M30 759.4) — the ONLY
change is the rule set (gacb→gacf), so the live delta isolates benchfloor + the
poffinfloor drop.

**Watch next session (live A/B 55011605 vs 736.7):** score vs 736.7 (M33) / 759.4 (M30);
**bench-0-after-t3 loss share** (offline cut 27%→15% — the target metric; does it
transfer where deck-tech's grim gain did not?); **basic-play-at-bench≤1 rate** (offline
98%); zero-prize opening-blowout share (was 35% of M33 losses); the pre-registered
**gust-snipe watch** (benched basic gusted+KO'd for a prize → add opp-prize-≤1 veto if
seen). Re-run `scripts/live_postmortem.py 55011605` + implied-ELO solve at ~40 games.
Rules-matrix residue: `gac` (plain poffinfloor revert) also held all beds — if benchfloor
under-delivers live, gac is the fallback (poffinfloor is confirmed droppable).
