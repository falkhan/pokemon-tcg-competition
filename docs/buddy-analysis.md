# Buddy agent analysis — why 0.615 vs our solver (M9 Leg 0)

Source: kagglehub `jek1wantaufik/buddy/other/pokemon` (public model, 485-line
`main.py` + `deck.pkl`). Measured: **0.615 pooled vs `solver:lucario`
(n=1200, seeds 0.611/0.617)**, meta co-gate **0.578** (our ship 0.448).
Deck is card-for-card our `decks/lucario.csv`; bundled `cg/` byte-identical
to ours. We do NOT ship this code (user decision 2026-07-16); we learn from
it. Vetting: read end-to-end before execution; `deck.pkl` disassembled with
pickletools (primitives only).

## Architecture: one plan, every prompt serves it

The core difference from our pilot is not better per-option scores — it is
**cross-prompt coherence**. At every MAIN prompt (turn ≥ 2) buddy runs a small
forward search over (my attackers × my 2 attacks × opponent board) and stores
the argmax as a turn-scoped `AttackPlan {attacker, target, attack_index,
remain_hp, energy}` (module globals, reset when `state.turn` changes).
Every other prompt then scores options *relative to the plan*:

| Prompt | Plan-coupled behavior |
|---|---|
| PLAY Switch | -1 unless plan.attacker is on bench (6000 to enable the plan) |
| PLAY Boss's Orders | 3200 only when plan.target is on the opp bench, else -1 |
| RETREAT | 2000 only when plan.attacker ≥ 1, else -1 |
| SWITCH/TO_ACTIVE picker | +100 for exactly plan.attacker / plan.target |
| ATTACH energy | +200 for the planned attacker when plan.energy ("one more attach enables the attack") |
| ATTACK | +100 for the planned attack index |
| EVOLVE Makuhita→Hariyama | -1 if Makuhita is plan.target=active... (keeps the planned KO viable) |

Our Leg 1 Fix B failed (0.470/0.465) because it reproduced ONE row of this
table (play gust when a faster target exists) without the plan: the gust play
and the target pick and the attack choice were decided by three independent
scorers. Buddy never fires a disruption trainer unless the same turn's plan
cashes it in.

## The plan search itself (lines 188-278)

- Candidates: active + (bench if a Switch/retreat is available); attacks
  hardcoded per attacker (Mega Lucario 130/270, Hariyama 210, Solrock 70
  iff Lunatone on field — the deck's actual combat lines).
- Targets: opp active + (bench if Boss's Orders in hand or Hariyama evolve
  available — i.e., gust access is a *precondition checked before planning
  a bench kill*).
- Damage adjusted for weakness/resistance; score ≈ `pokemon_score(target)`
  (prizes*1000 + energies*150 + tools*100 + stage bonus + hp) scaled by
  `damage/hp` when not lethal; **+win short-circuit 50000 when prizes-left ≤
  planned prize take**.
- Energy feasibility inline: a plan that needs one more energy is allowed iff
  a Fighting energy is in hand and none attached yet this turn (`plan.energy`
  then routes the attach).

## Heuristics ours lacks (port candidates, ranked)

1. **Plan coherence** (the architecture itself) — the big one. Our pilot is
   stateless per prompt; our solver searches within-turn but only overrides
   ~1% of prompts (lethal tier) and its greedy fallback is plan-free.
2. **Prize-trade awareness**: Mega Lucario attacks score -500 when own prizes
   remaining ∈ {2,3} — don't feed the opponent's comeback with a 3-prize body
   when they are 2-3 prizes from winning; prefer the 1-prize attackers.
   (`pokemon_score` also subtracts prize-reduction tools/energies — id 12
   energy, Lillie cape.)
3. **Mega Brave discard scaling**: attack choice values 130-dmg Mega Brave at
   +60 × min(3, Fighting energy in discard) — deck-specific attack knowledge
   our combat table abstracts away.
4. **Win-detection short-circuit** (50000): if planned prizes ≥ prizes left,
   take the line regardless of tempo scores. Our solver has this only inside
   the DFS override tier.
5. **Setup/TO_HAND economics**: fetch scoring counts field/hand copies
   (diminishing returns per duplicate), values Lunatone/Solrock as a pair,
   Riolu count capped by Mega Lucario line needs; energy fetch valued only if
   attach still available. First-player-aware setup choice (Solrock 4 vs 2).
6. **Supporter sequencing**: Premium Power Pro (attack boost) allowed to
   burn the supporter slot ONLY if attacking (5000) or as a Carmine-enabling
   fallback (3050 vs Carmine 3000 vs Lillie 3100) — supporter slot is
   explicitly rationed. Carmine plays at flat 3000 — notably buddy does NOT
   protect keepers like our Fix A tried to, evidence our Fix A over-corrected.

## Weaknesses / non-portables

- 100% deck-specific: card ids, attack damages, evolution lines hardcoded.
  Zero deck-agnostic value; the port must re-derive these from our combat
  tables (we already have `_CARD`/`_ATK` race math).
- Hidden plan state means BC/DAgger fidelity on plan-coupled prompts will
  alias (the M1–M3 AttackPlan trap) — DAgger round 2 gates on WIN RATE only.
- No deck-out awareness (our taper/near-deckout guards have no counterpart) —
  possible edge in grind games.

## Port plan sketch (rules track, next session)

Turn-scoped plan in our pilot: compute at MAIN from `rl/combat` race math
(deck-agnostic: candidates = my board, attacks from `_ATK`, targets = opp
board gated on gust-in-hand), store `(attacker, target, attack, needs_attach)`
in the pilot closure (per-instance — NOT module globals; two pilots share a
worker process). Score PLAY/RETREAT/SWITCH/ATTACH/ATTACK options relative to
the plan, keeping every existing tier as fallback when no plan exists.
Ship-safety: the closure state pattern is exactly what `submission_rules/`
already tolerates (buddy proves it inside kaggle env); byte-parity twins get
the same closure.
