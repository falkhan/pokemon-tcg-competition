# M27 Phase C — capability diff vs `sample-agent/main.py` (the 0.505 pilot)

`docs/M22.md:748-753` named this comparison "the obvious next step" and it was
never done. Four milestones later: **every pilot we have built sits at 0.17–0.22
vs `rule:dragapult`; this agent sits at ~0.505 on the same deck.** 508 lines,
read end to end.

**Scope rule:** `sample-agent-dragapult/main.py` stays SEALED — `rule:dragapult`
is our only doubly-clean out-of-loop instrument (`docs/DECISIONS.md`
2026-07-20). Everything below is from `sample-agent/main.py` (= `rule:lucario`,
`rl/teacher.py:18`), which is already in every training pool.

---

## The sharpened question: how does it decide *when* to play a supporter?

Probe 3 says our clone plays supporters at 0.51× and stadiums at 0.06× the
teacher's rate on identical menus. Here is what a 0.505 pilot does instead
(`sample-agent/main.py:416-455`):

```python
elif o.type == OptionType.PLAY:
    ...
    else:                                     # not a Pokémon = trainer
        score = 10000
        if card.id == Switch:
            score = -1 if plan.attacker <= 0 else 6000
        elif card.id == Premium_Power_Pro:
            if state.supporterPlayed and plan.remain_hp <= 0:   score = -1
            elif not can_attack:
                score = 3050 if (not state.supporterPlayed
                                 and hand_counts[Carmine] > 0
                                 and hand_counts[Lillie_Determination] == 0) else -1
            else:                                                score = 5000
        elif card.id == Boss_Orders:
            score = 3200 if plan.target >= 1 else -1      # ← gust iff the plan wants a bench target
        elif card.id == Carmine:              score = 3000
        elif card.id == Lillie_Determination: score = 3100
        elif card.id == Gravity_Mountain:
            score = -1 if stadium_id != 0 else 10000      # ← stadium iff no stadium in play
```

**The supporter/stadium decision is a PRECONDITION TEST, not a scalar
preference.** Three shapes, all binary:

| card | predicate | our equivalent |
|---|---|---|
| Boss's Orders (gust) | `plan.target >= 1` — the attack plan names a **bench** target | `plan.needs_gust` exists at `encode_plan` slot 18 (`rl/plan.py:71`), **but nothing couples it to the gust OPTION** |
| Gravity Mountain (stadium) | `stadium_id == 0` — no stadium in play | not encoded at option level at all |
| Premium Power Pro | 3-branch on `state.supporterPlayed` × `can_attack` × what else is in hand | `supporterPlayed` is 1 of 1,708 state scalars, no option interaction |
| Carmine / Lillie | flat 3000 / 3100 — **always play a draw supporter**, with a strict tie-break between the two | learned from 166 positive rows |

Our net must infer every one of these from raw features through a late-fusion
MLP. Boss's Orders has **31 positive val rows**. Of course it fails.

### 🔴 The single highest-value finding

`rl/plan.py:116` already computes `needs_gust = tslot > 0`, and `rl/plan.py:137`
already gates bench-target plan enumeration on a gust card being in hand
(`GUST_IDS = {1182}`). **We have the exact same coupling the sample agent has —
in one direction only.** The plan won't propose a bench target without a gust in
hand, but nothing makes the model *play* the gust once the plan has chosen one.
The sample agent closes that loop with one line (`score = 3200 if plan.target >= 1`).

That is a precise, cheap, one-feature explanation for Boss's Orders at **0.8%
offline / 0.94% live vs the teacher's 6.5% / 3.5%** — the anti-wall tool whose
absence the wall bleed (kangaskhan+crustle 0.31, n=39) is named after.

---

## Structural differences (ranked by explains-a-measured-defect × cheap)

### 1. Plan is RE-DERIVED after every action; ours is frozen for the turn

`sample-agent/main.py:145-148` resets `plan` on a turn change, then the whole
plan block (`:207-297`) re-runs **on every MAIN prompt**. So after playing a
card, the plan is recomputed against the new board.

Ours commits once at the turn's first MAIN prompt and holds it for every submenu
(`submission/main.py:198-204`, mirrored in `rl/matchrunner.py:466`). That was a
*measured* fix (0.278 → 0.345) so it is not simply wrong — but it means our plan
cannot react to its own setup within the turn. Note the asymmetry is baked into
training too: `plan_labels` rows exist only at first-MAIN states
(`rl/plan_iter.py:169`), so this cannot be changed on one side only.

### 2. Two-way plan↔gust coupling (above) — **the actionable one**

`:266-268` — bench targets are only *considered* if `can_op_switch`; `:444-448`
— the gust is only *played* if a bench target was chosen. We have the first
half, not the second.

### 3. Turn-ending is structurally last

Score tiers: ABILITY 30000 > PLAY-Pokémon 20000 > PLAY-trainer 10000 > EVOLVE
9000 > ATTACH ~8000 > Hero Cape 7000 > Switch 6000 > supporters 3000–3200 >
RETREAT 2000 > **ATTACK 1000**. The turn-ending action is the lowest-scored real
action; everything else outranks it by construction.

**This is the M26/M27 stub's "free actions first" discipline — and Probe 1 shows
we already have it** (clone ends the turn 12.9% vs teacher 12.1%). Our net
learned this ordering from data. Confirms the stub's masking/backstop levers
would have bought nothing. Recorded as a negative result, not a lever.

### 4. Explicit per-turn-resource conditioning at card-search time

`:409-413` — searching a Basic Fighting Energy to hand is worth +30 only if
`not ability_used or not state.energyAttached`, else −1. The per-turn attach
budget conditions the *search* decision, several plies before the attach.
Our CARD class (28.9% of the corpus, val 0.704) has no such conditioning.

### 5. Anti-duplicate search logic

`:376` — `score = 200 - hand_counts[card.id] * 100`, then per-card board-count
adjustments (`field_counts[card.id] >= 1 → -250` for Lunatone/Solrock).
Diminishing returns on drawing copies of what we already have. Plausibly related
to the `fetch-dead-evolution` flag doubling in the clone (0.44 → 0.94/game,
`docs/M25.md`).

### 6. Threat model — the diff is smaller than M22 implied

This agent's opponent model is *also* thin: `pokemon_score` (`:97-115`) values
opponent Pokémon by prize count, energy, tools, stage and HP, and the plan loop
applies weakness/resistance (`:271-274`). It has **no bench-spread model
either**. So cause A (spread blindness, p<0.0001) is **not** what separates
0.505 from 0.22 on this deck — that finding came from the *dragapult* agent
(`docs/M22.md:565`, knapsack bench enumeration), which stays sealed.

**Consequence for the M27 plan: Phase E drops in priority.** Spread-threat
representation is still a real, confirmed, untouched defect, but the 0.505 pilot
does not have it, so it cannot explain our deficit against that pilot. Do Phase
D first.

---

## What to lift (Phase D shopping list)

All three are **option-level interaction features** — the same shape, the same
place (`encode_option_v2` in `rl/encoders.py`), and none of them is an override:

1. `is_gust_card × plan.needs_gust` — closes the plan↔gust loop (finding 2).
2. `is_stadium × no_stadium_in_play` — Gravity Mountain's exact predicate.
3. `is_supporter × not state.supporterPlayed` — makes the one-per-turn
   constraint visible at the option, not buried as 1 of 1,708 state scalars.

Cheap, targeted at the three cells Probe 3 measures as broken, and expressible
in the existing architecture. Pair with Phase D arm 1 (card-kind reweighting) so
the net has both the *signal* and the *gradient* to use it.

## What NOT to lift

- The score tiers (finding 3) — we already match the teacher on turn-ending.
- Per-card hard-coded scores — deck-specific, and the override law's five kills
  are all "hand-authored rule consumes a signal at play time." These features
  are *inputs*, not overrides; keep it that way.
