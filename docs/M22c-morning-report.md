# M22c-RL — morning report (2026-07-21)

## Headline

**Shipped — submission 54864089.** All safety checks passed. It did **not** measurably improve
against the opponent it's never trained on, but we shipped it anyway (slots were free) to get live
data and to test the idea properly.

## What we tried, in plain terms

Our agent had been doing 45% of its practice against *our own* bot — and that bot loses 2-to-1 to
the free sample agent the game ships with, on the same deck. So the agent had been getting good at
beating a weak sparring partner, which is why it looked strong on our home metric and fell apart
against real opponents.

The fix we tested: **swap the weak sparring partner for the strong one.** Same everything else,
just a better opponent to learn from. One change, so any result is clearly attributable to it.

## Technical gates (the safety checks that block a broken submission)

All five **PASSED** — the bundle is sound and nothing is broken:

| check | what it verifies | result |
|---|---|---|
| weight parity | the shipped copy computes the same thing as the trained model | ✅ PASS (diff 0.0000012) |
| bundle isolation | the submission runs standalone, no missing pieces | ✅ PASS |
| memory parity | the opponent-memory feature matches training | ✅ PASS |
| deck match | the shipped deck is the legal 60-card list we intended | ✅ PASS |
| gate game | it plays a full game without crashing | ✅ PASS |

## Did it get stronger?

**No — not measurably.** Against dragapult (the opponent it has never trained on, our honest exam):

- **New agent: 0.189** over 800 games · **old agent (B3): 0.172**
- Difference: **+1.7 percentage points — but the measurement noise is ±5.5pp, so this is INSIDE
  THE NOISE.** We cannot call it an improvement. *(Solid: this is our best-powered number.)*
- Mirror (vs our own bot): 0.480, down slightly from 0.494 — expected, since we stopped drilling
  against that bot; not a collapse.
- Floor (vs a random player): 0.920 — held, still safe.

**One number worth understanding:** the agent trained *half its time* against the strong opponent
and still only beats it 36.5% of the time. That means 10 rounds of practice wasn't enough to
actually master the harder opponent — it's still mid-learning. So this was a **weak test** of the
idea, not a clean verdict: "one short leg against a strong teacher" isn't the same as "trained to
convergence against a strong teacher."

**Good news hidden in it:** the safety check we built last week worked. If the agent had just
memorised its new practice opponent (the classic trap), it would score high against *that*
opponent and stay flat against dragapult. It didn't — it's flat against both. So no cheating, but
no gain yet either.

## What it means + next

The "our teacher is too weak" theory is **neither confirmed nor killed** — one short training run
couldn't move the needle, and the agent clearly hadn't finished learning the harder opponent. The
cheap next test is a **longer run** against the strong teacher to see if it converges to a real
gain.

But the deeper pattern from yesterday still stands: *every* agent we've built — neural or
rule-based, with search or without — sits at 0.17–0.22 against dragapult, while the game's own
sample agent sits at 0.50 on the same deck. Within-turn search didn't close that (+2.25pp,
unresolved) and a better threat model didn't either (falsified). The one lever left untouched is
giving the agent an explicit **planning head** — the ability to commit to a multi-step line and
execute it coherently — which is the one thing the sample agent does that none of ours do. That's
a real architecture change and the likely subject of the next milestone.

**Bottom line:** shipped and safe, no strength gain proven, thesis still open, and we now have a
third live arm to watch. The instruments held — nothing got dressed up as a result that wasn't
one.
