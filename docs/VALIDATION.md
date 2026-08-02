# Validation protocol — pre- and post-ship, per pilot

Standing doc (created 2026-08-01, M39). Applies to **every** shipped pilot
from M39 on. Supersedes the ad-hoc per-milestone batteries; individual
milestone plans may ADD legs, never remove them.

Written because the campaign's validation has been failing in a specific,
measurable way: we have been reading live samples far too small to support
the conclusions drawn from them, while the instruments that *are* sound
(offline gates at n=800, deterministic mechanism probes) were treated as
secondary.

---

## 0. The finding that shapes this protocol

**Our live samples cannot resolve the effects we are chasing.** Computed
from the actual ship records:

| sub | n | W-L | WR | implied ELO | **95% CI** |
|---|---|---|---|---|---|
| M37 55065484 | 63 | 33-30 | 0.524 | 732 | **[646, 819]** |
| M38 55146658 | 54 | 26-28 | 0.481 | 659 | **[566, 753]** |

Those intervals **overlap across almost their entire width**. The headline
"M38 regressed from ~719 to 659" is a point-estimate difference of ~73 ELO
read off two distributions ~180 ELO wide. It is a reasonable *working
hypothesis*; it is not an established fact, and this doc exists so we stop
writing it down as one.

**Minimum detectable effect, 80% power, α=0.05** (two-proportion):

| effect | ≈ ELO | n needed per arm |
|---|---|---|
| +5pp WR | +35 | 1560 |
| +7pp WR | +49 | 792 |
| +10pp WR | +70 | 384 |
| +15pp WR | +105 | 166 |
| **+28pp WR** | **+196** | **50** ← what one day of ladder actually buys |

At **n=50 a live read detects only a ~200 ELO swing.** Every improvement
this campaign is designed to produce is smaller than that.

**Per-matchup live cells are worse still.** Fisher exact, two-sided, on the
cells the M38 post-mortem built its ranking from:

| cell | M37 → M38 | p |
|---|---|---|
| grim | 6-2 → 1-6 | **0.041** (marginal — and one of ~8 matchups examined, so ~1 such result is expected by chance alone) |
| wall | 3-6 → 1-5 | **0.604** (indistinguishable from noise) |

"Wall went 1-5" is a headline in the post-mortem and it is **not a
detectable change**. The grim flip is the strongest per-matchup result we
have and it barely clears p=0.05 before any multiplicity correction.

### What this does and does not invalidate

**Does not.** The M39 plan's direction is still well-founded, because it
rests on instruments that *are* powered: G5's offline wall matrix (n=1200
plain / n=800 gacfr3), the corrected-teacher kills (n=400/bed), and
descriptive mechanism forensics (deck-out counts, burn audits, fire probes)
which do not require significance at all.

**Does.** Any claim of the form "matchup X moved by Y" from a live sample,
and any gating decision made on one. Note the same cell has now been
misread twice in opposite directions — grim was called "self-resolved" at
6-2 (M37) and "the bed lied" at 1-6 (M38). Both reads were n ≤ 8.

### The lever we have been leaving on the table

Slots are consumed by *submitting*, not by *waiting*. M38 accrued 54 games
in ~1 day, so:

| time live | ≈ n | 95% CI half-width |
|---|---|---|
| 1 day | 54 | ±93 ELO |
| 3 days | ~150 | ±56 ELO |
| 5 days | ~250 | ±43 ELO |
| 7 days | ~380 | ±35 ELO |

**Leaving a pilot up for 3–5 days costs nothing and roughly halves the
error bar.** Rushing the next submit at n=50 throws away statistical power
that is free. This is the single cheapest validation improvement available
to us — and per §2's declared-dwell rule, each ship picks its own point on
this table before it goes live.

---

## 1. Pre-ship protocol (per pilot)

Four tiers. Tiers 1–2 are **deterministic and n-independent** — they are
the strongest guarantees we have and they are cheap. Tier 3 carries the
strength claim. Tier 4 is a smoke test and must never be quoted as
evidence of strength.

### Tier 1 — Artifact correctness (binary, blocking)

Every check runs **against the exported tarball**, not the checkpoint —
gate the artifact that ships, not the one that trained.

| check | pass condition |
|---|---|
| Twin parity | `submission/rl/*` byte-identical to `rl/*` |
| From-tarball verify | unpack, confirm deck md5, weights bit-identical to the gated checkpoint, fix string matches intent |
| `prize_semantics_probe` | 474/474 (standing regression) |
| Fix-name resolution | every name in the fix string resolves — no silent ignore (the M39 P1 `ValueError` hardening) |
| Latency budget | p50/p99 per decision and s/game measured from the tarball, against the 600 s Kaggle budget |

### Tier 2 — Mechanism verification (deterministic, n-independent)

**The most underused instrument we own, and the one that has never lied.**
M37's `racemode_fire_probe.py` established the pattern: 9/9 wall games
trigger-true, 0 fires / 723 prompts on decks with no wall Pokémon. That is
a *proof of behavior*, not a sample estimate.

- **Fire probe for every changed rule.** Does it fire in exactly the states
  intended, and provably never elsewhere? For a **strip** the probe is
  inverted: confirm the removed rules produce 0 fires and that the
  remaining behavior matches the gated arm exactly.
- **Behavioral diff vs the parent pilot.** Run parent and candidate on
  identical seeds; count decision divergences and classify them. A strip
  must diverge **only** in states where the stripped rules fired. Any
  divergence outside that set means the ship is not the single variable we
  think it is — investigate before export, do not explain it away.
- The probe written pre-ship is the **same instrument** re-run post-ship on
  live replays. Write it once, use it twice.

### Tier 3 — Strength (this is where the claim lives)

- **Weighted gate (G-3)** on the bed roster, n=800/bed, 2-seed. The
  weighted pool is THE number; unweighted reported for continuity.
- **Coverage (G-8):** roster must cover ≥80% of live-mix mass, and no claim
  may be made about a band with no bed for it.
- **Ceiling cell** vs the high-band bed, reported separately, never blended
  into the pool.
- Per-cell z vs control; single-variable cells only.

### Tier 4 — Smoke test + human review

- **QC battery** (`scripts/qc_battery.py`), 3 games × the full opponent
  roster. **Explicitly a smoke test:** it catches crashes, timeouts,
  illegal actions, and catastrophic breakage. At n=3 its minimum detectable
  effect is near 100pp — it cannot measure strength. Quoting "11W-1L" as
  evidence of quality (M38) is a category error and is now barred.
- **Piotr's manual replay review + explicit go.** Mandatory, no exceptions.
  This is the qualitative instrument, and it is genuinely powerful for
  behavioral pathology that statistics cannot see at our sample sizes.

---

## 2. Post-ship protocol (per pilot)

### Pre-registered read schedule

Reading a live sample continuously and reacting to it is a multiple-
comparisons machine. Read points are fixed in advance:

| read | n | what it may conclude |
|---|---|---|
| **Mechanism** | ≥20 | Fire probe on live replays: did the change fire as designed? **No strength verdict permitted at this read.** |
| **Provisional** | ≥50 | Point estimate + CI. Catastrophe detection only (a >150 ELO move). Not a next-slot trigger. |
| **Decision** | **declared per ship** | The primary read. Loss anatomy, watch items, next-slot decision. |
| **Confirmation** | +100 over the decision read | Tightens the interval if the decision read is ambiguous and the calendar allows. |

### The declared-dwell rule (G-10)

**Piotr's call, 2026-08-01: dwell is decided per ship, not fixed.** The
decision-read n is therefore **declared in the ship commit, before the
pilot goes live**, chosen against the effect that ship is trying to detect:

| declared n | ≈ days live | 95% CI half-width | suitable for |
|---|---|---|---|
| 50 | 1 | ±93 ELO | catastrophe detection only |
| 150 | 3 | ±56 ELO | a normal ship verdict |
| 250 | 5 | ±43 ELO | a contested or headline result |
| 380 | 7 | ±35 ELO | a campaign-target claim |

**Why declaration is the load-bearing part.** The hazard in a per-ship call
is not the flexibility — it is reading the number continuously and stopping
when it looks good, which inflates false-positive rate without leaving any
trace in the record. Fixing the target in advance keeps every bit of the
flexibility and removes the hazard entirely.

Reading *early* is always permitted for catastrophe detection and mechanism
probes. What is fixed in advance is the n at which a **verdict** may be
taken. If a ship needs a longer dwell than declared, that is a new
declaration written down with its reason — not a silent extension.

**One fixed floor:** the campaign-target claim (implied ELO ≥ 1000)
requires **n ≥ 150** whatever the per-ship declaration says. A claim that
large cannot rest on a ±93 interval.

### Standing rules

1. **Every live ELO is reported as point estimate + 95% CI.** A bare point
   estimate is not a result. This applies in diaries, post-mortems, plans,
   and MILESTONES rows.
2. **Per-matchup live cells are forensic only.** They generate hypotheses
   for offline testing; they never gate a decision, never justify a code
   change on their own, and are labelled `forensic (n=X)` wherever quoted.
   An offline bed at n=800 is the instrument for matchup claims.
3. **Parent stays live as a concurrent control (G-5).** Report the paired
   same-window delta, not the child against the parent's frozen number —
   pool drift is otherwise confounded with the change.
4. **Watch items carry both a threshold and a read-n** (G-2). "Watch the
   wall cell" is not a watch item; "wall WR < 0.35 at n ≥ 30 in-family"
   is.
5. **A null live result is reported as null, not as a regression.** If the
   CI spans zero, the honest statement is "no detectable change at n=X,"
   and the offline gate remains the best estimate of the effect.

### Post-mortem contents (standing)

Loss anatomy by mode, mechanism-probe results, weighted-mix refresh,
watch-item resolutions with their pre-registered thresholds, CI on every
number, and an explicit list of claims the sample was **too small to
support** — that last section is what would have caught the wall-cell
overreach in M38.

---

## 2b. Offline batteries are samples too (added 2026-08-01, M39 P0.7)

The power discipline above was written for LIVE reads. M39's P0.7 showed it
applies to the offline gate as well, which had been treated as if its
numbers were exact.

**`matchrunner --workers 8` is not run-reproducible.** The engine RNG is
per worker process and `mp.Pool` assigns chunks by timing, so the seed does
not pin the result — M37 established this (`scripts/m37_decide.py` B4) and
M39 re-confirmed it by running one identical command five times at seed 1:
**0.471 / 0.494 / 0.526 / 0.506 / 0.537**.

The spread is ordinary binomial, not excess variance, but the consequence
is that **every cell is an independent sample and a "seed" is just a label**.
So the same power table applies:

| cell n | 95% CI half-width | smallest delta it can resolve |
|---|---|---|
| 400 | ±4.9pp | ~14pp |
| 800 (the campaign's habitual 2-seed cell) | ±3.5pp | **~10pp** |
| 1600 | ±2.5pp | ~7pp |
| 2400 | ±2.0pp | ~5pp |

**The campaign's standard n=800 cell cannot resolve a 5pp rules effect** —
which is the size of most rules effects we chase. M39 P0.7 measured the
same contrast twice at n=800 and got +5.0pp (z=+2.00) and −2.1pp (z=−0.14);
at n=2400 the truth was −0.2pp. A published conclusion was corrected as a
direct result.

- **G-12 Offline power.** A gate cell that decides a ship must be powered
  for the effect it claims: n≥2400 for a ~5pp claim, n≥800 only for ~10pp+.
  Cells below that power may be reported, but a delta inside their MDE is
  "no signal", never "no effect" — and never a confirmation.
- **Corollary — replicate before believing a surprise.** A single battery
  producing an unexpected significant result is a hypothesis. Re-run it
  before it enters a plan, a diary conclusion, or a ship decision.
- **Corollary — count your contrasts.** A 12-contrast ablation will throw a
  p≈0.05 result by chance. Say how many were tested whenever one is quoted.

## 3. New standing guardrails

- **G-9 Statistical honesty.** No live claim without a CI. No gating
  decision on a live cell with n < 30, and no matchup-level gating from
  live data at any n. Offline beds are the instrument for matchup claims.
- **G-10 Dwell time.** A pilot stays live until the decision read (n ≥ 150)
  unless it is catastrophically broken. Submitting the next pilot early
  spends statistical power we get for free.
- **G-11 Mechanism proof per ship.** Every ship carries a fire/behavioral-
  diff probe for its changed behavior, run pre-ship offline and re-run
  post-ship on live replays. A ship whose mechanism cannot be probed is a
  ship whose live result cannot be attributed.

---

## 4. Application to the M39 ships

**Ship A (strip).** The ideal case for this protocol: a config-only change
makes the Tier-2 behavioral diff nearly a proof. Pre-ship, the diff against
the M38 pilot must show divergences *only* in states where
deckguard/ash/conserve/benchfloor/racemode3 fired; post-ship, the inverted
fire probe must show 0 fires for the stripped rules across the live sample.
That is an n-independent verification that the intended change — and only
it — reached production. The ELO read is secondary and will be wide.

**Ship B (net × rules).** Two lanes move, so per G-7 it does not go out
until Ship A's decision read (n ≥ 150). Its mechanism probe covers the P2a
trigger set and the P2b demote conditions. Its strength claim comes from
the weighted gate; the live read confirms direction and catches
catastrophe, and nothing more should be asked of it.

**Consequence to accept up front:** with 4 slots, we will *never* have a
live sample that resolves a 50-ELO improvement. The offline weighted gate
is and remains the strength instrument. Live validation exists to confirm
transfer direction, prove mechanism, and catch catastrophe — and the
campaign's repeated error has been asking it for more than that.
