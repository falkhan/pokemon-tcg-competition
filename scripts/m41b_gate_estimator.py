"""M41b § II.3c — is there a lower-variance gate estimator? Decided on data.

The pre-registered question (docs/M41b-plan.md § II.3c): gates score a
Bernoulli win/loss; can the per-game MARGIN the engine already records shrink
the interval on the same games? Pre-registered consequence: "If adjustment
does not shrink the interval materially, drop it and say so."

This runs three estimators over one battery cell and reports what each is
worth, with a bootstrap as the referee:

  raw        the binomial win rate, what every gate uses today
  adjusted   CUPED-style regression adjustment on the prize margin
  seat       the same win rate with the SLOT-FAIR design accounted for

The middle one is the one under test and the one to be careful with. CUPED
buys variance reduction when the covariate is (a) measured BEFORE the
treatment and (b) independent of which arm is being tested. The prize margin
is neither: it is a same-game, post-outcome quantity, and a stronger arm
produces better margins. Two consequences the report makes explicit rather
than assuming:

  * With E[X] estimated in-sample, the adjustment is an ALGEBRAIC NO-OP for
    the point estimate: mean(Y - t*(X - mean(X))) == mean(Y), exactly. Any
    "tighter" interval computed from the residual spread is therefore a
    smaller number attached to the same estimator, i.e. an understatement of
    real uncertainty -- the failure mode this milestone exists to catch.
  * The bootstrap settles it empirically: resample games, recompute both
    estimators, compare the SPREAD of the estimates themselves.

The third estimator is the constructive half. `play_series` is slot-fair by
construction -- side a takes seat g % 2, so each chunk holds exactly as many
seat-0 as seat-1 games. Seat is assigned by the DESIGN, before any game, and
independently of the arm, which is precisely what the margin is not. If the
seats differ in win rate (going first is worth something in this game), the
naive binomial SE charges the gate for between-seat variance the design has
already eliminated, and the stratified formula is a free, correct tightening.

    uv run python scripts/m41b_gate_estimator.py runs/<cell>.jsonl
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z = 1.959963985                      # two-sided 95%
BOOTSTRAP = 4000


def load_games(path: Path) -> list[tuple[float, float | None, int]]:
    """[(outcome, margin, seat)] for one battery checkpoint.

    outcome is side a's score (win 1.0 / loss 0.0 / draw 0.5, matching
    `series_wr`). margin is side a's prize lead at the final state, i.e.
    b's prizes remaining minus a's -- positive means a is ahead. seat is
    recovered from the index WITHIN its chunk: `play_series` gives side a
    seat g % 2 and `_make_jobs` keeps chunks even, so position parity is the
    seat exactly.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    out = []
    for row in rows[1:]:
        results = row["results"]
        margins = row.get("margins") or [None] * len(results)
        for i, res in enumerate(results):
            score = 1.0 if res == 0 else (0.5 if res == 2 else 0.0)
            m = margins[i] if i < len(margins) else None
            # MARGIN_FIELDS = (a_prizes, b_prizes, a_deck, b_deck)
            margin = float(m[1] - m[0]) if m else None
            out.append((score, margin, i % 2))
    return out


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def var(xs) -> float:
    """Sample variance (n-1), the one that feeds an interval."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def raw_estimate(games) -> tuple[float, float]:
    """(win rate, 95% half-width) — what every gate reports today."""
    y = [g[0] for g in games]
    return mean(y), Z * math.sqrt(var(y) / len(y)) if y else 0.0


def theta(y, x) -> float:
    """Cov(Y,X)/Var(X), the variance-minimising adjustment coefficient."""
    vx = var(x)
    if vx == 0:
        return 0.0
    my, mx = mean(y), mean(x)
    cov = sum((a - my) * (b - mx) for a, b in zip(y, x)) / (len(y) - 1)
    return cov / vx


def adjusted_estimate(games) -> tuple[float, float, float, float]:
    """(point, half-width from residual spread, theta, correlation).

    The half-width here is the one a naive implementation would report. The
    report below is explicit that it is NOT a valid interval for this
    estimator; it is computed so the size of the illusion is on the record.
    """
    paired = [(g[0], g[1]) for g in games if g[1] is not None]
    if len(paired) < 2:
        return 0.0, 0.0, 0.0, 0.0
    y = [p[0] for p in paired]
    x = [p[1] for p in paired]
    t = theta(y, x)
    mx = mean(x)
    adj = [yi - t * (xi - mx) for yi, xi in zip(y, x)]
    vy, vx = var(y), var(x)
    rho = (t * math.sqrt(vx / vy)) if vy > 0 and vx > 0 else 0.0
    return mean(adj), Z * math.sqrt(var(adj) / len(adj)), t, rho


def seat_estimate(games) -> tuple[float, float, dict]:
    """(win rate, 95% half-width, per-seat detail) under the slot-fair design.

    Seat counts are fixed by the design, not sampled, so the correct variance
    is the stratified one: sum over seats of (w_s^2 * V_s / n_s). The naive
    binomial SE additionally charges the between-seat term, which the
    alternation has already removed.
    """
    strata = {}
    for score, _margin, seat in games:
        strata.setdefault(seat, []).append(score)
    n = sum(len(v) for v in strata.values())
    if not n:
        return 0.0, 0.0, {}
    point = sum(len(v) / n * mean(v) for v in strata.values())
    variance = sum((len(v) / n) ** 2 * var(v) / len(v)
                   for v in strata.values() if len(v) > 1)
    detail = {s: (len(v), mean(v)) for s, v in sorted(strata.items())}
    return point, Z * math.sqrt(variance), detail


def bootstrap_spread(games, reps: int, seed: int = 12345) -> dict:
    """Empirical SD of each estimator under resampling — the referee.

    Whatever an estimator's formula claims, this is how much it actually
    moves from sample to sample. Games are resampled WITHIN seat so the
    slot-fair design is preserved across replicates.
    """
    rng = random.Random(seed)
    by_seat = {}
    for g in games:
        by_seat.setdefault(g[2], []).append(g)
    raw, adj, seat = [], [], []
    for _ in range(reps):
        sample = []
        for _s, pool in by_seat.items():
            sample += [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
        raw.append(raw_estimate(sample)[0])
        adj.append(adjusted_estimate(sample)[0])
        seat.append(seat_estimate(sample)[0])
    return {"raw": math.sqrt(var(raw)), "adjusted": math.sqrt(var(adj)),
            "seat": math.sqrt(var(seat))}


def report(path: Path, reps: int) -> int:
    games = load_games(path)
    have_margin = [g for g in games if g[1] is not None]
    print(f"cell: {path.name}")
    print(f"games: {len(games)}  with margin: {len(have_margin)}")
    if not games:
        print("FAIL: no games — nothing to decide on.")
        return 1
    if not have_margin:
        print("FAIL: no margins in this checkpoint. Pre-M41b batteries dropped "
              "them in the worker; re-run the cell to decide § II.3c.")
        return 1

    p_raw, h_raw = raw_estimate(games)
    p_adj, h_adj, t, rho = adjusted_estimate(games)
    p_seat, h_seat, detail = seat_estimate(games)

    print("\n--- estimators ---------------------------------------------")
    print(f"raw       wr {p_raw:.4f}  +/- {h_raw:.4f}")
    print(f"adjusted  wr {p_adj:.4f}  +/- {h_adj:.4f}   "
          f"(theta {t:+.4f}, corr(Y,margin) {rho:+.3f})")
    print(f"seat      wr {p_seat:.4f}  +/- {h_seat:.4f}")
    for s, (n_s, m_s) in detail.items():
        print(f"            seat {s}: n={n_s:<6d} wr {m_s:.4f}")

    print("\n--- is the adjustment real? --------------------------------")
    drift = abs(p_adj - p_raw)
    print(f"point estimates differ by {drift:.2e} "
          f"({'identical to float noise — an ALGEBRAIC NO-OP' if drift < 1e-9 else 'DIFFERENT — investigate'})")
    if h_raw > 0:
        print(f"residual-spread interval is {100 * (1 - h_adj / h_raw):.1f}% "
              "narrower than raw, on an estimator that is the SAME NUMBER")

    print(f"\n--- bootstrap ({reps} reps), the referee -------------------")
    sd = bootstrap_spread(games, reps)
    print(f"actual SD of the estimate    raw {sd['raw']:.5f}   "
          f"adjusted {sd['adjusted']:.5f}   seat {sd['seat']:.5f}")
    print(f"implied 95% half-width       raw {Z * sd['raw']:.4f}   "
          f"adjusted {Z * sd['adjusted']:.4f}   seat {Z * sd['seat']:.4f}")

    print("\n--- verdict -------------------------------------------------")
    if drift < 1e-9:
        print("MARGIN ADJUSTMENT: DROP. The margin is a same-game, "
              "post-outcome quantity, so with E[X] taken in-sample the "
              "adjusted estimator IS the raw one; its narrower residual "
              "interval understates real uncertainty rather than reducing "
              "it. Pre-registered consequence applies: drop it and say so.")
    else:
        print("MARGIN ADJUSTMENT: point estimate moved — do not ship this "
              "without understanding why.")
    gain = 1 - (sd["seat"] / sd["raw"]) if sd["raw"] > 0 else 0.0
    print(f"SEAT STRATIFICATION: {gain * 100:+.1f}% change in the true SD. "
          + ("Worth adopting — it is free and it is correct: seat is fixed by "
             "the slot-fair design, before any game and independently of the "
             "arm." if gain > 0.02 else
             "Not material on this cell — the seats do not differ enough "
             "here to pay for the extra machinery."))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="a run_pairs jsonl carrying margins")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    a = ap.parse_args()
    return report(Path(a.checkpoint), a.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
