"""M40 X0 — the ladder accrual audit. Does live read bandwidth SPLIT?

docs/M40-plan.md §3 prices the whole milestone on this: at 10 / 30 / 55
games/day the campaign gets ~3 / ~8 / ~15 decision-grade live reads before the
2026-09-13 deadline (n=150 per G-10). That table is the difference between
"pick three experiments very carefully" and "iterate". The number it is
parameterised on has never been measured — the only throughput statement in the
repo is a prose docstring (rl/live_monitor.py: "~60-100 episodes/day across ALL
active arms"), and the phrase "across all active arms" is precisely the
ambiguity: if the ladder gives a TEAM a fixed number of games and splits them
across live submissions, parallel reads are an illusion and M40 must go back to
sequential ships.

Everything needed is already on disk. data/kaggle/listings/<sub>.parquet is one
row per episode with create_time/end_time, written by the model_monitor
notebook via rl.kaggle_ingest.list_episodes. This script performs no network
I/O; pass --stale-days to control how loudly it complains about age.

TWO QUESTIONS, and the second one turned out to be the binding one.

1. THE SPLIT TEST: group calendar days by k = how many submissions were LIVE
   that day, and compare mean TOTAL games/day against mean PER-SUB games/day as
   k rises. Per-sub flat + total rising = accrual is PER SUBMISSION and parallel
   reads are real; total flat + per-sub falling as 1/k = accrual is PER TEAM and
   parallel reads are an illusion.

2. THE LIFETIME CURVE — the question the plan never asked. Accrual is NOT
   stationary within a submission's life. Every submission gets a burst on its
   first day and decays to zero within ~4 days, so "games/day" is not a rate,
   it is a point on a curve, and a submission's LIFETIME TOTAL is capped. If
   that cap is below the declared read-n, then n is unreachable no matter how
   much calendar is left, and the whole budget table is answering the wrong
   question.

   This was verified against the live API, not just the cache: refetching an
   old submission returns exactly its cached rows, so the decay is real and
   not an artifact of the notebook's 3-day refresh window.

First and last days of a submission's life are PARTIAL by construction (it went
live at some hour), so they bias per-day rates in both directions. The split
table is reported twice: all days, and interior days only.

Usage:
    uv run python scripts/m40_accrual.py
    uv run python scripts/m40_accrual.py --read-n 150 --deadline 2026-09-13
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LISTINGS = ROOT / "data/kaggle/listings"
OUT = ROOT / "data/m40_accrual.json"


def _day(ts: str | None) -> str | None:
    return ts[:10] if ts else None


def load_days() -> tuple[dict[int, dict[str, int]], dict[int, str]]:
    """sub -> {day: games}, and sub -> newest episode timestamp."""
    per: dict[int, dict[str, int]] = {}
    newest: dict[int, str] = {}
    for f in sorted(LISTINGS.glob("*.parquet")):
        sid = int(f.stem)
        df = pl.read_parquet(f)
        if df.height == 0:
            continue
        # end_time is the episode's finish; create_time is when it was queued.
        # A still-running episode has no end_time, so coalesce rather than drop.
        stamps = df.select(
            pl.coalesce([pl.col("end_time"), pl.col("create_time")]).alias("t")
        )["t"].to_list()
        stamps = [s for s in stamps if s]
        if not stamps:
            continue
        counts: dict[str, int] = defaultdict(int)
        for s in stamps:
            counts[_day(s)] += 1
        per[sid] = dict(sorted(counts.items()))
        newest[sid] = max(stamps)
    return per, newest


def daterange(a: str, b: str) -> list[str]:
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    return [date.fromordinal(o).isoformat()
            for o in range(d0.toordinal(), d1.toordinal() + 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-n", type=int, default=150,
                    help="G-10 declared read size a decision needs")
    ap.add_argument("--deadline", default="2026-09-13")
    ap.add_argument("--stale-days", type=int, default=2)
    a = ap.parse_args()

    per, newest = load_days()
    if not per:
        print(f"no listings under {LISTINGS}")
        return 2

    # ---- per-submission throughput ---------------------------------------
    print("=" * 78)
    print("PER-SUBMISSION ACCRUAL  (interior = excluding the partial first/last day)")
    print("=" * 78)
    print(f"{'sub':>10} {'first':>11} {'last':>11} {'days':>5} {'games':>6}"
          f" {'g/day':>7} {'interior':>9} {'int g/day':>10}")
    rows = {}
    for sid, days in sorted(per.items()):
        ds = sorted(days)
        span = daterange(ds[0], ds[-1])
        total = sum(days.values())
        interior = [d for d in span if d not in (ds[0], ds[-1])]
        int_games = sum(days.get(d, 0) for d in interior)
        int_rate = (int_games / len(interior)) if interior else None
        rows[sid] = dict(first=ds[0], last=ds[-1], span_days=len(span),
                         games=total, rate=total / len(span),
                         interior_days=len(interior),
                         interior_rate=int_rate)
        print(f"{sid:>10} {ds[0]:>11} {ds[-1]:>11} {len(span):>5} {total:>6}"
              f" {total / len(span):>7.1f} {len(interior):>9}"
              f" {('%.1f' % int_rate) if int_rate is not None else '-':>10}")

    # ---- the split test ---------------------------------------------------
    # A sub is LIVE on every day inside its first..last window, including days
    # it happened to record nothing — an idle day inside a sub's life is the
    # very evidence a split would produce.
    live_on: dict[str, list[int]] = defaultdict(list)
    interior_on: dict[str, list[int]] = defaultdict(list)
    for sid, days in per.items():
        ds = sorted(days)
        for d in daterange(ds[0], ds[-1]):
            live_on[d].append(sid)
            if d not in (ds[0], ds[-1]):
                interior_on[d].append(sid)

    def split_table(day_map, label):
        by_k: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for d, subs in sorted(day_map.items()):
            k = len(subs)
            total = sum(per[s].get(d, 0) for s in subs)
            by_k[k].append((total, total / k))
        print(f"\n{label}")
        print(f"  {'k live':>7} {'days':>5} {'mean total/day':>15}"
              f" {'mean per-sub/day':>17}")
        out = {}
        for k in sorted(by_k):
            tot = sum(t for t, _ in by_k[k]) / len(by_k[k])
            perp = sum(p for _, p in by_k[k]) / len(by_k[k])
            out[k] = dict(days=len(by_k[k]), mean_total=tot, mean_per_sub=perp)
            print(f"  {k:>7} {len(by_k[k]):>5} {tot:>15.1f} {perp:>17.1f}")
        return out

    print("\n" + "=" * 78)
    print("THE SPLIT TEST — does throughput divide across concurrently live subs?")
    print("=" * 78)
    all_k = split_table(live_on, "ALL DAYS (first/last are partial -> biased low)")
    int_k = split_table(interior_on,
                        "INTERIOR DAYS ONLY (full days; the honest table)")

    # Both k-tables above are CONFOUNDED BY AGE and must not be read as the
    # answer: accrual decays hard over a submission's life (see the lifetime
    # curve below), and new submissions are launched exactly as old ones decay,
    # so k correlates with "how many young subs are live" rather than with any
    # sharing rule. The two tables even disagree — all-days says per-sub RISES
    # from k=1 to k=4, interior-days says it falls — which is what an
    # unidentified confound looks like.
    #
    # The age-controlled test: a submission's DAY-0 burst, conditioned on how
    # many OTHER submissions were live that day. Age is held fixed at 0 by
    # construction, so a team-level split has nowhere to hide.
    print("\nAGE-CONTROLLED SPLIT TEST — day-0 burst vs concurrent submissions")
    print("  (age fixed at 0, so this isolates sharing from decay)")
    print(f"  {'others live':>12} {'subs':>5} {'mean day-0':>11} {'day-0 games':>32}")
    d0: dict[int, list[int]] = defaultdict(list)
    for sid, days in per.items():
        first = sorted(days)[0]
        others = len([s for s in live_on[first] if s != sid])
        d0[others].append(days[first])
    verdict = "INCONCLUSIVE — day-0 observed at only one concurrency level"
    for o in sorted(d0):
        v = sorted(d0[o])
        print(f"  {o:>12} {len(v):>5} {sum(v) / len(v):>11.1f}"
              f" {str(v):>32}")
    if len(d0) > 1:
        lo, hi = min(d0), max(d0)
        m_lo = sum(d0[lo]) / len(d0[lo])
        m_hi = sum(d0[hi]) / len(d0[hi])
        ratio = m_hi / max(m_lo, 1e-9)
        # Under a hard team split the day-0 burst would fall roughly as
        # (1+lo)/(1+hi) once (1+others) submissions share the same budget.
        split_pred = (1 + lo) / (1 + hi)
        print(f"\n  {lo} others -> {hi} others: day-0 burst ratio {ratio:.2f} "
              f"(hard split predicts {split_pred:.2f}; no split predicts 1.00)")
        verdict = ("PER-SUBMISSION — parallel reads are REAL. A submission's "
                   "day-0 burst does not shrink when others are live."
                   if ratio >= (split_pred ** 0.5) else
                   "SPLIT — day-0 bursts shrink with concurrency; parallel "
                   "reads are an illusion (plan §4 kill: sequential ships).")
    print(f"\n  VERDICT (age-controlled): {verdict}")

    # ---- the lifetime curve: the constraint the plan never priced ---------
    today = max(max(d for d in days) for days in per.values())
    print("\n" + "=" * 78)
    print("LIFETIME ACCRUAL CURVE — games by AGE of the submission, not by date")
    print("=" * 78)
    by_age: dict[int, list[int]] = defaultdict(list)
    curves = {}
    for sid, days in sorted(per.items()):
        ds = sorted(days)
        span = daterange(ds[0], ds[-1])
        curve = [days.get(d, 0) for d in span]
        curves[sid] = curve
        # A submission still accruing today has a censored tail: its last day
        # is partial and its curve is not finished. Excluded from the age
        # profile so a half-lived sub does not read as a decayed one.
        if ds[-1] < today:
            for age, g in enumerate(curve):
                by_age[age].append(g)
    print(f"  {'age(d)':>7} {'subs':>5} {'mean':>7} {'median':>7} {'max':>5}")
    for age in sorted(by_age):
        v = sorted(by_age[age])
        print(f"  {age:>7} {len(v):>5} {sum(v) / len(v):>7.1f}"
              f" {v[len(v) // 2]:>7} {v[-1]:>5}")

    finished = {s: sum(c) for s, c in curves.items()
                if sorted(per[s])[-1] < today}
    live_now = {s: sum(c) for s, c in curves.items()
                if sorted(per[s])[-1] >= today}
    tot = sorted(finished.values())
    print(f"\n  LIFETIME TOTAL over {len(tot)} FINISHED submissions:"
          f"  min {tot[0]}  median {tot[len(tot) // 2]}  max {tot[-1]}")
    print(f"  still accruing: "
          + ", ".join(f"{s} at {g}" for s, g in sorted(live_now.items())))

    # ---- read bandwidth ---------------------------------------------------
    days_left = (date.fromisoformat(a.deadline)
                 - date.fromisoformat(today)).days
    print("\n" + "=" * 78)
    print(f"READ BANDWIDTH  (G-10 declared read n={a.read_n}, "
          f"deadline {a.deadline}, {days_left} days left)")
    print("=" * 78)
    reached = [s for s, g in {**finished, **live_now}.items() if g >= a.read_n]
    if reached:
        print(f"  submissions that reached n={a.read_n}: {reached}")
    else:
        print(f"  *** NO submission in campaign history has reached "
              f"n={a.read_n}. ***")
        print(f"      Best ever: {tot[-1]} games. Median finished: "
              f"{tot[len(tot) // 2]}.")
        print(f"      A declared read-n above the lifetime cap is not a "
              f"dwell, it is an unreachable bar: the sub dies before the")
        print(f"      sample arrives, so the read never resolves and the "
              f"slot is spent anyway.")
        # What IS attainable, at the observed cap.
        attain = tot[len(tot) // 2]
        print(f"\n  attainable per-submission n (median finished): ~{attain}")
        print(f"  ...which at a true 50% baseline gives a 95% CI of about "
              f"+/-{1.96 * (0.25 / attain) ** 0.5 * 100:.1f}pp")
        print(f"     = roughly +/-{1.96 * (0.25 / attain) ** 0.5 * 700:.0f} "
              f"implied ELO (ELO_PER_WR=700, scripts/live_ci.py)")
    lifetimes = [len(c) for s, c in curves.items() if s in finished]
    lifetimes.sort()
    med_life = lifetimes[len(lifetimes) // 2]
    print(f"\n  median submission LIFETIME: {med_life} days"
          f"  -> ~{days_left // med_life} sequential ship-and-read cycles left")
    if verdict.startswith("PER-SUBMISSION"):
        print(f"  accrual is per-submission, so cycles can OVERLAP: the limit "
              f"is the {days_left}-day calendar, not bandwidth.")

    # ---- freshness --------------------------------------------------------
    stale = [(s, newest[s]) for s in sorted(newest)
             if (datetime.fromisoformat(today).date()
                 - datetime.fromisoformat(newest[s][:10]).date()).days
             <= a.stale_days]
    print(f"\n  currently-accruing subs (episode within {a.stale_days}d of "
          f"{today}): {', '.join(str(s) for s, _ in stale) or 'none'}")
    print("  NOTE: this reads the LOCAL listings cache. Refresh the live subs "
          "(delete their parquet, re-run the monitor notebook) before treating "
          "today's numbers as final.")

    OUT.write_text(json.dumps(
        dict(per_sub=rows, split_all_days=all_k, split_interior=int_k,
             verdict=verdict, read_n=a.read_n, asof=today), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
