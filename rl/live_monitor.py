"""M22a: the live measurement leg of SHIP -> FIGHT -> MEASURE -> REPLAN.

Refreshes Kaggle episodes for the tracked arms, then reports whether a
*decidable* comparison exists yet — and refuses to emit one when it does not.

The refusal is the point. M21 shipped twice on differences the instrument could
not resolve (B2 z=0.68, B3 z=0.80), and on 2026-07-20 an early live read of
"B3 601 vs B2 529" was quoted directionally when B3's score was still the mu=600
opening prior decaying (it went 600 -> 511 -> 663 -> 601 within four hours).
Both errors were available to anyone who computed the power first. So this tool
computes it first, structurally:

  * `status`  -> per-arm n / W-L / WR with CI, and "accruing k/N" when short.
  * `compare` -> emits a verdict ONLY if the observed gap exceeds the MDE at the
                 current n. Otherwise prints what n would be required and exits
                 non-zero so a cron job cannot mistake it for a result.

Live WR is the terminal benchmark because it is the only signal that is not our
own artifact (see docs/M22.md: every offline gate opponent is in the training
pool). It is also the slowest: throughput is ~60-100 episodes/day across ALL
active arms, so a 10pp gap needs ~8-13 days of wall clock at two arms.

CLI:
    uv run python -m rl.live_monitor status  --subs 54846434 54849475
    uv run python -m rl.live_monitor compare --subs 54846434 54849475 [--mde-pp 10]
    uv run python -m rl.live_monitor cron    --subs 54846434 54849475 [--notify]
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass

import polars as pl

from rl.kaggle_ingest import EPISODES_PQ, refresh

# 80% power, alpha=0.05 two-sided: (z_{a/2} + z_b)^2
Z_SUM_SQ = (1.959964 + 0.8416) ** 2

# A fresh submission opens at mu=600 and decays over ~a day; anything below this
# is dominated by the opening prior, not by play. See .claude/skills/measure-agent.
MIN_N_FOR_ANY_READ = 30


@dataclass(frozen=True)
class Arm:
    sub: int
    n: int
    wins: int
    losses: int
    draws: int
    score: float
    last_seen: str

    @property
    def wr(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.n if self.n else 0.0

    @property
    def se(self) -> float:
        return math.sqrt(0.25 / self.n) if self.n else float("inf")

    def ci95(self) -> tuple[float, float]:
        return (max(0.0, self.wr - 1.96 * self.se), min(1.0, self.wr + 1.96 * self.se))


def mde(n_a: int, n_b: int) -> float:
    """Minimum detectable difference in WR at 80% power, both arms measured."""
    if not n_a or not n_b:
        return float("inf")
    return math.sqrt(Z_SUM_SQ * 0.25 * (1 / n_a + 1 / n_b))


def n_needed(delta: float) -> int:
    """Games per arm to resolve a true WR gap of `delta` at 80% power."""
    if delta <= 0:
        return sys.maxsize
    return math.ceil(2 * Z_SUM_SQ * 0.25 / delta**2)


def read_arms(subs: list[int]) -> list[Arm]:
    df = pl.read_parquet(EPISODES_PQ)
    me = (pl.when(pl.col("our_seat") == 0).then(pl.col("submission_id_0"))
            .otherwise(pl.col("submission_id_1")).alias("sub"))
    rw = (pl.when(pl.col("our_seat") == 0).then(pl.col("reward_0"))
            .otherwise(pl.col("reward_1")).alias("rw"))
    orw = (pl.when(pl.col("our_seat") == 0).then(pl.col("reward_1"))
             .otherwise(pl.col("reward_0")).alias("orw"))
    sc = (pl.when(pl.col("our_seat") == 0).then(pl.col("updated_score_0"))
            .otherwise(pl.col("updated_score_1")).alias("sc"))
    d = df.with_columns([me, rw, orw, sc]).sort("end_time")

    arms = []
    for sub in subs:
        x = d.filter(pl.col("sub") == sub)
        if not x.height:
            arms.append(Arm(sub, 0, 0, 0, 0, float("nan"), "never"))
            continue
        w = x.filter(pl.col("rw") > pl.col("orw")).height
        l = x.filter(pl.col("rw") < pl.col("orw")).height
        arms.append(Arm(sub, x.height, w, l, x.height - w - l,
                        x["sc"][-1], x["end_time"][-1][:16]))
    return arms


def fmt_status(arms: list[Arm], target: int) -> str:
    lines = ["arm            n     W-L      liveWR  95% CI            score   last seen"]
    for a in arms:
        if not a.n:
            lines.append(f"{a.sub}   -    -        -       -                 -       never")
            continue
        lo, hi = a.ci95()
        lines.append(f"{a.sub}  {a.n:>4}  {a.wins:>3}-{a.losses:<3}  "
                     f"{a.wr:>6.3f}  [{lo:.3f}, {hi:.3f}]  {a.score:>7.1f}  {a.last_seen}")
    short = [a for a in arms if a.n < target]
    if short:
        lines.append("")
        for a in short:
            lines.append(f"  accruing: {a.sub} at {a.n}/{target} "
                         f"({target - a.n} more needed)")
    return "\n".join(lines)


def compare(arms: list[Arm], mde_pp: float | None) -> tuple[bool, str]:
    """Return (decidable, report). Never asserts a winner below the MDE."""
    if len(arms) < 2:
        return False, "compare needs >=2 arms"
    a, b = arms[0], arms[1]
    out = [fmt_status(arms, MIN_N_FOR_ANY_READ), ""]

    if a.n < MIN_N_FOR_ANY_READ or b.n < MIN_N_FOR_ANY_READ:
        out.append(f"NO READ — an arm is below n={MIN_N_FOR_ANY_READ}; scores this early are "
                   f"the mu=600 opening prior decaying, not play.")
        return False, "\n".join(out)

    gap = a.wr - b.wr
    resolvable = mde(a.n, b.n)
    out.append(f"observed gap : {gap * 100:+.1f}pp  ({a.sub} minus {b.sub})")
    out.append(f"MDE at this n: {resolvable * 100:.1f}pp  (80% power, alpha=0.05)")

    if abs(gap) < resolvable:
        need = n_needed(abs(gap)) if gap else sys.maxsize
        out.append("")
        out.append(f"NO READ — the gap is INSIDE the noise floor. Resolving a true "
                   f"{abs(gap) * 100:.1f}pp gap needs n={need}/arm "
                   f"(have {a.n} and {b.n}).")
        out.append("Reporting a winner here would repeat the M21 error. Keep accruing.")
        if mde_pp:
            out.append(f"Note: at the {mde_pp:.0f}pp design target, n={n_needed(mde_pp / 100)}"
                       f"/arm is required.")
        return False, "\n".join(out)

    winner, loser = (a, b) if gap > 0 else (b, a)
    out.append("")
    out.append(f"READ: {winner.sub} > {loser.sub} by {abs(gap) * 100:.1f}pp "
               f"(exceeds the {resolvable * 100:.1f}pp MDE).")
    return True, "\n".join(out)


def notify(text: str) -> None:
    try:
        subprocess.run(["hermes", "send", "-t", "telegram", text],
                       check=False, timeout=60, capture_output=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        print("(hermes unavailable — notification skipped)", file=sys.stderr)


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "compare", "cron"):
        s = sub.add_parser(name)
        s.add_argument("--subs", type=int, nargs="+", required=True)
        s.add_argument("--target", type=int, default=200,
                       help="accrual target per arm (default 200 ~ a 14pp read)")
        s.add_argument("--mde-pp", type=float, default=10.0,
                       help="design target gap in pp, for the 'n required' hint")
        if name == "cron":
            s.add_argument("--notify", action="store_true")
            s.add_argument("--max-new", type=int, default=300)
    a = p.parse_args()

    if a.cmd == "cron":
        refresh(our_subs=a.subs, max_new=a.max_new)

    arms = read_arms(a.subs)

    if a.cmd == "status":
        print(fmt_status(arms, a.target))
        return

    decidable, report = compare(arms, a.mde_pp)
    print(report)
    if a.cmd == "cron" and getattr(a, "notify", False):
        head = "LIVE READ AVAILABLE" if decidable else "live: still accruing (no read)"
        notify(f"{head}\n\n{report}")
    if not decidable:
        sys.exit(1)          # so cron cannot mistake "accruing" for a result


if __name__ == "__main__":
    _main()
