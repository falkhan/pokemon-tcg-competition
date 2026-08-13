"""M44 league — round-robin (pilot, deck) pairs, Bradley-Terry ELO ranking.

The selection instrument of docs/M44-plan.md r4: every unordered player pair
is one CELL (run_pairs alternates seats internally — enumerating both orders
would double the games for nothing), cells are cached as one jsonl each under
--out, and the ranking is a Bradley-Terry MLE fit (Zermelo/MM iteration) on
the pooled grid, rescaled so the `pin` anchor sits at exactly its pinned
rating (tuned = 1000).

Cache design (registered deviation from the r4 text): run_pairs' header pins
n, so "upgrading" an n=200 cell to n=800 in place would header-mismatch.
Instead each (n, seed) is its own cell FILE —
    cell_<idA>__<idB>__<digest8>__n<N>_s<S>.jsonl
and the decode POOLS every file matching the pair's current identity digest
(digest8 = sha256(md5A|deckA|md5B|deckB) over id-sorted players). A promoted
net changes the digest, so stale cells fall out of the glob instead of
tripping run_pairs' header refusal; `--total 800` after `--total 200` plays
one 600-game top-up cell at the next seed. Never run two league invocations
concurrently (run_pairs appends without locking).

Loss-cause columns read the `reasons` key the M44 matchrunner persists
(engine RESULT reason: 1=prizes 2=deckout 3=benchout 4=effect, None='?').

Registered stats (docs/M44-plan.md): n=200/cell intermediate rankings are
progress prints (per-cell CI ±6.9pp); the final n=800/cell table is the
selection instrument; top-2-of-4 fitted ELOs carry winner's-curse inflation
and are never quoted as unbiased strength. The league ELO is a CLOSED-
population measure, not a ladder predictor.

Usage:
  uv run python scripts/m44_league.py run  --roster docs/specs/m44_roster.json \
      --total 200 [--seed 0] [--workers 8] [--out data/m44_league] [--hermes]
  uv run python scripts/m44_league.py rank --roster docs/specs/m44_roster.json \
      [--out data/m44_league] [--label "after leg K"] [--hermes]

`run` plays the missing games then prints the ranking; `rank` only decodes.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CAUSE_BY_REASON = {1: "prizes", 2: "deckout", 3: "benchout", 4: "effect"}
CAUSE_COLS = ("prizes", "deckout", "benchout", "effect", "?")
LN10_400 = math.log(10.0) / 400.0


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_roster(path: Path) -> list[dict]:
    """Validate + return players. Hard-fails (SystemExit) on: duplicate ids,
    a spec parse_spec rejects, a model net whose md5 does not match its pin,
    or a pin count != 1. Rule anchors carry md5 'rule'."""
    from rl.matchrunner import parse_spec

    players = json.loads(path.read_text())["players"]
    ids = [p["id"] for p in players]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"roster: duplicate ids in {ids}")
    pins = [p for p in players if "pin" in p]
    if len(pins) != 1:
        raise SystemExit(f"roster: exactly one pinned anchor required, got "
                         f"{[p['id'] for p in pins]}")
    for p in players:
        try:
            parse_spec(p["spec"])
        except Exception as exc:
            raise SystemExit(f"roster: spec for {p['id']} rejected: {exc}")
        if p.get("net"):
            actual = md5(ROOT / p["net"])
            if actual != p["md5"]:
                raise SystemExit(
                    f"roster: md5 mismatch for {p['id']}: {p['net']} is "
                    f"{actual}, roster pins {p['md5']} — the net changed "
                    "under the roster; re-pin deliberately or restore the file")
    return players


def _pmd5(p: dict) -> str:
    return p.get("md5") or "rule"


def pair_digest(pa: dict, pb: dict) -> str:
    """Identity of a cell's contents, order-invariant (id-sorted)."""
    a, b = sorted((pa, pb), key=lambda p: p["id"])
    key = f"{_pmd5(a)}|{a['deck']}|{_pmd5(b)}|{b['deck']}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def cell_name(pa: dict, pb: dict, n: int, seed: int) -> str:
    a, b = sorted((pa["id"], pb["id"]))
    return f"cell_{a}__{b}__{pair_digest(pa, pb)}__n{n}_s{seed}.jsonl"


def cell_extra(pa: dict, pb: dict) -> dict:
    """Stamped into the run_pairs header: which roster player is side a (the
    id-sorted first), plus the identity md5s — a swapped checkpoint refuses
    loudly even if a stale file sits at the right name."""
    a, b = sorted((pa, pb), key=lambda p: p["id"])
    return {"a": a["id"], "b": b["id"], "a_md5": _pmd5(a), "b_md5": _pmd5(b),
            "a_deck": a["deck"], "b_deck": b["deck"]}


def read_pair_cells(out: Path, pa: dict, pb: dict) -> tuple[list[int], list]:
    """Pool results+reasons over every cell file matching the pair's CURRENT
    digest. Results are from the id-sorted-first player's perspective."""
    a_id, b_id = sorted((pa["id"], pb["id"]))
    want = cell_extra(pa, pb)
    results: list[int] = []
    reasons: list = []
    pattern = f"cell_{a_id}__{b_id}__{pair_digest(pa, pb)}__n*_s*.jsonl"
    for path in sorted(out.glob(pattern)):
        lines = [json.loads(ln) for ln in path.read_text().splitlines()
                 if ln.strip()]
        if not lines:
            continue
        got = lines[0].get("extra")
        if got != want:
            raise SystemExit(f"{path}: header extra {got} != roster {want} — "
                             "stale or tampered cell; delete it deliberately")
        for row in lines[1:]:
            results.extend(row["results"])
            reasons.extend(row.get("reasons") or [None] * len(row["results"]))
    return results, reasons


def bt_fit(w: list[list[float]], n: list[list[float]], pin_idx: int,
           pin: float, iters: int = 10000, tol: float = 1e-12
           ) -> tuple[list[float], list[float]]:
    """Pre-registered Bradley-Terry MLE (docs/M44-plan.md Audit):
    P(i beats j) = 1/(1+10^((Rj-Ri)/400)); Zermelo/MM on pi_i = 10^(Ri/400):
        pi_i <- W_i / sum_j n_ij/(pi_i+pi_j)
    draws already folded into w as half a win each side; rescale so
    R[pin_idx] == pin exactly. SE from observed Fisher information
    I_ii = (ln10/400)^2 * sum_j n_ij p_ij (1-p_ij)."""
    P = len(w)
    wi = [sum(w[i]) for i in range(P)]
    for i in range(P):
        played = sum(n[i])
        if played and (wi[i] == 0.0 or wi[i] == played):
            raise SystemExit(
                f"bt_fit: player {i} has a degenerate record "
                f"({wi[i]:g}/{played:g}) — the MLE diverges. Escalate: this "
                "means someone won/lost EVERYTHING at n>=200/cell.")
    pi = [1.0] * P
    for _ in range(iters):
        new = []
        for i in range(P):
            denom = sum(n[i][j] / (pi[i] + pi[j])
                        for j in range(P) if j != i and n[i][j] > 0)
            new.append(wi[i] / denom if denom > 0 else pi[i])
        gauge = math.exp(sum(math.log(x) for x in new) / P)
        new = [x / gauge for x in new]
        delta = max(abs(math.log(new[i] / pi[i])) for i in range(P))
        pi = new
        if delta < tol:
            break
    r = [400.0 * math.log10(x) for x in pi]
    shift = pin - r[pin_idx]
    r = [x + shift for x in r]
    se = []
    for i in range(P):
        info = sum(
            n[i][j] * (pi[i] / (pi[i] + pi[j])) * (pi[j] / (pi[i] + pi[j]))
            for j in range(P) if j != i and n[i][j] > 0)
        se.append(1.0 / math.sqrt(info) / LN10_400 if info > 0 else float("inf"))
    return r, se


def build_grid(players: list[dict], out: Path):
    """W/N grids + per-player W-L-D + loss-cause counters from the cells."""
    P = len(players)
    w = [[0.0] * P for _ in range(P)]
    n = [[0.0] * P for _ in range(P)]
    wld = [[0, 0, 0] for _ in range(P)]
    causes = [{c: 0 for c in CAUSE_COLS} for _ in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            pa, pb = players[i], players[j]
            # results are from the id-sorted-first player's perspective
            first, second = ((i, j) if sorted((pa["id"], pb["id"]))[0] == pa["id"]
                             else (j, i))
            results, reasons = read_pair_cells(out, pa, pb)
            for r, reason in zip(results, reasons):
                n[i][j] += 1
                n[j][i] += 1
                if r == 2:
                    w[first][second] += 0.5
                    w[second][first] += 0.5
                    wld[first][2] += 1
                    wld[second][2] += 1
                    continue                     # a draw has no loser
                winner, loser = (first, second) if r == 0 else (second, first)
                w[winner][loser] += 1.0
                wld[winner][0] += 1
                wld[loser][1] += 1
                causes[loser][CAUSE_BY_REASON.get(reason, "?")] += 1
    return w, n, wld, causes


def rank_table(players: list[dict], out: Path, label: str = "") -> dict:
    w, n, wld, causes = build_grid(players, out)
    pin_idx = next(i for i, p in enumerate(players) if "pin" in p)
    r, se = bt_fit(w, n, pin_idx, float(players[pin_idx]["pin"]))

    prev = None
    ranks = sorted(out.glob("ranking_*.json"))
    if ranks:
        prev = json.loads(ranks[-1].read_text())["ratings"]

    order = sorted(range(len(players)), key=lambda i: -r[i])
    lines = [f"M44 league ranking{' — ' + label if label else ''} "
             f"({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})",
             f"{'#':>2} {'id':<6} {'deck':<15} {'ELO':>7} {'±SE':>5} "
             f"{'W-L-D':>12} {'Δprev':>6}  losses pz/do/bo/eff/?"]
    for rank, i in enumerate(order, 1):
        p = players[i]
        d = (f"{r[i] - prev[p['id']]:+.0f}" if prev and p["id"] in prev else "—")
        pin_mark = "*" if i == pin_idx else " "
        c = causes[i]
        lines.append(
            f"{rank:>2} {p['id']:<6}{pin_mark}{p['deck']:<15} {r[i]:>7.1f} "
            f"{se[i]:>5.1f} "
            f"{wld[i][0]:>4}-{wld[i][1]}-{wld[i][2]:<3} {d:>6}  "
            f"{c['prizes']}/{c['deckout']}/{c['benchout']}/{c['effect']}"
            f"/{c['?']}")
    lines.append("* = pinned anchor. Closed-population ELO — not a ladder "
                 "predictor; top-2 selection inflates the winners (registered).")
    table = "\n".join(lines)
    print(table, flush=True)

    record = {"ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
              "label": label,
              "ratings": {p["id"]: r[i] for i, p in enumerate(players)},
              "se": {p["id"]: se[i] for i, p in enumerate(players)},
              "wld": {p["id"]: wld[i] for i, p in enumerate(players)},
              "loss_causes": {p["id"]: causes[i]
                              for i, p in enumerate(players)},
              "n_games": {p["id"]: int(sum(n[i]))
                          for i, p in enumerate(players)},
              "table": table}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ranking_{record['ts']}.json").write_text(
        json.dumps(record, indent=1))
    return record


def hermes_send(text: str) -> None:
    try:
        subprocess.run(["hermes", "send", "-t", "telegram", text],
                       timeout=30, check=False)
    except Exception as exc:                     # best-effort, never fatal
        print(f"hermes send failed: {exc}", file=sys.stderr)


def cmd_run(args) -> int:
    from rl.matchrunner import parse_spec, run_pairs

    players = load_roster(Path(args.roster))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pa, pb = sorted((players[i], players[j]), key=lambda p: p["id"])
            existing_cells = sorted(
                out.glob(f"cell_{pa['id']}__{pb['id']}__"
                         f"{pair_digest(pa, pb)}__n*_s*.jsonl"))
            have = len(read_pair_cells(out, pa, pb)[0])
            missing = max(0, args.total - have)
            missing += missing % 2               # keep cells slot-fair
            if not missing:
                print(f"cell {pa['id']}vs{pb['id']}: have {have} >= "
                      f"{args.total} — cached", flush=True)
                continue
            seed = args.seed + len(existing_cells)
            path = out / cell_name(pa, pb, missing, seed)
            print(f"cell {pa['id']}vs{pb['id']}: have {have}, playing "
                  f"{missing} (seed {seed}) -> {path.name}", flush=True)
            run_pairs([(parse_spec(pa["spec"]), parse_spec(pb["spec"]),
                        missing)],
                      workers=args.workers, seed=seed, checkpoint=str(path),
                      key_extra=cell_extra(pa, pb))
    record = rank_table(players, out, label=args.label)
    if args.hermes:
        hermes_send(record["table"])
    return 0


def cmd_rank(args) -> int:
    players = load_roster(Path(args.roster))
    record = rank_table(players, Path(args.out), label=args.label)
    if args.hermes:
        hermes_send(record["table"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("rank", cmd_rank)):
        s = sub.add_parser(name)
        s.add_argument("--roster", required=True)
        s.add_argument("--out", default=str(ROOT / "data/m44_league"))
        s.add_argument("--label", default="")
        s.add_argument("--hermes", action="store_true",
                       help="send the table via the Hermes telegram gateway")
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("--total", type=int, required=True,
                           help="target games per pair (tops up cached cells)")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.cmd == "run" and not 2 <= args.workers <= 8:
        p.error("--workers must be 2..8 (workers<=1 silently drops the "
                "run_pairs checkpoint; 8 is the box's proven-stable ceiling)")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
