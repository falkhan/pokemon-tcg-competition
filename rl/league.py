"""Persistent openskill league over (deck, pilot) entries (M7.2, M7-plan §4).

Every deck and pilot the project produces is rated on ONE persistent PlackettLuce
table (data/league/league.json) against a FIXED anchor population — the anti-
mirror-overfit field that every prior search lacked (DECISIONS.md 2026-07-08:
mirror-only hill-climbing and small-sample openskill both promoted worse decks).

Two entry classes:
- Anchors (frozen=True): permanent members that pin the rating scale across weeks
  — the rule experts, tuned-Lucario, generic pilot on both known decks, bc_v1, and
  random; plus the harvested-meta decks after M7.0 (versioned meta_vN). "Frozen"
  means permanent membership and exclusion from candidate scheduling/promotion;
  their RATINGS still update — the self-test needs anchors rated among themselves,
  and the scale is pinned by the population being permanent, not by fixing mu.
- Candidates: new decks (piloted by generic) and new pilots, rated vs every anchor
  first (games_per_anchor each), then vs nearest peers.

Draws are skipped for rating (as in rate_population) but count toward `games`
and as half a win in win rates (as in field_fitness). Gates are DATA, not
prints: `gate --entry <id>` writes data/league/gates/<entry>.json with
thresholds, n, measured values, and the field version, so the diary can cite
them. Ship-eligibility additionally requires the rl.gate suite on the exported
bundle — that runs on the bundle artifact and is never auto-run from here.

Usage:
  python -m rl.league init [--meta data/kaggle/meta_v1]
  python -m rl.league add --deck decks/gen/deck_<hash>.csv --pilot generic
  python -m rl.league run --games-per-anchor 40 --workers 4
  python -m rl.league standings
  python -m rl.league gate --entry generic+lucario [--n-scale 0.25]
  python -m rl.league promote --deck <csv|name>
  python -m rl.league selftest [--games 40]
  python -m rl.league log-submission --sub 54474043 --entry <id>
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

from rl.deck_search import validate_deck
from rl.kaggle_ingest import _write_deck_csv, deck_hash
from rl.matchrunner import OpponentSpec, resolve_deck, series_wr, spec_deck

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LEAGUE_DIR = DATA / "league"
LEAGUE_JSON = LEAGUE_DIR / "league.json"
LEAGUE_DECKS = LEAGUE_DIR / "decks"
GATES_DIR = LEAGUE_DIR / "gates"
SUBMISSIONS_JSON = LEAGUE_DIR / "submissions.json"

GAMES_PER_ANCHOR = 40
TOP_PEERS = 3

# The plan-§4 anchor seven. tuned pilots the Lucario deck (no decks/tuned.csv).
ANCHORS: list[tuple[str, OpponentSpec]] = [
    ("lucario_expert", ("rule", "lucario", "lucario")),
    ("iono_expert", ("rule", "iono", "iono")),
    ("tuned_lucario", ("rule", "tuned", "lucario")),
    ("generic+lucario", ("generic", "lucario")),
    ("generic+iono", ("generic", "iono")),
    ("bc_v1+kyogre", ("model", str(ROOT / "checkpoints" / "bc_v1.pt"), "kyogre")),
    ("random+kyogre", ("random", "kyogre")),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclasses.dataclass
class Entry:
    entry_id: str
    pilot: OpponentSpec      # deck slot normalized to data/league/decks/<hash>.csv
    deck_hash: str
    mu: float
    sigma: float
    games: int
    frozen: bool             # anchor: permanent + excluded from candidate scheduling
    origin: str              # "anchor"|"template"|"hillclimb"|"harvested"|"checkpoint"

    def ordinal(self) -> float:
        return self.mu - 3 * self.sigma


@dataclasses.dataclass
class League:
    entries: dict[str, Entry]
    meta: dict               # field_version, champion_deck, shipped_entry, updated_at


def new_league() -> League:
    champion = deck_hash(resolve_deck("lucario"))    # the tuned seed is the bar to beat
    return League(entries={}, meta={"field_version": "base_v1",
                                    "champion_deck": champion,
                                    "shipped_entry": None,
                                    "updated_at": _now()})


def load_league(path: Path | None = None) -> League:
    raw = json.loads(Path(path or LEAGUE_JSON).read_text())
    entries = {eid: Entry(**{**e, "pilot": tuple(e["pilot"])})
               for eid, e in raw["entries"].items()}
    return League(entries=entries, meta=raw["meta"])


def save_league(league: League, path: Path | None = None) -> None:
    league.meta["updated_at"] = _now()
    path = Path(path or LEAGUE_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "entries": {eid: dataclasses.asdict(e) for eid, e in league.entries.items()},
        "meta": league.meta,
    }, indent=2))


def register_deck(deck) -> tuple[str, Path]:
    """Validate + snapshot a deck under data/league/decks/<hash>.csv."""
    ids = resolve_deck(deck)
    legal, reasons = validate_deck(ids)
    if not legal:
        raise ValueError(f"illegal deck: {reasons}")
    h = deck_hash(ids)
    csv = LEAGUE_DECKS / f"{h}.csv"
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        _write_deck_csv(csv, ids)
    return h, csv


def _normalize_spec(spec: OpponentSpec, csv: Path) -> OpponentSpec:
    """Rewrite the spec's deck slot to the registered csv path (a plain string),
    so league.json round-trips cleanly and specs stay hashable tuples."""
    if spec[0] in ("rule", "model"):
        return (spec[0], spec[1], str(csv))
    return (spec[0], str(csv))


def add_entry(league: League, spec: OpponentSpec, entry_id: str | None = None,
              frozen: bool = False, origin: str = "template") -> Entry:
    """Register the spec's deck and add the entry (idempotent by entry_id)."""
    h, csv = register_deck(spec_deck(spec))
    entry_id = entry_id or f"{spec[0]}+{Path(str(spec_deck(spec))).stem}"
    if entry_id in league.entries:
        return league.entries[entry_id]
    from openskill.models import PlackettLuce
    r = PlackettLuce().rating(name=entry_id)
    e = Entry(entry_id=entry_id, pilot=_normalize_spec(spec, csv), deck_hash=h,
              mu=r.mu, sigma=r.sigma, games=0, frozen=frozen, origin=origin)
    league.entries[entry_id] = e
    return e


def bootstrap_anchors(league: League) -> list[Entry]:
    """Add the plan-§4 anchors (idempotent). A missing bc_v1 checkpoint skips
    that anchor with a warning rather than failing the whole bootstrap."""
    added = []
    for entry_id, spec in ANCHORS:
        if spec[0] == "model" and not Path(spec[1]).exists():
            print(f"WARNING: {entry_id}: checkpoint {spec[1]} missing — anchor skipped",
                  flush=True)
            continue
        added.append(add_entry(league, spec, entry_id=entry_id, frozen=True,
                               origin="anchor"))
    return added


def add_meta_anchors(league: League, snapshot: Path) -> list[Entry]:
    """Anchor the harvested meta_vN decks (generic-piloted) and bump the field
    version — promotions must re-hold after this (M7-plan §4)."""
    manifest = json.loads((Path(snapshot) / "manifest.json").read_text())
    added = []
    for d in manifest["decks"]:
        spec = ("generic", str(Path(snapshot) / d["csv"]))
        entry_id = f"generic+{d['archetype']}@{manifest['version']}"
        added.append(add_entry(league, spec, entry_id=entry_id, frozen=True,
                               origin="harvested"))
    league.meta["field_version"] = manifest["version"]
    return added


# ---------------------------------------------------------------------------
# Rating + scheduling + run
# ---------------------------------------------------------------------------
def record_result(league: League, id_a: str, id_b: str, result: int) -> None:
    """Apply one game (0 = a won, 1 = b won, 2 = draw). Draws skip the rating
    update (rate_population convention) but count toward games. Anchor ratings
    update too — see the module docstring on frozen semantics."""
    a, b = league.entries[id_a], league.entries[id_b]
    a.games += 1
    b.games += 1
    if result == 2:
        return
    from openskill.models import PlackettLuce
    model = PlackettLuce()
    win, lose = (a, b) if result == 0 else (b, a)
    res = model.rate([[model.rating(mu=win.mu, sigma=win.sigma, name=win.entry_id)],
                      [model.rating(mu=lose.mu, sigma=lose.sigma, name=lose.entry_id)]])
    win.mu, win.sigma = res[0][0].mu, res[0][0].sigma
    lose.mu, lose.sigma = res[1][0].mu, res[1][0].sigma


def schedule(league: League, games_per_anchor: int = GAMES_PER_ANCHOR,
             top_peers: int = TOP_PEERS, peer_games: int = GAMES_PER_ANCHOR
             ) -> list[tuple[str, str, int]]:
    """(candidate, opponent, n_games) pairs. Phase A: a candidate that hasn't
    finished its anchor schedule plays EVERY anchor. Phase B: established
    candidates play their nearest-ordinal peers. Interrupted runs may replay
    some anchor games — extra n only sharpens ratings; per-pair bookkeeping is
    deliberately omitted."""
    anchors = [e for e in league.entries.values() if e.frozen]
    candidates = [e for e in league.entries.values() if not e.frozen]
    pairs = []
    for cand in candidates:
        if cand.games < games_per_anchor * len(anchors):
            pairs += [(cand.entry_id, a.entry_id, games_per_anchor) for a in anchors]
        else:
            peers = sorted((p for p in candidates if p.entry_id != cand.entry_id),
                           key=lambda p: abs(p.ordinal() - cand.ordinal()))
            pairs += [(cand.entry_id, p.entry_id, peer_games)
                      for p in peers[:top_peers]]
    return pairs


def run(league: League, games_per_anchor: int = GAMES_PER_ANCHOR, workers: int = 4,
        game_fn=None) -> None:
    """Play the schedule through the match runner and persist. [ENGINE] unless
    game_fn is injected (tests)."""
    from rl.matchrunner import run_pairs
    sched = schedule(league, games_per_anchor)
    if not sched:
        print("nothing to schedule (no candidates)", flush=True)
        return
    pairs = [(league.entries[a].pilot, league.entries[b].pilot, n) for a, b, n in sched]
    for (a, b, _), results in zip(sched, run_pairs(pairs, workers=workers, game_fn=game_fn)):
        for r in results:
            record_result(league, a, b, r)
        print(f"{a} vs {b}: wr={series_wr(results):.3f} (n={len(results)})", flush=True)
    save_league(league)


def standings(league: League) -> list[Entry]:
    ordered = sorted(league.entries.values(), key=lambda e: e.ordinal(), reverse=True)
    print(f"{'entry':<36} {'ordinal':>8} {'mu':>7} {'sigma':>6} {'games':>6}  origin")
    for e in ordered:
        tag = "*" if e.frozen else " "
        print(f"{tag}{e.entry_id:<35} {e.ordinal():>8.2f} {e.mu:>7.2f} "
              f"{e.sigma:>6.2f} {e.games:>6}  {e.origin}", flush=True)
    return ordered


# ---------------------------------------------------------------------------
# Gates as data (M7-plan §3.3) — each gate takes an injectable play callable
# ---------------------------------------------------------------------------
def _default_play(spec_a, spec_b, n, seed, stats):
    from rl.matchrunner import play_series
    return play_series(spec_a, spec_b, n, seed=seed, stats=stats)


def _gate(threshold, n, value, passed, **extra) -> dict:
    return {"threshold": threshold, "n": n, "value": round(float(value), 4),
            "pass": bool(passed), "date": _now(), **extra}


def _held_out_decks(league: League, k: int = 5) -> list[str]:
    """G5 default: decks/gen candidates NOT in the league (unseen by definition)."""
    gen = ROOT / "decks" / "gen"
    if not (gen / "manifest.json").exists():
        return []
    in_league = {e.deck_hash for e in league.entries.values()}
    out = [str(gen / d["csv"]) for d in json.loads((gen / "manifest.json").read_text())["decks"]
           if d["hash"] not in in_league]
    return out[:k]


def _redeck(pilot: OpponentSpec, deck: str) -> OpponentSpec:
    """The same pilot brain on a different deck (G5)."""
    return (pilot[0], pilot[1], deck) if pilot[0] in ("rule", "model") else (pilot[0], deck)


def run_gates(league: League, entry_id: str, play=None, held_out: list | None = None,
              n_scale: float = 1.0) -> dict:
    """Run G1-G6 for an entry and write data/league/gates/<entry>.json.

    play(spec_a, spec_b, n, seed, stats) -> list[int]; the default is the
    [ENGINE] match runner. n_scale shrinks every n for smoke runs and is
    recorded in the JSON so scaled-down records are honestly labeled.
    """
    play = play or _default_play
    e = league.entries[entry_id]
    deck = spec_deck(e.pilot)
    scaled = lambda n: max(2, int(n * n_scale))  # noqa: E731
    stats: dict = {}

    # G1 legality/safety: play vs random; errors are counted by the runner's
    # stats and any exception fails the gate outright.
    g1_n, g1_errors = scaled(200), 0
    try:
        play(e.pilot, ("random", deck), g1_n, 11, stats)
        g1_errors = stats.get("a", {}).get("errors", 0)
    except Exception as exc:  # noqa: BLE001 — the gate's whole job is to catch this
        g1_errors = stats.get("a", {}).get("errors", 0) or 1
        print(f"G1 crash: {exc}", flush=True)
    g1 = _gate(0, g1_n, g1_errors, g1_errors == 0)

    # G2-G4: win-rate gates (draws count 0.5, series_wr). Stats keep accumulating
    # into the same dict — G6 reads the totals.
    g2_r = play(e.pilot, ("random", deck), scaled(200), 12, stats)
    g2 = _gate(0.90, len(g2_r), series_wr(g2_r), series_wr(g2_r) >= 0.90)
    g3_r = play(e.pilot, ("generic", deck), scaled(400), 13, stats)
    g3 = _gate(0.55, len(g3_r), series_wr(g3_r), series_wr(g3_r) >= 0.55)
    g4_r = play(e.pilot, ("rule", "lucario", "lucario"), scaled(400), 14, stats)
    g4 = _gate(0.35, len(g4_r), series_wr(g4_r), series_wr(g4_r) >= 0.35)

    # G5 unseen-deck generalization: pilot re-decked on held-out decks vs a fixed
    # reference, compared to the generic pilot on the same deck (>= generic - 3pp).
    ref = ("generic", "lucario")
    per_deck, deltas = {}, []
    for h in (held_out if held_out is not None else _held_out_decks(league)):
        mine = series_wr(play(_redeck(e.pilot, h), ref, scaled(200), 15, None))
        base = series_wr(play(("generic", h), ref, scaled(200), 16, None))
        deltas.append(mine - base)
        per_deck[Path(h).stem] = {"pilot_wr": round(mine, 4), "generic_wr": round(base, 4)}
    g5 = (_gate(-0.03, scaled(200), min(deltas), min(deltas) >= -0.03,
                reference=str(ref), per_deck=per_deck)
          if deltas else _gate(-0.03, 0, 0.0, False, reason="no held-out decks"))

    # G6 time budget from the stats accumulated over G1-G4: mean move < 50ms.
    # No timing data (an injected play that skips stats) fails honestly.
    a = stats.get("a", {"moves": 0, "time_s": 0.0})
    mean_ms = 1000.0 * a["time_s"] / a["moves"] if a["moves"] else 0.0
    g6 = _gate(50.0, a["moves"], mean_ms, a["moves"] > 0 and mean_ms < 50.0,
               note="mean move ms over G1-G4; p99 game overage measured on-Kaggle only")

    record = {"entry_id": entry_id, "field_version": league.meta["field_version"],
              "n_scale": n_scale, "date": _now(),
              "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}}
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    (GATES_DIR / f"{entry_id}.json").write_text(json.dumps(record, indent=2))
    for name, g in record["gates"].items():
        print(f"{name}: {'PASS' if g['pass'] else 'FAIL'} "
              f"(value={g['value']}, threshold={g['threshold']}, n={g['n']})", flush=True)
    return record


# ---------------------------------------------------------------------------
# Promotion + submissions log (M7-plan §4)
# ---------------------------------------------------------------------------
def anchor_field(league: League) -> list[OpponentSpec]:
    return [e.pilot for e in league.entries.values() if e.frozen]


def promote_deck(league: League, deck, games_per_opp: int = 60,
                 fitness_fn=None, play=None) -> dict:
    """Deck promotion: field_fitness >= 0.55 vs the full anchor field under the
    generic pilot AND >= 0.50 vs the current champion (same pilot, n=400).
    Updates meta.champion_deck on pass. [ENGINE] unless fns injected."""
    ids = resolve_deck(deck)
    h, _ = register_deck(ids)
    if fitness_fn is None:
        from rl.deck_search import field_fitness
        fitness_fn = field_fitness
    play = play or _default_play

    fitness, per_opp = fitness_fn(ids, anchor_field(league), games_per_opp=games_per_opp)
    champ = league.meta["champion_deck"]
    champ_r = play(("generic", ids), ("generic", str(LEAGUE_DECKS / f"{champ}.csv")),
                   400, 17, None)
    vs_champ = series_wr(champ_r)
    promoted = fitness >= 0.55 and vs_champ >= 0.50
    record = {"deck_hash": h, "fitness": round(fitness, 4), "vs_champion": round(vs_champ, 4),
              "thresholds": {"fitness": 0.55, "vs_champion": 0.50},
              "promoted": promoted, "field_version": league.meta["field_version"],
              "per_opponent": per_opp, "date": _now()}
    if promoted:
        league.meta["champion_deck"] = h
        save_league(league)
    print(f"deck {h[:12]}: fitness={fitness:.3f} vs_champion={vs_champ:.3f} "
          f"-> {'PROMOTED' if promoted else 'rejected'}", flush=True)
    return record


def ship_eligible(gates_record: dict, deck_record: dict, vs_shipped_wr: float
                  ) -> tuple[bool, list[str]]:
    """Pure ship-eligibility arithmetic per M7-plan §4. The rl.gate suite on the
    exported bundle is a separate, manual step — never auto-run from here."""
    reasons = []
    failed = [n for n, g in gates_record["gates"].items() if not g["pass"]]
    if failed:
        reasons.append(f"gates failed: {failed}")
    if not deck_record["promoted"]:
        reasons.append("deck not promoted")
    if vs_shipped_wr < 0.55:
        reasons.append(f"only {vs_shipped_wr:.2f} vs shipped pair (need >= 0.55)")
    if not reasons:
        reasons.append("eligible — now run `python -m rl.gate` on the exported bundle")
    return (not failed and deck_record["promoted"] and vs_shipped_wr >= 0.55), reasons


def log_submission(submission_id: int, entry_id: str, league: League, note: str = "") -> None:
    """Append to data/league/submissions.json — every submission id feeds ingestion."""
    rows = json.loads(SUBMISSIONS_JSON.read_text()) if SUBMISSIONS_JSON.exists() else []
    rows.append({"submission_id": submission_id, "entry_id": entry_id,
                 "deck_hash": league.entries[entry_id].deck_hash,
                 "date": _now(), "note": note})
    SUBMISSIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_JSON.write_text(json.dumps(rows, indent=2))


# ---------------------------------------------------------------------------
# Self-test: anchors must reproduce the M6 ordering (M7-plan §7 M7.2 gate)
# ---------------------------------------------------------------------------
def anchor_round_robin(league: League, games: int = 40, workers: int = 4,
                       game_fn=None) -> None:
    """All anchor pairs x games, rated in place. [ENGINE] unless game_fn given."""
    from rl.matchrunner import run_pairs
    ids = [e.entry_id for e in league.entries.values() if e.frozen]
    sched = [(ids[i], ids[j], games) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    pairs = [(league.entries[a].pilot, league.entries[b].pilot, n) for a, b, n in sched]
    for (a, b, _), results in zip(sched, run_pairs(pairs, workers=workers, game_fn=game_fn)):
        for r in results:
            record_result(league, a, b, r)


def check_anchor_ordering(ordinals: dict[str, float]) -> tuple[bool, list[str]]:
    """The M6-measured ordering: {lucario_expert, tuned_lucario} > generic+lucario
    > bc_v1+kyogre > random+kyogre. Soft on expert-vs-tuned (a statistical tie at
    M5's n). Iono anchors are present but unordered here (different deck)."""
    problems = []
    experts = min(ordinals.get("lucario_expert", 0), ordinals.get("tuned_lucario", 0))
    chain = [("experts (min of lucario/tuned)", experts),
             ("generic+lucario", ordinals.get("generic+lucario", 0)),
             ("bc_v1+kyogre", ordinals.get("bc_v1+kyogre", 0)),
             ("random+kyogre", ordinals.get("random+kyogre", 0))]
    for (name_hi, hi), (name_lo, lo) in zip(chain, chain[1:]):
        if hi <= lo:
            problems.append(f"{name_hi} ({hi:.2f}) should outrank {name_lo} ({lo:.2f})")
    return not problems, problems


def selftest(games: int = 40, workers: int = 4, game_fn=None) -> bool:
    """Fresh in-memory league -> anchors -> round-robin -> ordering check. [ENGINE]"""
    league = new_league()
    bootstrap_anchors(league)
    anchor_round_robin(league, games=games, workers=workers, game_fn=game_fn)
    ordinals = {e.entry_id: e.ordinal() for e in standings(league)}
    ok, problems = check_anchor_ordering(ordinals)
    print("selftest: " + ("PASS — anchors reproduce the M6 ordering" if ok
                          else "FAIL: " + "; ".join(problems)), flush=True)
    return ok


def _load_or_init() -> League:
    if LEAGUE_JSON.exists():
        return load_league()
    league = new_league()
    bootstrap_anchors(league)
    save_league(league)
    return league


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create league.json with the anchor field")
    s.add_argument("--meta", type=Path, default=None, help="also anchor a meta_vN snapshot")

    s = sub.add_parser("add", help="add a candidate (deck, pilot) entry")
    s.add_argument("--deck", required=True)
    s.add_argument("--pilot", default="generic", help="generic | rule:X | model:<ckpt>")
    s.add_argument("--id", dest="entry_id", default=None)
    s.add_argument("--origin", default="template")

    s = sub.add_parser("run", help="play the schedule and update ratings")
    s.add_argument("--games-per-anchor", type=int, default=GAMES_PER_ANCHOR)
    s.add_argument("--workers", type=int, default=4)

    sub.add_parser("standings", help="print the table")

    s = sub.add_parser("gate", help="run G1-G6 for an entry, write gates/<entry>.json")
    s.add_argument("--entry", required=True)
    s.add_argument("--n-scale", type=float, default=1.0)

    s = sub.add_parser("promote", help="deck promotion vs anchor field + champion")
    s.add_argument("--deck", required=True)

    s = sub.add_parser("selftest", help="anchors reproduce the M6 ordering")
    s.add_argument("--games", type=int, default=40)
    s.add_argument("--workers", type=int, default=4)

    s = sub.add_parser("log-submission", help="record a Kaggle submission id")
    s.add_argument("--sub", type=int, required=True)
    s.add_argument("--entry", required=True)
    s.add_argument("--note", default="")

    a = p.parse_args()
    if a.cmd == "init":
        league = _load_or_init()
        if a.meta:
            add_meta_anchors(league, a.meta)
        save_league(league)
        print(f"{LEAGUE_JSON}: {len(league.entries)} entries "
              f"(field {league.meta['field_version']})", flush=True)
    elif a.cmd == "add":
        league = _load_or_init()
        if a.pilot in ("generic", "random"):
            spec: OpponentSpec = (a.pilot, a.deck)
        else:                                   # "rule:X" / "model:<ckpt>"
            kind, ident = a.pilot.split(":", 1)
            spec = (kind, ident, a.deck)
        e = add_entry(league, spec, entry_id=a.entry_id, origin=a.origin)
        save_league(league)
        print(f"added {e.entry_id} (deck {e.deck_hash[:12]})", flush=True)
    elif a.cmd == "run":
        run(_load_or_init(), games_per_anchor=a.games_per_anchor, workers=a.workers)
    elif a.cmd == "standings":
        standings(_load_or_init())
    elif a.cmd == "gate":
        league = _load_or_init()
        run_gates(league, a.entry, n_scale=a.n_scale)
    elif a.cmd == "promote":
        league = _load_or_init()
        promote_deck(league, a.deck)
    elif a.cmd == "selftest":
        raise SystemExit(0 if selftest(games=a.games, workers=a.workers) else 1)
    elif a.cmd == "log-submission":
        league = _load_or_init()
        log_submission(a.sub, a.entry, league, note=a.note)
        print(f"logged submission {a.sub} -> {a.entry}", flush=True)


if __name__ == "__main__":
    _main()
