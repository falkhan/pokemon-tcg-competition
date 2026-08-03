"""Deck operations for the deck lab: counts, summary, warnings, IO, smoke runs.

Streamlit-free so it stays testable; the app holds no logic. Deck state here is
``dict[card_id, copies]`` rather than a 60-list — the UI edits copy counts, and
expanding to a list is a leaf operation.

Two rules that are easy to get wrong and expensive to get wrong:

* **Evolution completeness is by card NAME, never by id.** `evolves_from_id`
  names one printing; decks legally play any same-named card, and
  ``decks/lucario.csv`` runs the off-printing Riolu. An id-membership test
  reports a false "missing basic" on a deck we have shipped.
* **Games run in a SUBPROCESS.** The cg engine keeps one global mutable
  ``Battle`` per process; driving a battle inside Streamlit's rerun model would
  corrupt it. ``run_smoke`` shells out to the matchrunner CLI and parses stdout.
"""
import dataclasses
import hashlib
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tcg.cardpool import (COST_COLS, ENERGY_TYPES, RARE_CANDY_ID, attacks_by_card,
                          cards, pre_evo_names, self_starting_ids)
from tcg.deck_search import DECK_SIZE, MAX_COPIES, validate_deck

ROOT = Path(__file__).resolve().parent.parent
DECK_DIR = ROOT / "decks"
CUSTOM_DIR = DECK_DIR / "custom"          # module global so tests can repoint it
KAGGLE_DECK_DIR = ROOT / "data" / "kaggle"

# Deck names become file names — keep them boring and inside CUSTOM_DIR.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")

MAX_WORKERS = 8      # project hard rule: 12 deadlocks via libcg.so corruption
SUMMARY_RE = re.compile(
    r"^(\S+) vs (\S+): (\d+)W (\d+)L (\d+)D over (\d+) \(wr=([0-9.]+)\)")

DeckCounts = dict[int, int]


# ---------------------------------------------------------------------------
# Counts <-> ids
# ---------------------------------------------------------------------------
def deck_ids(counts: DeckCounts) -> list[int]:
    """Expand counts to a flat 60-ish list, grouped by card type then id."""
    ft = cards()
    order = sorted(counts, key=lambda c: (ft[c]["card_type"] if c in ft else 99, c))
    return [cid for cid in order for _ in range(counts[cid])]


def counts_from_ids(ids: Iterable[int]) -> DeckCounts:
    return dict(Counter(int(i) for i in ids))


def deck_size(counts: DeckCounts) -> int:
    return sum(counts.values())


def deck_hash(ids: list[int]) -> str:
    """Order-invariant sha1 — the deck identity used across the project.

    Must stay byte-identical to rl.kaggle_ingest.deck_hash (pinned in tests) so
    a deck built here joins to the harvest and census tables.
    """
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()


def copies_by_name(counts: DeckCounts) -> dict[str, int]:
    """Copies per card NAME, basic energy excluded — the real 4-copy surface.

    154 names have more than one printing, so 3x Riolu #677 + 2x Riolu #974 is
    five copies of Riolu and illegal.
    """
    ft = cards()
    out: Counter = Counter()
    for cid, n in counts.items():
        r = ft.get(cid)
        if r is not None and not r["is_basic_energy"]:
            out[r["name_norm"]] += n
    return dict(out)


def legality(counts: DeckCounts) -> tuple[bool, list[str]]:
    return validate_deck(deck_ids(counts))


# ---------------------------------------------------------------------------
# Summary + notes
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Note:
    level: str      # "error" | "warn" | "info"
    code: str
    message: str


@dataclasses.dataclass
class DeckSummary:
    n_cards: int
    kind_counts: dict[str, int]
    type_counts: dict[str, int]
    stage_counts: dict[str, int]
    pokemon_energy: dict[str, int]
    energy_cards: dict[str, int]
    energy_demand: dict[str, int]
    attack_cost_curve: dict[int, int]
    retreat_curve: dict[int, int]
    prize_liability: int
    ex_count: int
    ace_spec: str | None
    top_attackers: list[tuple[str, int, int]]


def summarize(counts: DeckCounts) -> DeckSummary:
    from tcg.cardpool import CARD_TYPES

    ft, atk = cards(), attacks_by_card()
    kind, ctype, stage = Counter(), Counter(), Counter()
    p_energy, e_cards, demand = Counter(), Counter(), Counter()
    cost_curve, retreat = Counter(), Counter()
    prize, n_ex, ace = 0, 0, None
    attackers: list[tuple[str, int, int]] = []

    for cid, n in counts.items():
        r = ft.get(cid)
        if r is None:
            continue
        kind[r["kind"]] += n
        ctype[CARD_TYPES.get(r["card_type"], "?")] += n
        if r["is_ace_spec"]:
            ace = r["name_norm"]
        if r["is_basic_energy"] or r["is_special_energy"]:
            e_cards[r["type_name"] or "Special"] += n
        if r["is_pokemon"]:
            stage[r["stage"]] += n
            p_energy[r["type_name"] or "?"] += n
            retreat[int(r["retreat_cost"] or 0)] += n
            prize += int(r["prizes_on_ko"] or 0) * n
            if r["tier"] in ("ex", "Mega ex"):
                n_ex += n
            for a in atk.get(cid, []):
                cost_curve[int(a["cost_total"] or 0)] += n
                for col, tid in COST_COLS.items():
                    c = int(a.get(col) or 0)
                    if c and tid != 0:          # colorless is payable by anything
                        demand[ENERGY_TYPES[tid]] += c * n
                attackers.append((r["name_norm"], int(a["damage"] or 0),
                                  int(a["cost_total"] or 0)))

    attackers.sort(key=lambda t: (-t[1], t[2]))
    seen, top = set(), []
    for name, dmg, cost in attackers:
        if name not in seen:
            seen.add(name)
            top.append((name, dmg, cost))
        if len(top) == 6:
            break

    return DeckSummary(
        n_cards=deck_size(counts), kind_counts=dict(kind), type_counts=dict(ctype),
        stage_counts=dict(stage), pokemon_energy=dict(p_energy),
        energy_cards=dict(e_cards), energy_demand=dict(demand),
        attack_cost_curve=dict(cost_curve), retreat_curve=dict(retreat),
        prize_liability=prize, ex_count=n_ex, ace_spec=ace, top_attackers=top,
    )


def evolution_notes(counts: DeckCounts) -> list[Note]:
    """Missing / inverted evolution lines, matched by NAME (see module docstring).

    Advisory, never an error: real cards break the rule. Cinderace starts from
    hand via its own ability, which is why every harvested `archaludon` list
    runs 4 of it behind no Raboot — see cardpool.self_starting_ids().
    """
    ft = cards()
    pre = pre_evo_names()
    have = copies_by_name(counts)
    has_rare_candy = counts.get(RARE_CANDY_ID, 0) > 0
    self_start = {ft[cid]["name_norm"] for cid in self_starting_ids() if cid in ft}
    notes: list[Note] = []

    for name, n in sorted(have.items()):
        if name in self_start:
            continue
        cur = name
        while cur in pre:
            parent = pre[cur]
            if have.get(parent, 0) == 0:
                # Rare Candy legally skips the MIDDLE stage when the Basic is there.
                grand = pre.get(parent)
                skippable = (has_rare_candy and grand is not None
                             and have.get(grand, 0) > 0)
                notes.append(Note(
                    "info" if skippable else "warn", "missing_preevo",
                    f"{n}x {name} has no {parent} in the deck"
                    + (" — legal via Rare Candy" if skippable else
                       "; those copies can only be played by evolving")))
                break
            if have[name] > have[parent]:
                notes.append(Note(
                    "warn", "line_inverted",
                    f"{have[name]}x {name} but only {have[parent]}x {parent}: "
                    "some copies can never evolve"))
            cur = parent
    # Deduplicate: a 3-stage line can report the same parent twice.
    seen, out = set(), []
    for note in notes:
        if note.message not in seen:
            seen.add(note.message)
            out.append(note)
    return out


def energy_notes(counts: DeckCounts) -> list[Note]:
    """Attacks that demand an energy type the deck cannot pay for.

    Special energy is credited: `decks/greattusk_wall.csv` runs 0 basic Fighting
    behind 11 Fighting symbols of demand and is a deck we ship beds against —
    its 8 special energy pay those costs. Demand is copies-weighted across every
    printed attack, so a 1-of tech's off-type attack shows up here too; that is
    why an uncovered type with special energy on hand is info, not a warning.
    """
    s = summarize(counts)
    n_special = s.energy_cards.get("Special", 0)
    notes = []
    for etype, demanded in sorted(s.energy_demand.items(), key=lambda kv: -kv[1]):
        if not demanded or s.energy_cards.get(etype):
            continue
        if n_special:
            notes.append(Note(
                "info", "energy_via_special",
                f"attacks demand {demanded} {etype} symbols and the deck runs no "
                f"basic {etype}; {n_special} special energy may cover it"))
        else:
            notes.append(Note(
                "warn", "no_energy_for_type",
                f"attacks demand {demanded} {etype} symbols; the deck runs "
                f"0 {etype} energy and no special energy"))
    if not sum(s.energy_cards.values()):
        notes.append(Note("warn", "no_energy", "the deck runs no energy at all"))
    return notes


def shape_notes(counts: DeckCounts) -> list[Note]:
    s = summarize(counts)
    notes = []
    n = s.n_cards
    if n != DECK_SIZE:
        notes.append(Note("error", "size",
                          f"{n} cards — {DECK_SIZE - n:+d} to legal"))
    if not s.kind_counts.get("Pokemon"):
        notes.append(Note("error", "no_pokemon", "no Pokemon in the deck"))
    if s.prize_liability and s.kind_counts.get("Pokemon"):
        notes.append(Note("info", "prize_liability",
                          f"{s.prize_liability} prizes on the board across "
                          f"{s.kind_counts['Pokemon']} Pokemon "
                          f"({s.ex_count} ex/Mega ex)"))
    return notes


def deck_notes(counts: DeckCounts) -> list[Note]:
    """All soft warnings, errors first. Legality errors come from legality()."""
    notes = shape_notes(counts) + evolution_notes(counts) + energy_notes(counts)
    rank = {"error": 0, "warn": 1, "info": 2}
    return sorted(notes, key=lambda x: rank.get(x.level, 3))


# ---------------------------------------------------------------------------
# Deck file IO
# ---------------------------------------------------------------------------
def list_deck_files() -> list[tuple[str, Path]]:
    """(label, path) for every deck on disk, grouped: decks/, custom/, gen/, harvested."""
    out: list[tuple[str, Path]] = []
    for d, pattern in ((DECK_DIR, "*.csv"), (CUSTOM_DIR, "*.csv"),
                       (DECK_DIR / "gen", "*.csv"), (KAGGLE_DECK_DIR, "*_deck.csv")):
        if not d.exists():
            continue
        for p in sorted(d.glob(pattern)):
            out.append((p.relative_to(ROOT).as_posix(), p))
    root_deck = ROOT / "deck.csv"
    if root_deck.exists():
        out.append(("deck.csv (shipped)", root_deck))
    return out


def load_counts(path: Path) -> DeckCounts:
    from tcg.decks import load_deck_file
    return counts_from_ids(load_deck_file(Path(path)))


def rel_to_root(path: Path) -> str:
    """Display path, repo-relative when it is under ROOT (tests repoint
    CUSTOM_DIR to a tmp dir, where relative_to would raise)."""
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def custom_deck_path(name: str) -> Path:
    """Where `name` would be saved. Always inside CUSTOM_DIR, never over a
    curated deck in decks/ or a harvested one in data/kaggle/."""
    if not NAME_RE.match(name or ""):
        raise ValueError(
            f"invalid deck name {name!r}: letters, digits, '_' and '-' only "
            "(must start alphanumeric, max 48 chars)")
    return CUSTOM_DIR / f"{name}.csv"


def save_deck(name: str, counts: DeckCounts, *, allow_illegal: bool = False,
              overwrite: bool = False) -> Path:
    """Write decks/custom/<name>.csv.

    Refuses to clobber an existing file unless `overwrite`. Loading
    decks/lucario.csv and saving keeps the name "lucario", which would land on
    decks/custom/lucario.csv — harmless the first time and a silent loss of
    work the second, so the caller has to mean it.
    """
    path = custom_deck_path(name)
    ok, reasons = legality(counts)
    if not ok and not allow_illegal:
        raise ValueError("deck is not legal: " + "; ".join(reasons))
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{rel_to_root(path)} already exists — "
            "rename the deck or confirm the overwrite")
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in deck_ids(counts)) + "\n",
                    encoding="utf-8")
    return path


def suggest_name(stem: str = "untitled") -> str:
    """The first `stem`, `stem_2`, `stem_3`... that is not already taken."""
    if not CUSTOM_DIR.exists() or not (CUSTOM_DIR / f"{stem}.csv").exists():
        return stem
    n = 2
    while (CUSTOM_DIR / f"{stem}_{n}.csv").exists():
        n += 1
    return f"{stem}_{n}"


# ---------------------------------------------------------------------------
# Smoke test — subprocess only
# ---------------------------------------------------------------------------
def deck_spec(path: Path, kind: str = "generic") -> str:
    """'generic:decks/custom/foo.csv'.

    parse_spec splits on ':', so a Windows absolute path ('C:\\...') produces a
    garbage spec. Always ROOT-relative POSIX; resolve_deck resolves it.
    """
    p = Path(path)
    rel = p.relative_to(ROOT) if p.is_absolute() else p
    return f"{kind}:{rel.as_posix()}"


def smoke_command(a: str, b: str, games: int = 40, workers: int = 1,
                  seed: int = 0) -> list[str]:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS} "
                         "(12 has deadlocked repeatedly via libcg.so)")
    if games < 1:
        raise ValueError("games must be >= 1")
    return [sys.executable, "-m", "rl.matchrunner", "play", "--a", a, "--b", b,
            "-n", str(games), "--workers", str(workers), "--seed", str(seed)]


def parse_series_line(stdout: str) -> dict | None:
    """The LAST matchrunner summary line, or None. --diag prints per-game lines
    before it, so first-match parsing reads a game as the series."""
    found = None
    for line in stdout.splitlines():
        m = SUMMARY_RE.match(line.strip())
        if m:
            found = {"a": m.group(1), "b": m.group(2), "w": int(m.group(3)),
                     "l": int(m.group(4)), "d": int(m.group(5)),
                     "n": int(m.group(6)), "wr": float(m.group(7))}
    return found


def run_smoke(a: str, b: str, games: int = 40, workers: int = 1, seed: int = 0,
              timeout_s: int = 300) -> dict:
    """Run a short series in a SUBPROCESS. Never drive a battle in-process."""
    cmd = smoke_command(a, b, games=games, workers=workers, seed=seed)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout_s}s", "cmd": cmd,
                "stdout": "", "stderr": ""}
    result = parse_series_line(proc.stdout or "")
    if proc.returncode != 0 or result is None:
        return {"ok": False,
                "error": f"matchrunner exited {proc.returncode}" if proc.returncode
                         else "no summary line in output",
                "cmd": cmd, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
    return {"ok": True, "result": result, "cmd": cmd,
            "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
