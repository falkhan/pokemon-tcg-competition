"""Deck factory — enumerate evolution lines, score them against the meta, and
assemble legal template decks (M7.1, docs/M7-plan.md §2).

A deck the generic pilot can execute needs four parts, built in this order:
a complete evolution line as the damage core, matched basic energy sized to the
attacker's best attack, a proven trainer shell, and optionally a tech line of a
second type chosen for weakness coverage. Every generated deck is
validate_deck-legal by construction.

Two data facts this module rests on (both verified, see tests):

1. **Evolution is by NAME, not card id.** `evolves_from_id` points at one specific
   card, but decks legally play any same-named card (the tuned Lucario deck runs
   Riolu #677 while Mega Lucario ex points at Riolu #974). Lines are walked by
   name, picking the highest-HP printing of each pre-evolution.
2. **`attackId` maps to cards by cumulative `n_attacks`** in card_id order — the
   parquets were written in one pass. This gives per-attack damage AND cost with
   no engine dependency (cards_features alone only has `min_attack_cost`, the
   *cheapest* attack, which is the wrong denominator for damage-per-energy).

The meta weakness histogram defaults to the card pool (M6's measurement:
Fire/Fighting/Lightning are the top-covered types at 220/188/155 Pokémon) and is
replaced by the harvested-meta histogram once M7.0's [NET] spike lands
(`--meta data/kaggle/opp_decks.parquet`) — the first feedback edge from
ingestion into deck building.

Usage:
  python -m rl.deck_build lines [--meta data/kaggle/opp_decks.parquet] [--top 20]
  python -m rl.deck_build shells [--harvest data/kaggle/opp_decks.parquet]
  python -m rl.deck_build generate --n 40 [--meta ...] [--top-k 10]
"""
import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import polars as pl

from rl.deck_search import DECK_SIZE, MAX_COPIES, _ft, validate_deck
from rl.kaggle_ingest import deck_hash, _write_deck_csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DECK_DIR = ROOT / "decks"
GEN_DIR = DECK_DIR / "gen"
SHELLS_JSON = DATA / "shells.json"

# Line-count templates, basic -> final (docs/M7-plan.md §2.1). The tech variants
# are deliberately thin — they support coverage, not a second engine.
MAIN_COUNTS = {1: [4], 2: [4, 3], 3: [4, 3, 2]}
TECH_COUNTS = {1: [2], 2: [2, 2]}

# score_line weights. SETUP_PENALTY starts strong on purpose: the generic pilot's
# measured residual weakness is slow setups (M6 floor-test losses); revisit as its
# closing improves. PRIZE_PENALTY charges multi-prize liability (ex/mega bodies).
# Variable attacks print their best-case damage (conditions invisible to the
# features), so they count at a discount when picking a line's best attack.
COVERAGE_WEIGHT = 1.0
SETUP_PENALTY = 8.0
PRIZE_PENALTY = 10.0
VARIABLE_DISCOUNT = 0.5

# Basic-energy card ids are 1..8 == their energy_type_id (verified in tests).
_BASIC_ENERGY = {int(r["energy_type_id"]): int(r["card_id"])
                 for r in pl.read_parquet(DATA / "cards_features.parquet")
                            .filter(pl.col("is_basic_energy")).iter_rows(named=True)}


def _attacks_by_card() -> dict[int, list[dict]]:
    """card_id -> attack rows, via the cumulative-n_attacks contract (fact 2 above)."""
    cards = pl.read_parquet(DATA / "cards_features.parquet").sort("card_id")
    attacks = pl.read_parquet(DATA / "attacks_features.parquet")
    atk = {r["attackId"]: r for r in attacks.iter_rows(named=True)}
    out: dict[int, list[dict]] = {}
    next_id = 1
    for r in cards.iter_rows(named=True):
        n = r["n_attacks"]
        out[r["card_id"]] = [atk[i] for i in range(next_id, next_id + n)]
        next_id += n
    assert next_id - 1 == attacks.height, \
        f"attackId mapping broken: assigned {next_id - 1}, table has {attacks.height}"
    return out


ATTACKS = _attacks_by_card()

# Best printing per Pokémon name (highest HP) — the card a deck would actually play.
_BEST_BY_NAME: dict[str, int] = {}
for _cid, _r in _ft.items():
    if _r["is_pokemon"]:
        best = _BEST_BY_NAME.get(_r["name"])
        if best is None or _r["hp"] > _ft[best]["hp"]:
            _BEST_BY_NAME[_r["name"]] = _cid


@dataclasses.dataclass
class Line:
    """One playable evolution line, basic -> final (docs/M7-plan.md §2.2)."""
    stages: list[int]          # card_ids basic -> final
    final: int
    energy_type: int
    peak_damage: int           # damage of the final's best attack
    best_cost: int             # energy cost of that attack (NOT min_attack_cost)
    setup_cost: int            # best_cost + evolution steps (attach ~1/turn)
    hp: int
    prize_liability: int       # prizes_on_ko of the final (ex/mega = 2-3)

    @property
    def name(self) -> str:
        return _ft[self.final]["name"]


def _pre_evolution(card_id: int) -> int | None:
    """The best printing of this card's pre-evolution NAME, or None if broken."""
    evo_from = _ft[card_id]["evolves_from_id"]
    if evo_from == -1 or evo_from not in _ft:
        return None
    return _BEST_BY_NAME.get(_ft[evo_from]["name"])


def enumerate_lines() -> list[Line]:
    """Every complete evolution line in the pool whose final can actually attack."""
    lines = []
    for cid, r in _ft.items():
        if not r["is_pokemon"] or not ATTACKS[cid]:
            continue
        if r["is_basic"]:
            stages = [cid]
        elif r["is_stage1"]:
            basic = _pre_evolution(cid)
            if basic is None:
                continue
            stages = [basic, cid]
        elif r["is_stage2"]:
            mid = _pre_evolution(cid)
            basic = _pre_evolution(mid) if mid is not None else None
            if mid is None or basic is None:
                continue
            stages = [basic, mid, cid]
        else:
            continue
        def _effective(a: dict) -> float:
            return a["damage"] * (VARIABLE_DISCOUNT if a["is_variable"] else 1.0)
        best = max(ATTACKS[cid], key=_effective)
        if best["damage"] <= 0:
            continue
        lines.append(Line(
            stages=stages, final=cid, energy_type=int(r["energy_type_id"]),
            peak_damage=int(_effective(best)), best_cost=int(best["cost_total"]),
            setup_cost=int(best["cost_total"]) + len(stages) - 1,
            hp=int(r["hp"]), prize_liability=int(r["prizes_on_ko"]),
        ))
    return lines


def meta_weakness_histogram(meta: Path | None = None) -> dict[int, float]:
    """energy_type -> share of meta Pokémon weak to it.

    Default: the whole card pool (M6's measurement). With `meta` pointing at
    M7.0's opp_decks.parquet, the histogram is re-weighted by what opponents
    actually play — the ingestion -> deck-building feedback edge (§5.4).
    """
    weak: Counter = Counter()
    if meta is not None:
        for row in pl.read_parquet(meta).iter_rows(named=True):
            for cid in row["deck"]:
                if cid in _ft and _ft[cid]["is_pokemon"] and _ft[cid]["weakness_id"] > 0:
                    weak[_ft[cid]["weakness_id"]] += 1
    else:
        for r in _ft.values():
            if r["is_pokemon"] and r["weakness_id"] > 0:
                weak[r["weakness_id"]] += 1
    total = sum(weak.values())
    return {t: n / total for t, n in weak.items()} if total else {}


def score_line(line: Line, meta_types: dict[int, float]) -> float:
    """damage-per-energy x weakness-coverage, minus prize/setup liabilities (§2.2)."""
    dpe = line.peak_damage / max(line.best_cost, 1)
    coverage = meta_types.get(line.energy_type, 0.0)
    return (dpe * (1.0 + COVERAGE_WEIGHT * coverage)
            - PRIZE_PENALTY * (line.prize_liability - 1)
            - SETUP_PENALTY * line.setup_cost)


# ---------------------------------------------------------------------------
# Trainer shells (§2.3) — three sources, in deployment order
# ---------------------------------------------------------------------------
def _is_trainer(cid: int) -> bool:
    r = _ft[cid]
    return bool(r["is_item"] or r["is_supporter"] or r["is_stadium"] or r["is_tool"])


def mine_shells(out: Path | None = None) -> dict[str, list[int]]:
    """Source 1: trainer shells mined from the known-good decks/*.csv (proven,
    but only a few samples). Writes data/shells.json."""
    shells = {}
    for csv in sorted(DECK_DIR.glob("*.csv")):
        ids = [int(x) for x in csv.read_text().split() if x.strip()]
        shells[f"{csv.stem}_engine"] = [i for i in ids if _is_trainer(i)]
    path = out or SHELLS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shells, indent=2))
    print(f"{path}: {len(shells)} shells mined from {DECK_DIR}", flush=True)
    return shells


def harvested_shell(opp_decks: Path, size: int = 26) -> list[int]:
    """Source 3 (the real answer, post-M7.0): the most-played trainers across
    harvested opponent decks, copy-counts averaged and capped at 4."""
    per_deck: Counter = Counter()
    n_decks = 0
    seen: set[str] = set()
    for row in pl.read_parquet(opp_decks).iter_rows(named=True):
        if row["deck_hash"] in seen:      # count each distinct deck once
            continue
        seen.add(row["deck_hash"])
        n_decks += 1
        per_deck.update(cid for cid in row["deck"] if cid in _ft and _is_trainer(cid))
    shell: list[int] = []
    for cid, total in per_deck.most_common():
        copies = min(MAX_COPIES, max(1, round(total / n_decks)))
        shell.extend([cid] * min(copies, size - len(shell)))
        if len(shell) >= size:
            break
    return shell


def load_shells() -> dict[str, list[int]]:
    if not SHELLS_JSON.exists():
        return mine_shells()
    return json.loads(SHELLS_JSON.read_text())


# ---------------------------------------------------------------------------
# Deck assembly (§2.1) — legal by construction
# ---------------------------------------------------------------------------
def _energy_count(line: Line) -> int:
    """10-14 basics: more for expensive attackers (the pilot attaches ~1/turn)."""
    return 10 if line.best_cost <= 1 else 12 if line.best_cost == 2 else 14


DEFAULT_ENERGY_TYPE = 6  # Fighting — for colorless attackers (any basic pays {C})


def _main_energy_type(line: Line, tech: Line | None) -> int:
    """Colorless attackers (energy_type 0) take any basic energy: share the tech's
    type when there is one (a mono pile beats a split), else a fixed default."""
    if line.energy_type:
        return line.energy_type
    if tech is not None and tech.energy_type:
        return tech.energy_type
    return DEFAULT_ENERGY_TYPE


def build_deck(line: Line, shell: list[int], tech: Line | None = None) -> list[int]:
    """Assemble core + energy + tech + shell into exactly 60 legal cards.

    The shell is truncated to the room left (never padded with unknown cards —
    leftover room becomes extra basic energy, the one always-legal filler).
    """
    cards: list[int] = []
    for cid, n in zip(line.stages, MAIN_COUNTS[len(line.stages)]):
        cards.extend([cid] * n)

    etype = _main_energy_type(line, tech)
    main_energy = _energy_count(line)
    tech_energy = 0
    if tech is not None:
        for cid, n in zip(tech.stages, TECH_COUNTS[len(tech.stages)]):
            cards.extend([cid] * n)
        if tech.energy_type and tech.energy_type != etype:  # dual-type: tech gets 4 basics
            tech_energy = 4
    cards.extend([_BASIC_ENERGY[etype]] * main_energy)
    if tech_energy:
        cards.extend([_BASIC_ENERGY[tech.energy_type]] * tech_energy)

    room = DECK_SIZE - len(cards)
    cards.extend(shell[:room])
    cards.extend([_BASIC_ENERGY[etype]] * (DECK_SIZE - len(cards)))

    legal, reasons = validate_deck(cards)
    assert legal, f"template produced an illegal deck ({line.name}): {reasons}"
    return cards


# Card ids vetted TRULY harmless against the raw effect text (2026-07-11): the
# parquet's max_damage==0 only sees PRINTED damage, and the original floor picks
# (Iron Crown ex / Fezandipiti ex / Kyurem / Mega Skarmory ex) all carry "this
# attack does N damage to ..." effect attacks (100/110x3/220!) the engine fully
# implements — every historical floor number was measured against a deck that
# hits. Vetted by excluding any card whose Effect Explanation matches
# r'damage|poison|burn|paralyz|confus|knock|defending|opponent|discard' (case-
# insensitive) in pokemon-tcg-ai-battle/EN_Card_Data.csv (gitignored competition
# data, hence frozen here), HP-descending order. Re-derive with the snippet in
# docs/M7.md if the card pool ever changes.
_FLOOR_HARMLESS_IDS = (1009, 344, 548, 608, 653, 814, 875, 199, 160, 177, 183, 206)


def build_floor_deck(top: int = 4, copies: int = 4, out: Path | None = None) -> list[int]:
    """The M7.2b floor-test punching bag: a legal deck whose Pokémon can never
    deal damage — zero printed damage AND no damaging/removal attack effects
    (see _FLOOR_HARMLESS_IDS) — so it can't take a prize by KO. A competent
    pilot must beat it ~100%; M6's ad-hoc version measured 75% (self-decking)
    and was never committed — this one is reproducible.

    Deliberately shell-less: the gate measures OUR closing speed, not the
    punching bag's consistency. Committed at decks/floor_zero_damage.csv.
    """
    cards = pl.read_parquet(DATA / "cards_features.parquet")
    zero = (cards.filter(pl.col("is_pokemon") & pl.col("is_basic")
                         & (pl.col("max_damage") == 0) & ~pl.col("has_variable_attack")
                         & pl.col("card_id").is_in(list(_FLOOR_HARMLESS_IDS)))
                 .sort(["hp", "card_id"], descending=[True, False])
                 .head(top))
    deck: list[int] = []
    for cid in zero["card_id"]:
        deck.extend([int(cid)] * copies)
    etype = int(zero["energy_type_id"][0]) or DEFAULT_ENERGY_TYPE
    deck.extend([_BASIC_ENERGY.get(etype, _BASIC_ENERGY[DEFAULT_ENERGY_TYPE])]
                * (DECK_SIZE - len(deck)))    # basic energy: copy-cap exempt filler

    legal, reasons = validate_deck(deck)
    assert legal, f"floor deck is illegal: {reasons}"
    if out is not None:
        _write_deck_csv(out, deck)
    return deck


def pick_tech(lines: list[Line], main: Line, meta_types: dict[int, float]) -> Line | None:
    """Best weakness-coverage tech: a different-type, <=2-stage, low-liability line
    maximizing coverage x damage-per-energy against the meta (§2.1 item 4)."""
    best, best_score = None, 0.0
    for ln in lines:
        if (ln.energy_type == main.energy_type or len(ln.stages) > 2
                or ln.prize_liability > 2 or ln.final == main.final):
            continue
        s = meta_types.get(ln.energy_type, 0.0) * (ln.peak_damage / max(ln.best_cost, 1))
        if s > best_score:
            best, best_score = ln, s
    return best


# ---------------------------------------------------------------------------
# Template enumeration (§2.4): top-K lines x shells x {no-tech, coverage-tech}
# ---------------------------------------------------------------------------
def generate(n: int = 40, meta: Path | None = None, top_k: int = 10,
             out: Path | None = None) -> list[tuple[Path, list[int]]]:
    """Emit up to n deduped, legal candidate decks to decks/gen/, round-robin
    across the top-K lines so no archetype floods the batch."""
    meta_types = meta_weakness_histogram(meta)
    lines = sorted(enumerate_lines(), key=lambda ln: score_line(ln, meta_types),
                   reverse=True)
    mains = lines[:top_k]
    shells = load_shells()
    out_dir = out or GEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # candidates[i] = all builds for main line i (each main's variants stay together)
    candidates: list[list[tuple[str, list[int], dict]]] = []
    seen: set[str] = set()
    for rank, main in enumerate(mains):
        tech = pick_tech(lines, main, meta_types)
        variants = []
        for shell_name, shell in shells.items():
            for t in (None, tech):
                deck = build_deck(main, shell, t)
                h = deck_hash(deck)
                if h in seen:
                    continue
                seen.add(h)
                variants.append((h, deck, {
                    "hash": h, "line": main.name, "line_rank": rank,
                    "score": round(score_line(main, meta_types), 2),
                    "energy_type": main.energy_type, "shell": shell_name,
                    "tech": t.name if t else None,
                }))
        candidates.append(variants)

    # Round-robin across mains: archetype diversity survives any n cutoff.
    picked = []
    for i in range(max((len(v) for v in candidates), default=0)):
        for variants in candidates:
            if i < len(variants) and len(picked) < n:
                picked.append(variants[i])

    results, manifest = [], []
    for h, deck, info in picked:
        csv = out_dir / f"deck_{h[:12]}.csv"
        _write_deck_csv(csv, deck)
        info["csv"] = csv.name
        manifest.append(info)
        results.append((csv, deck))
    (out_dir / "manifest.json").write_text(json.dumps({
        "params": {"n": n, "top_k": top_k, "meta": str(meta) if meta else "pool"},
        "decks": manifest,
    }, indent=2))
    archetypes = len({m["line"] for m in manifest})
    print(f"{out_dir}: {len(results)} candidates across {archetypes} archetypes",
          flush=True)
    return results


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("lines", help="print the top-scored evolution lines")
    s.add_argument("--meta", type=Path, default=None)
    s.add_argument("--top", type=int, default=20)

    s = sub.add_parser("shells", help="mine data/shells.json from decks/*.csv")
    s.add_argument("--harvest", type=Path, default=None,
                   help="also add meta_staples_v1 from an opp_decks.parquet")

    sub.add_parser("floor", help="write the zero-damage floor-test deck")

    s = sub.add_parser("generate", help="emit candidate decks to decks/gen/")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--meta", type=Path, default=None)
    s.add_argument("--top-k", type=int, default=10)

    a = p.parse_args()
    if a.cmd == "lines":
        meta_types = meta_weakness_histogram(a.meta)
        ranked = sorted(enumerate_lines(), key=lambda ln: score_line(ln, meta_types),
                        reverse=True)
        print(f"{'line':<28} {'stg':>3} {'dmg':>4} {'cost':>4} {'hp':>4} "
              f"{'przs':>4} {'score':>7}")
        for ln in ranked[:a.top]:
            print(f"{ln.name:<28} {len(ln.stages):>3} {ln.peak_damage:>4} "
                  f"{ln.best_cost:>4} {ln.hp:>4} {ln.prize_liability:>4} "
                  f"{score_line(ln, meta_types):>7.1f}")
    elif a.cmd == "shells":
        shells = mine_shells()
        if a.harvest:
            shells["meta_staples_v1"] = harvested_shell(a.harvest)
            SHELLS_JSON.write_text(json.dumps(shells, indent=2))
            print(f"added meta_staples_v1 ({len(shells['meta_staples_v1'])} cards)",
                  flush=True)
    elif a.cmd == "floor":
        out = DECK_DIR / "floor_zero_damage.csv"
        build_floor_deck(out=out)
        print(f"wrote {out}", flush=True)
    elif a.cmd == "generate":
        generate(n=a.n, meta=a.meta, top_k=a.top_k)


if __name__ == "__main__":
    _main()
