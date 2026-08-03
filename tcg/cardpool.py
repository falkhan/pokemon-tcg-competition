"""Card pool for the deck lab: rows, the attack join, facets and filtering.

Streamlit-free and engine-free at import so it stays testable and cheap. The
Streamlit app is the only consumer that knows about widgets; everything a deck
builder needs to ASK of the card pool lives here.

Four data facts this module rests on, each of which has produced a wrong answer
somewhere in this project:

1. **`is_ex` and `is_mega_ex` are DISJOINT** (121 / 30, zero overlap), despite
   ``cg/api.py``'s docstring implying ex includes Mega. "Is it an ex?" is
   ``is_ex or is_mega_ex``.
2. **`attackId` maps to cards by cumulative `n_attacks`** in card_id order —
   ``attacks_features.parquet`` carries no card_id. The walk is copied from
   ``rl/deck_build.py::_attacks_by_card`` rather than imported, because that
   module drags ``rl.kaggle_ingest`` and runs a full best-printing pass at
   import. Its trailing assert comes along: it is the only thing standing
   between a re-exported parquet and silently wrong attack data.
3. **`max_damage` is BASE damage.** Scaling attackers record their base, so
   Alakazam reads 10 and Team Rocket's Spidops reads 0 — both headline real
   ladder decks. A naive damage floor hides them; hence `keep_variable`.
4. **Card names mix straight and curly apostrophes** (53 carry U+2019), and a
   cp1252 reader turns those into `â€™`. Normalise before matching on names.

Attack rules text and ability text are NOT in the parquets — they come from the
engine, imported lazily and degrading to empty so the pool works without it.
"""
import dataclasses
import re
from pathlib import Path

import polars as pl

from tcg.deck_search import CARD_ROWS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CARDS_PQ = DATA_DIR / "cards_features.parquet"
ATTACKS_PQ = DATA_DIR / "attacks_features.parquet"

CARD_TYPES = {0: "Pokemon", 1: "Item", 2: "Tool", 3: "Supporter", 4: "Stadium",
              5: "Basic Energy", 6: "Special Energy"}
ENERGY_TYPES = {0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
                5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon",
                10: "Rainbow"}
# Single letters for compact cost strings ("FFC"). N = draagoN, Y = rainbow:
# D is taken by Darkness and R by fiRe in the printed symbols.
ENERGY_ABBR = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D",
               8: "M", 9: "N", 10: "Y"}
COST_COLS = {f"cost_{ENERGY_TYPES[i]}": i for i in range(10)}

# Canonical TCG hues, lifted verbatim from notebooks/card_pool_eda.ipynb where
# they were validated CVD-separable. Two residual WARNs are carried there by
# secondary encoding (Fire<->Lightning CVD 7.9, Colorless contrast 2.45:1), so
# anywhere these are used the type NAME or its symbol must appear too — colour
# is never the only cue.
ENERGY_COLORS = {
    "Grass": "#3c8a2e", "Psychic": "#7c3aa6", "Water": "#2f86e0",
    "Fighting": "#b05f2a", "Darkness": "#5b4f9e", "Colorless": "#c29f45",
    "Fire": "#d64550", "Lightning": "#bd8500", "Metal": "#4577c2",
    "Dragon": "#9b7210", "Rainbow": "#8a8a8a",
}
ENERGY_EMOJI = {
    "Grass": "🌿", "Fire": "🔥", "Water": "💧", "Lightning": "⚡",
    "Psychic": "🔮", "Fighting": "👊", "Darkness": "🌑", "Metal": "⚙️",
    "Dragon": "🐉", "Colorless": "⭐", "Rainbow": "🌈",
}
TRAINER_EMOJI = {1: "🎒", 2: "🔧", 3: "🧑", 4: "🏟️", 5: "🔋", 6: "🔌"}
TIER_EMOJI = {"Mega ex": "🌟", "ex": "✨", "Tera": "💎", "regular": ""}
STAGE_EMOJI = {"Basic": "●", "Stage 1": "●●", "Stage 2": "●●●"}


def energy_emoji(type_name: str | None) -> str:
    return ENERGY_EMOJI.get(type_name or "", "")


def energy_color(type_name: str | None) -> str:
    return ENERGY_COLORS.get(type_name or "", "#8a8a8a")


def card_icon(row: dict) -> str:
    """One glyph identifying the card at a glance: energy symbol for anything
    typed, a trainer glyph otherwise."""
    if row["is_pokemon"] or row["is_basic_energy"]:
        return energy_emoji(row.get("type_name"))
    return TRAINER_EMOJI.get(row["card_type"], "🃏")

# "Marnie's Grimmsnarl ex" -> Marnie. Applied to POKEMON ONLY: over the whole
# pool it also catches "Boss's Orders" and "Hero's Cape", which are not themes.
OWNER_RE = re.compile(r"^(.+?)'s ")

RARE_CANDY_ID = 1079
STAGES = ("Basic", "Stage 1", "Stage 2")
TIERS = ("Mega ex", "ex", "Tera", "regular")

_CARDS: dict[int, dict] | None = None
_FRAME: pl.DataFrame | None = None
_ATTACKS: dict[int, list[dict]] | None = None
_ATK_TEXT: dict[int, dict] | None = None
_ABILITIES: dict[int, list[tuple[str, str]]] | None = None


def norm_name(name: str) -> str:
    """Card names arrive with straight and curly apostrophes, and cp1252 readers
    turn the curly form into `â€™`. Match on one normalised form."""
    return str(name).replace("â€™", "'").replace("’", "'")


def _kind(row: dict) -> str:
    if row["is_pokemon"]:
        return "Pokemon"
    if row["is_basic_energy"] or row["is_special_energy"]:
        return "Energy"
    return "Trainer"


def _stage(row: dict) -> str | None:
    if not row["is_pokemon"]:
        return None
    return "Stage 2" if row["is_stage2"] else ("Stage 1" if row["is_stage1"] else "Basic")


def tier_of(row: dict) -> str | None:
    """Mega ex / ex / Tera / regular. The first two are disjoint flags (fact 1)."""
    if not row["is_pokemon"]:
        return None
    if row["is_mega_ex"]:
        return "Mega ex"
    if row["is_ex"]:
        return "ex"
    if row["is_tera"]:
        return "Tera"
    return "regular"


def cards() -> dict[int, dict]:
    """card_id -> feature row plus derived name_norm / owner / tier / stage / kind.

    Deep-copies CARD_ROWS: that dict is the ship gate's own table
    (tcg.deck_search), and adding columns to it in place would mutate the
    validator's view of the pool.
    """
    global _CARDS
    if _CARDS is None:
        out = {}
        for cid, row in CARD_ROWS.items():
            r = dict(row)
            r["name_norm"] = norm_name(r["name"])
            r["kind"] = _kind(r)
            r["stage"] = _stage(r)
            r["tier"] = tier_of(r)
            m = OWNER_RE.match(r["name_norm"]) if r["is_pokemon"] else None
            r["owner"] = m.group(1) if m else None
            r["type_name"] = (ENERGY_TYPES.get(r["energy_type_id"])
                              if r["is_pokemon"] or r["is_basic_energy"] else None)
            out[cid] = r
        _CARDS = out
    return _CARDS


def attacks_by_card() -> dict[int, list[dict]]:
    """card_id -> attack rows, via the cumulative-n_attacks contract (fact 2)."""
    global _ATTACKS
    if _ATTACKS is None:
        cards_df = pl.read_parquet(CARDS_PQ).sort("card_id")
        attacks = pl.read_parquet(ATTACKS_PQ)
        atk = {r["attackId"]: r for r in attacks.iter_rows(named=True)}
        out: dict[int, list[dict]] = {}
        next_id = 1
        for r in cards_df.iter_rows(named=True):
            n = r["n_attacks"]
            out[r["card_id"]] = [atk[i] for i in range(next_id, next_id + n)]
            next_id += n
        assert next_id - 1 == attacks.height, \
            f"attackId mapping broken: assigned {next_id - 1}, table has {attacks.height}"
        _ATTACKS = out
    return _ATTACKS


def energy_cost_string(attack_row: dict) -> str:
    """{'cost_Fighting': 2, 'cost_Colorless': 1} -> 'FFC'. Colorless last."""
    parts = []
    for col, tid in COST_COLS.items():
        if tid == 0:
            continue
        parts.append(ENERGY_ABBR[tid] * int(attack_row.get(col) or 0))
    parts.append(ENERGY_ABBR[0] * int(attack_row.get("cost_Colorless") or 0))
    return "".join(parts)


def cards_frame() -> pl.DataFrame:
    """The filterable frame: card rows + a per-card attack aggregate.

    The aggregate is folded in ONCE so every facet stays a single polars
    expression instead of a Python loop over 1267 cards per widget tick.
    """
    global _FRAME
    if _FRAME is None:
        atk = attacks_by_card()
        rows = []
        for cid, r in cards().items():
            a = atk.get(cid, [])
            costs = [int(x["cost_total"] or 0) for x in a]
            rows.append({
                "card_id": cid,
                "icon": card_icon(r),
                "name": r["name_norm"],
                "badge": TIER_EMOJI.get(r["tier"] or "", "")
                         + ("🅰" if r["is_ace_spec"] else ""),
                "card_type": r["card_type"],
                "type_label": CARD_TYPES.get(r["card_type"], "?"),
                "kind": r["kind"],
                "stage": r["stage"],
                "tier": r["tier"],
                "owner": r["owner"],
                "energy_type_id": r["energy_type_id"],
                "type_name": r["type_name"],
                "hp": r["hp"],
                "retreat_cost": r["retreat_cost"],
                "weakness": ENERGY_TYPES.get(r["weakness_id"]),
                "resistance": ENERGY_TYPES.get(r["resistance_id"]),
                "is_pokemon": r["is_pokemon"],
                "is_ace_spec": r["is_ace_spec"],
                "is_basic_energy": r["is_basic_energy"],
                "prizes_on_ko": r["prizes_on_ko"],
                "n_attacks": r["n_attacks"],
                "atk_min_cost": min(costs) if costs else None,
                "atk_max_cost": max(costs) if costs else None,
                "atk_max_damage": max((int(x["damage"] or 0) for x in a), default=0),
                "atk_any_variable": any(bool(x["is_variable"]) for x in a),
            })
        _FRAME = pl.DataFrame(rows).sort("card_id")
    return _FRAME


def owners() -> list[str]:
    """Pokemon owner themes, most cards first. Trainer possessives excluded."""
    counts: dict[str, int] = {}
    for r in cards().values():
        if r["owner"]:
            counts[r["owner"]] = counts.get(r["owner"], 0) + 1
    return sorted(counts, key=lambda o: (-counts[o], o))


def pre_evo_names() -> dict[str, str]:
    """normalised name -> normalised pre-evolution name.

    By NAME, never by id: `evolves_from_id` names one printing, and real decks
    legally play any same-named card (decks/lucario.csv runs Riolu #677 while
    Mega Lucario ex points at #974). Verified total and single-valued.
    """
    ft = cards()
    out: dict[str, str] = {}
    for r in ft.values():
        pre = r["evolves_from_id"]
        if r["is_pokemon"] and pre is not None and pre != -1 and pre in ft:
            out[r["name_norm"]] = ft[pre]["name_norm"]
    return out


def printings_by_name() -> dict[str, list[int]]:
    """name -> every card id printing it. 154 names have more than one, and the
    4-copy limit is per NAME — 3x Riolu #677 + 2x Riolu #974 is illegal."""
    out: dict[str, list[int]] = {}
    for cid, r in cards().items():
        out.setdefault(r["name_norm"], []).append(cid)
    return out


def card_label(cid: int) -> str:
    r = cards().get(cid)
    if r is None:
        return f"#{cid} (unknown)"
    bits = [r["name_norm"], f"#{cid}"]
    if r["is_pokemon"]:
        bits.append(f"{r['type_name']} {r['hp']}HP")
    else:
        bits.append(CARD_TYPES.get(r["card_type"], "?"))
    return "  ".join(bits)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CardFilter:
    text: str = ""
    card_types: tuple[int, ...] = ()
    energy_types: tuple[int, ...] = ()
    owner: str | None = None
    hp: tuple[int, int] = (0, 400)
    stages: tuple[str, ...] = ()
    tiers: tuple[str, ...] = ()
    ace_spec_only: bool = False
    retreat: tuple[int, int] = (0, 4)
    attack_cost: tuple[int, int] = (0, 5)
    damage: tuple[int, int] = (0, 350)
    keep_variable: bool = True


def filter_cards(f: CardFilter) -> pl.DataFrame:
    """Apply a CardFilter. Cheap (~3 ms) — do not cache it."""
    df = cards_frame()
    if f.text:
        needle = norm_name(f.text).lower()
        df = df.filter(pl.col("name").str.to_lowercase().str.contains(needle, literal=True))
    if f.card_types:
        df = df.filter(pl.col("card_type").is_in(list(f.card_types)))
    if f.energy_types:
        # Trainers all carry energy_type_id 0, so an unguarded Colorless filter
        # returns 303 rows of which 199 are Trainers. Restrict to typed cards.
        df = df.filter(pl.col("type_name").is_not_null()
                       & pl.col("energy_type_id").is_in(list(f.energy_types)))
    if f.owner:
        df = df.filter(pl.col("owner") == f.owner)
    if f.stages:
        df = df.filter(pl.col("stage").is_in(list(f.stages)))
    if f.tiers:
        df = df.filter(pl.col("tier").is_in(list(f.tiers)))
    if f.ace_spec_only:
        df = df.filter(pl.col("is_ace_spec"))

    lo, hi = f.hp
    if (lo, hi) != (0, 400):
        df = df.filter(~pl.col("is_pokemon")
                       | pl.col("hp").is_between(lo, hi))
    lo, hi = f.retreat
    if (lo, hi) != (0, 4):
        df = df.filter(~pl.col("is_pokemon")
                       | pl.col("retreat_cost").is_between(lo, hi))

    # Attack facets apply only to cards that HAVE attacks; a Supporter is not
    # excluded for failing a damage floor it can never meet.
    lo, hi = f.attack_cost
    if (lo, hi) != (0, 5):
        df = df.filter((pl.col("n_attacks") == 0)
                       | pl.col("atk_min_cost").is_between(lo, hi))
    lo, hi = f.damage
    if (lo, hi) != (0, 350):
        keep = pl.col("atk_max_damage").is_between(lo, hi)
        if f.keep_variable:
            # Scaling attackers record BASE damage (fact 3): Alakazam 10,
            # Team Rocket's Spidops 0. Dropping them hides two real ladder decks.
            keep = keep | pl.col("atk_any_variable")
        df = df.filter((pl.col("n_attacks") == 0) | keep)
    return df


# ---------------------------------------------------------------------------
# Engine text (optional — degrades to empty when cg is absent or minimal)
# ---------------------------------------------------------------------------
def attack_texts() -> dict[int, dict]:
    """attackId -> {name, text}. Empty when the engine is unavailable.

    The getattr defaults are load-bearing, not padding: tests/fake_cg.py builds
    attacks with no `name` and no `text`, and conftest installs it always.
    """
    global _ATK_TEXT
    if _ATK_TEXT is None:
        try:
            from cg.api import all_attack
            _ATK_TEXT = {a.attackId: {"name": getattr(a, "name", "") or "",
                                      "text": getattr(a, "text", "") or ""}
                         for a in all_attack()}
        except Exception:  # noqa: BLE001 — the pool must work without the engine
            _ATK_TEXT = {}
    return _ATK_TEXT


def card_abilities() -> dict[int, list[tuple[str, str]]]:
    """card_id -> [(ability name, rules text)]. Empty without the engine."""
    global _ABILITIES
    if _ABILITIES is None:
        try:
            from cg.api import all_card_data
            out = {}
            for c in all_card_data():
                skills = getattr(c, "skills", None) or ()
                if skills:
                    out[c.cardId] = [(getattr(s, "name", "") or "",
                                      getattr(s, "text", "") or "") for s in skills]
            _ABILITIES = out
        except Exception:  # noqa: BLE001
            _ABILITIES = {}
    return _ABILITIES


def engine_available() -> bool:
    return bool(attack_texts())


# Evolved Pokemon that can reach the board WITHOUT their pre-evolution, via an
# ability that puts them into play from hand at setup. Discovered from the data:
# every one of 237 harvested `archaludon` seats runs 4x Cinderace (Stage 2, from
# Raboot) with no Raboot and no Rare Candy, because Cinderace's "Explosiveness"
# reads "If this Pokemon is in your hand when you are setting up to play, you
# may put it face down in the Active Spot." Without this exemption the deck lab
# reports a hard error on the most-played build of a whole archetype.
#
# Detected from ability text when the engine is present; the constant is the
# fallback so behaviour is identical under tests/fake_cg.py (which has no skills).
_SELF_START_RE = re.compile(r"put it face down in the active spot", re.I)
SELF_STARTING_IDS = frozenset({666})     # Cinderace
_SELF_START: frozenset[int] | None = None


def self_starting_ids() -> frozenset[int]:
    """Evolved Pokemon that can start from hand, so they need no pre-evolution."""
    global _SELF_START
    if _SELF_START is None:
        found = {cid for cid, skills in card_abilities().items()
                 if any(_SELF_START_RE.search(text or "") for _n, text in skills)}
        _SELF_START = frozenset(found | SELF_STARTING_IDS)
    return _SELF_START
