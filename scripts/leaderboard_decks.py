"""Leaderboard deck census: who is at the top of the ladder, and what are they playing.

We weight gate beds against `data/m39_live_mix.json` — the mix our agent has
actually PLAYED. That is not the same population as the top of the ladder, and
nobody had measured the difference. This builds the top-of-ladder distribution:
one row per leaderboard team, each labelled with the deck it is piloting.

Channel (three joins, only the first needs the network):

    leaderboard(top)            -> team_id, team_name, score   [rl.kaggle_ingest]
    episodes.parquet            -> team_id  -> submission_id (latest by end_time)
    opp_decks.parquet           -> submission_id -> the 60-card decklist
    cards_features.parquet      -> decklist -> energy / theme / main attacker

COVERAGE LIMIT — read before trusting a low number. Kaggle retired the
ListEpisodes teamId filter (probed 2026-07-15: only submissionId and ids[] are
accepted, teamId 400s). A team we have never played is therefore unreachable by
any targeted call; only the snowball harvest (`rl.kaggle_ingest refresh
--opp-subs`) can find them. This module is offline by design past the
leaderboard call: it REPORTS gaps as `never_seen` / `no_deck` rather than
back-filling them, so the coverage number stays honest.

Deck identity comes from the replay's deck step (already harvested into
opp_decks.parquet), and the main attacker is corroborated against what the
pilot actually attacked with in the cached replays -- a decklist alone cannot
tell a 1-of tech from the plan (see `describe_deck`).

Usage:
    uv run python scripts/leaderboard_decks.py --top 250
    uv run python scripts/leaderboard_decks.py --top 250 --no-evidence   # skip replays
    uv run python scripts/leaderboard_decks.py --offline                 # reuse cached leaderboard
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402

from m39_live_mix import _norm, classify  # noqa: E402
from rl.kaggle_ingest import (  # noqa: E402
    EPISODES_PQ, OPP_DECKS_PQ, RAW_DIR, _archetype_name, console_safe, leaderboard,
)

CARDS_PQ = ROOT / "data/cards_features.parquet"
CENSUS_PQ = ROOT / "data/kaggle/leaderboard_decks.parquet"
LEADERBOARD_PQ = ROOT / "data/kaggle/leaderboard.parquet"

# energy_type_id -> name. Basic energy card ids 1..8 map onto 1..8 exactly;
# 0 is Colorless (no basic energy prints) and 9 is Dragon.
ENERGY_TYPES = {
    0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
    5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon",
}

# "Marnie's Grimmsnarl ex" -> Marnie. The owner theme is printed into the card
# NAME in this pool, so it survives without EN_Card_Data.csv (which is not on
# disk). Observed owners: Arven, Cynthia, Erika, Ethan, Hop, Iono, Larry,
# Lillie, Marnie, Misty, N, Steven, Team Rocket.
OWNER_RE = re.compile(r"^(.+?)'s ")

# A theme only wins the fallback vote if it is actually the deck's identity and
# not a splashed engine card -- 4 copies is a full playset of one line.
THEME_MIN_COPIES = 4

# Scaling attackers ("30x the number of Energy...") record their BASE damage, so
# cards_features gives Alakazam 10 and Team Rocket's Spidops 0 -- both of which
# are the plan of the deck they headline. Score them as a mid-range attacker
# instead of as zero: enough to beat the 10-30 damage basics of their own
# evolution line, not enough to outrank a genuine printed heavy hitter.
VARIABLE_DAMAGE_PROXY = 100

_CARDS: dict[int, dict] | None = None


def cards() -> dict[int, dict]:
    """card_id -> feature row, names normalised. Cached across calls."""
    global _CARDS
    if _CARDS is None:
        rows = pl.read_parquet(CARDS_PQ).iter_rows(named=True)
        _CARDS = {}
        for r in rows:
            r["name"] = _norm(r["name"])
            _CARDS[r["card_id"]] = r
    return _CARDS


# ---------------------------------------------------------------------------
# Deck -> description (energy / theme / main attacker)
# ---------------------------------------------------------------------------
def _stage(row: dict) -> int:
    return 2 if row["is_stage2"] else (1 if row["is_stage1"] else 0)


def _damage(row: dict) -> int:
    """Printed damage, with scaling attackers floored at VARIABLE_DAMAGE_PROXY."""
    dmg = row["max_damage"] or 0
    return max(dmg, VARIABLE_DAMAGE_PROXY) if row["has_variable_attack"] else dmg


def _tier(row: dict) -> str:
    if row["is_mega_ex"]:
        return "Mega ex"          # mutually exclusive with is_ex in this pool
    if row["is_ex"]:
        return "ex"
    if row["is_tera"]:
        return "Tera"
    return "regular"


def _owner(name: str) -> str | None:
    m = OWNER_RE.match(name)
    return m.group(1) if m else None


def describe_deck(ids: list[int]) -> dict:
    """Identify a 60-card decklist from its cards.

    Returns energy identity, owner theme, main attacker + tier, the canonical
    family label, and the top-2-Pokemon slug.

    Choosing the main attacker is the delicate part. The key is
    (on-energy, max_damage, stage, copies, hp); every term was forced by a real
    mislabelling:

    * a damage floor -- ranking on printed damage alone crowns *Fezandipiti ex*
      (a genuinely 0-damage support ex) in 52 harvested decks, so zero-damage
      Pokemon are not eligible at all. Scaling attackers are the exception and
      get VARIABLE_DAMAGE_PROXY: *Alakazam* prints 10 and *Team Rocket's Spidops*
      prints 0, and both are the plan of the deck they headline.
    * energy consistency FIRST -- the Team Rocket Grass decks (Spidops /
      Tarountula, 445 games) run 7-8 Grass energy and a single Psychic
      *Team Rocket's Mewtwo ex* tech. Mewtwo has the highest printed damage in
      the list, so a damage-led key names the tech instead of the plan. Matching
      the deck's own basic-energy type fixes it.
    * damage ABOVE copies -- copies-first looks right (the plan is played in
      multiples, techs are 1-ofs) but inverts every evolution line: the basic is
      a 4-of while the Stage-2 ex is a 2-3-of, so it returns "Marnie's Impidimp"
      for Grimmsnarl and "Cynthia's Gabite" for Garchomp.

    That leaves one gap a decklist cannot close: a 1-of high-damage tech of the
    deck's OWN energy type still outranks the plan. `attacker_evidence` is the
    check for it -- the census flags every disagreement rather than papering over it.
    """
    ft = cards()
    known = [i for i in ids if i in ft]
    copies = Counter(known)

    basic_energy = Counter()
    n_special_energy = 0
    for cid, n in copies.items():
        row = ft[cid]
        if row["is_basic_energy"]:
            basic_energy[ENERGY_TYPES.get(row["energy_type_id"], "?")] += n
        elif row["is_special_energy"]:
            n_special_energy += n
    # None is a real answer: some Mega lists run 0 basic energy, all special.
    primary_energy = basic_energy.most_common(1)[0][0] if basic_energy else None

    mons = [cid for cid in copies if ft[cid]["is_pokemon"]]
    attackers = [cid for cid in mons if _damage(ft[cid]) > 0]

    attacker = attacker_tier = attacker_energy = None
    if attackers:
        def rank(cid: int):
            row = ft[cid]
            on_type = (primary_energy is not None
                       and ENERGY_TYPES.get(row["energy_type_id"]) == primary_energy)
            return (on_type, _damage(row), _stage(row), copies[cid], row["hp"] or 0)

        best = max(attackers, key=rank)
        attacker = ft[best]["name"]
        attacker_tier = _tier(ft[best])
        attacker_energy = ENERGY_TYPES.get(ft[best]["energy_type_id"])

    # Theme follows the attacker. The copy-weighted plurality is only a fallback
    # -- used alone it labels Blaziken ex decks "Lillie" off a splashed engine card.
    theme = _owner(attacker) if attacker else None
    if theme is None:
        votes = Counter()
        for cid in mons:
            owner = _owner(ft[cid]["name"])
            if owner:
                votes[owner] += copies[cid]
        if votes and votes.most_common(1)[0][1] >= THEME_MIN_COPIES:
            theme = votes.most_common(1)[0][0]

    ex_lines = sorted(
        ((ft[cid]["name"], copies[cid]) for cid in mons
         if ft[cid]["is_ex"] or ft[cid]["is_mega_ex"]),
        key=lambda t: (-t[1], t[0]),
    )
    pokemon_names = {ft[cid]["name"] for cid in mons}

    return {
        "label": attacker or "unknown",
        "family": classify(pokemon_names),
        "archetype_slug": _archetype_name(known),
        "theme": theme,
        "primary_energy": primary_energy,
        "energy_types": ", ".join(f"{k} x{v}" for k, v in basic_energy.most_common()),
        "n_special_energy": n_special_energy,
        "attacker": attacker,
        "attacker_tier": attacker_tier,
        "attacker_energy": attacker_energy,
        "ex_lines": ", ".join(f"{n} x{c}" for n, c in ex_lines),
    }


# ---------------------------------------------------------------------------
# Offline joins: team -> submission -> deck
# ---------------------------------------------------------------------------
def team_submissions() -> pl.DataFrame:
    """(team_id, submission_id, end_time, score) — both seats of every episode.

    A team's CURRENT agent is its most recent submission by end_time; older rows
    are kept so a team whose newest submission has no cached replay can still
    fall back to one that does.
    """
    ep = pl.read_parquet(EPISODES_PQ)
    seats = [
        ep.select(
            pl.col(f"team_id_{s}").alias("team_id"),
            pl.col(f"submission_id_{s}").alias("submission_id"),
            pl.col(f"updated_score_{s}").alias("seat_score"),
            pl.col("end_time"),
            pl.col("episode_id"),
        )
        for s in (0, 1)
    ]
    return pl.concat(seats).drop_nulls(["team_id", "submission_id"])


def decks_by_submission() -> dict[int, dict]:
    """submission_id -> {deck, deck_hash} from the most recent harvested episode."""
    od = pl.read_parquet(OPP_DECKS_PQ).drop_nulls("submission_id").sort("episode_id")
    out = {}
    for r in od.iter_rows(named=True):
        out[r["submission_id"]] = {"deck": list(r["deck"]), "deck_hash": r["deck_hash"]}
    return out


# ---------------------------------------------------------------------------
# Evidence layer: what did the pilot actually attack with?
# ---------------------------------------------------------------------------
def preevolutions(ids: list[int]) -> set[int]:
    """Card ids in this deck that something else in the SAME deck evolves from.

    Attack tallies otherwise over-count these: a pilot chips with Impidimp or
    Dwebble on turn 1-2 and evolves later, so raw attack events name the basic
    as the deck's attacker. Only lines actually present count — a Kangaskhan
    that nothing evolves from stays eligible.
    """
    ft = cards()
    present = set(ids)
    return {ft[i]["evolves_from_id"] for i in present
            if i in ft and ft[i]["evolves_from_id"] in present}


def attacker_evidence(episodes_by_seat: dict, decks: dict | None = None,
                      cap: int = 6) -> dict:
    """(submission_id) -> Counter of Pokemon that actually ATTACKED, from cached replays.

    A decklist says what a pilot COULD do; this says what they did. It is the
    tiebreak when the heuristic attacker is ambiguous (a high-damage 1-of tech
    versus the line the deck is actually built around), and it is what caught
    the wall lists whose Crustle never reaches the board while Mega Kangaskhan
    ex does all the attacking.

    Pass `decks` (submission_id -> decklist) to discount pre-evolution chip
    attacks; without it every attack counts equally.

    Offline: reads only replays already in data/kaggle/raw/. `cap` bounds the
    replays walked per submission so a 250-team census stays quick.
    """
    from cg.api import OptionType, SelectContext, to_observation_class
    from rl.replay_bc import iter_replay_decisions

    ft = cards()
    skip_by_sub = ({sub: preevolutions(d["deck"]) for sub, d in decks.items()}
                   if decks else {})
    out = defaultdict(Counter)
    for sub, seats in episodes_by_seat.items():
        for episode_id, seat in seats[:cap]:
            path = RAW_DIR / f"episode_{episode_id}.json.gz"
            if not path.exists():
                continue
            try:
                replay = json.loads(gzip.decompress(path.read_bytes()))
                steps = replay["steps"]
            except Exception:  # noqa: BLE001 — a corrupt cache entry is not fatal here
                continue
            drops = Counter()
            for _i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
                try:
                    obs = to_observation_class(obs_dict)
                    state, sel = obs.current, obs.select
                    if state is None or sel is None or sel.context != SelectContext.MAIN:
                        continue
                    chosen = sel.option[action[0]]
                    if OptionType(chosen.type) != OptionType.ATTACK:
                        continue
                    me = state.players[state.yourIndex]
                    active = [a for a in (me.active or []) if a]
                    if active and active[0].id not in skip_by_sub.get(sub, ()):
                        out[sub][ft.get(active[0].id, {}).get("name", active[0].id)] += 1
                except Exception:  # noqa: BLE001 — schema drift on one prompt
                    continue
    return {k: v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
CENSUS_SCHEMA = {
    "rank": pl.Int64, "team_id": pl.Int64, "team_name": pl.Utf8, "score": pl.Float64,
    "submission_id": pl.Int64, "last_seen": pl.Utf8, "n_episodes": pl.Int64,
    "deck_hash": pl.Utf8, "coverage": pl.Utf8,
    "label": pl.Utf8, "family": pl.Utf8, "archetype_slug": pl.Utf8, "theme": pl.Utf8,
    "primary_energy": pl.Utf8, "energy_types": pl.Utf8, "n_special_energy": pl.Int64,
    "attacker": pl.Utf8, "attacker_tier": pl.Utf8, "attacker_energy": pl.Utf8,
    "ex_lines": pl.Utf8, "played_attacker": pl.Utf8, "evidence_agrees": pl.Boolean,
}

_UNKNOWN = {
    "label": None, "family": None, "archetype_slug": None, "theme": None,
    "primary_energy": None, "energy_types": None, "n_special_energy": None,
    "attacker": None, "attacker_tier": None, "attacker_energy": None, "ex_lines": None,
}


def build_census(top: int = 250, evidence: bool = True, offline: bool = False,
                 evidence_cap: int = 6) -> pl.DataFrame:
    """One row per leaderboard team, labelled with the deck it is piloting."""
    if offline and LEADERBOARD_PQ.exists():
        lb = pl.read_parquet(LEADERBOARD_PQ).head(top).to_dicts()
        print(f"leaderboard: {len(lb)} teams (cached {LEADERBOARD_PQ.name})", flush=True)
    else:
        lb = leaderboard(top=top, quiet=True)
        LEADERBOARD_PQ.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(lb).write_parquet(LEADERBOARD_PQ)
        print(f"leaderboard: {len(lb)} teams fetched", flush=True)

    seats = team_submissions()
    decks = decks_by_submission()

    # Per team: submissions newest-first, plus episode count and last-seen.
    by_team: dict[int, list] = defaultdict(list)
    for r in seats.sort("end_time", descending=True).iter_rows(named=True):
        by_team[r["team_id"]].append(r)

    # (episode_id, seat) per submission, newest first, for the evidence walk.
    ep = pl.read_parquet(EPISODES_PQ)
    ep_by_sub: dict[int, list] = defaultdict(list)
    for s in (0, 1):
        for episode_id, sub in ep.select("episode_id",
                                         f"submission_id_{s}").drop_nulls().iter_rows():
            ep_by_sub[sub].append((episode_id, s))
    for v in ep_by_sub.values():
        v.sort(reverse=True)

    rows = []
    for rank, entry in enumerate(lb, start=1):
        row = {
            "rank": rank, "team_id": entry["team_id"], "team_name": entry["team_name"],
            "score": entry["score"], "submission_id": None, "last_seen": None,
            "n_episodes": 0, "deck_hash": None, "coverage": "never_seen",
            "played_attacker": None, "evidence_agrees": None, **_UNKNOWN,
        }
        history = by_team.get(entry["team_id"])
        if history:
            newest = history[0]
            row["last_seen"] = newest["end_time"]
            row["n_episodes"] = len({h["episode_id"] for h in history})
            # Their current submission; fall back to the newest one we hold a deck for.
            pick, coverage = newest["submission_id"], "no_deck"
            if pick in decks:
                coverage = "current"
            else:
                older = next((h["submission_id"] for h in history
                              if h["submission_id"] in decks), None)
                if older is not None:
                    pick, coverage = older, "older_submission"
            row["submission_id"], row["coverage"] = pick, coverage
            if coverage != "no_deck":
                deck = decks[pick]
                row["deck_hash"] = deck["deck_hash"]
                row.update(describe_deck(deck["deck"]))
        rows.append(row)

    if evidence:
        wanted = {r["submission_id"]: ep_by_sub.get(r["submission_id"], [])
                  for r in rows if r["submission_id"] and r["deck_hash"]}
        played = attacker_evidence(wanted, decks={s: decks[s] for s in wanted
                                                  if s in decks}, cap=evidence_cap)
        for r in rows:
            counts = played.get(r["submission_id"])
            if counts:
                top_played = counts.most_common(1)[0][0]
                r["played_attacker"] = str(top_played)
                r["evidence_agrees"] = (top_played == r["attacker"])

    df = pl.DataFrame(rows, schema=CENSUS_SCHEMA)
    CENSUS_PQ.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CENSUS_PQ)
    return df


def _report(df: pl.DataFrame) -> None:
    n = df.height
    cov = Counter(df["coverage"].to_list())
    labeled = n - cov["never_seen"] - cov["no_deck"]
    print(f"\ncensus: {n} teams, {labeled} labeled ({labeled / max(n, 1):.0%})", flush=True)
    for k, v in cov.most_common():
        print(f"  {k:18s} {v:4d}", flush=True)

    known = df.filter(pl.col("label").is_not_null())
    print(f"\ntop decks across the top {n}:", flush=True)
    dist = (known.group_by("label")
                 .agg(pl.len().alias("n"), pl.mean("score").alias("mean_score"),
                      pl.first("primary_energy"), pl.first("attacker_tier"))
                 .sort("n", descending=True).head(15))
    for r in dist.iter_rows(named=True):
        print(f"  {r['n']:4d} ({r['n'] / max(labeled, 1):5.1%})  "
              f"{console_safe(r['label'])[:32]:32s} {str(r['primary_energy']):10s} "
              f"{str(r['attacker_tier']):8s} mean={r['mean_score']:7.1f}", flush=True)

    print("\nby family:", flush=True)
    for r in (known.group_by("family").agg(pl.len().alias("n"))
                   .sort("n", descending=True).iter_rows(named=True)):
        print(f"  {r['family']:14s} {r['n']:4d} ({r['n'] / max(labeled, 1):5.1%})", flush=True)

    ev = df.filter(pl.col("evidence_agrees").is_not_null())
    if ev.height:
        agree = int(ev["evidence_agrees"].sum())
        print(f"\nreplay evidence: {agree}/{ev.height} attackers confirmed "
              f"({agree / ev.height:.0%})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=250, help="leaderboard depth")
    ap.add_argument("--offline", action="store_true",
                    help="reuse the cached leaderboard instead of fetching")
    ap.add_argument("--no-evidence", action="store_true",
                    help="skip the cached-replay attacker check")
    ap.add_argument("--evidence-cap", type=int, default=6,
                    help="replays walked per submission")
    a = ap.parse_args()
    df = build_census(top=a.top, evidence=not a.no_evidence, offline=a.offline,
                      evidence_cap=a.evidence_cap)
    _report(df)
    print(f"\nwrote {CENSUS_PQ.relative_to(ROOT)} ({df.height} rows)", flush=True)


if __name__ == "__main__":
    main()
