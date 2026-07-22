"""Kaggle episode-replay ingestion — harvest the real meta from the leaderboard (M7.0).

Why: the #1 measured bottleneck (docs/DECISIONS.md 2026-07-08) is opponent-field
diversity — every offline search overfits to 2 rule pilots + bc_v1, while the real
field is ~4,500 teams. Kaggle serializes every episode our submissions play, and the
env contract (main.py: the first action each agent returns IS its 60-card deck) means
each replay carries both decklists. This module downloads, caches, and parses those
replays into meta tables that feed deck building, the league anchor field, and
archetype-inferred determinization (docs/M7-plan.md §5).

Design: the network layer (`fetch_episode` / `list_episodes`) is a thin, injectable
seam over the Kaggle EpisodeService endpoints; everything downstream
(`parse_episode`, `harvest_decks`, `build_meta_field`, `forensics`) is pure and runs
offline against the immutable gzip cache in data/kaggle/raw/. Two replay payload
shapes are accepted: a kaggle_environments ``env.toJSON()`` dict (keys: steps,
rewards, statuses, configuration, info) and the ``GetEpisodeReplay`` wrapper
``{"replay": "<json-string>"}``. The parser is schema-TOLERANT (warnings, not
crashes — M7.0 exists to discover the schema); the ``verify`` command applies the
HARD asserts (60-int decks passing validate_deck) and prints the go/no-go report.

Observed schema: TBD — filled in by the M7.0 [NET] spike (runbook in docs/M7.md).
Until then the shapes above are assumptions tracked in docs/M7.md.

Usage:
  python -m rl.kaggle_ingest list --sub 54474043
  python -m rl.kaggle_ingest fetch --episode <id>
  python -m rl.kaggle_ingest import-file episode-<id>-replay.json --episode <id>
  python -m rl.kaggle_ingest verify --episode <id>
  python -m rl.kaggle_ingest refresh --subs 54474043 --max-new 100
  python -m rl.kaggle_ingest harvest --min-games 3
  python -m rl.kaggle_ingest meta --top-k 8 --dedupe-by archetype
  python -m rl.kaggle_ingest forensics --sub 54474043
"""
import argparse
import dataclasses
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from rl.deck_search import DECK_SIZE, _ft, validate_deck

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KAGGLE_DIR = DATA / "kaggle"
RAW_DIR = KAGGLE_DIR / "raw"
EPISODES_PQ = KAGGLE_DIR / "episodes.parquet"
OPP_DECKS_PQ = KAGGLE_DIR / "opp_decks.parquet"

LOGS_DIR = KAGGLE_DIR / "logs"

COMPETITION = "pokemon-tcg-ai-battle"
BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService"
THROTTLE_S = 1.0     # min seconds between network requests (politeness, §5.1)
ARCHETYPE_LABELER = "top2_pokemon_v2"  # per-deck card-identity naming (M25 labeler fix)

# Observation keys that carry no game state; an observation with only these is "blank".
_TRIVIAL_OBS_KEYS = {"remainingOverageTime", "step", "player"}

# Opponent spec tuple; canonical home is rl/matchrunner.py (M7.2), whose
# resolve_deck accepts the csv PATHs build_meta_field puts in the deck slot.
OpponentSpec = tuple


class SchemaError(ValueError):
    """A replay payload is structurally unusable (vs. merely missing fields)."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _get(d: dict, *names, default=None):
    """First present key among camelCase/snake_case spellings."""
    for n in names:
        if n in d:
            return d[n]
    return default


def deck_hash(ids: list[int]) -> str:
    """Order-invariant sha1 of a decklist — the (deck) identity used across M7.

    rl/league.py (M7.2) imports this; do not change the definition without
    re-hashing data/kaggle/*.parquet and data/league/.
    """
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()


def _write_deck_csv(path: Path, ids: list[int]) -> None:
    """decks/*.csv format: one card id per line, no header."""
    path.write_text("\n".join(str(i) for i in ids) + "\n")


# ---------------------------------------------------------------------------
# Card features (engine-free)
# ---------------------------------------------------------------------------
# Deliberately re-derived from cards_features.parquet instead of importing
# rl.encoders.FEAT: encoders imports rl.combat -> cg, and ingestion must run
# engine-free (docs/M7-plan.md §8). Extracting a shared cg-free rl/features.py
# is deferred to the M7.3 encoders-v2 work.
_FEAT: np.ndarray | None = None


def _feat() -> np.ndarray:
    global _FEAT
    if _FEAT is None:
        cards = pl.read_parquet(DATA / "cards_features.parquet")
        num = cards.drop(["card_id", "name"]).cast(pl.Float32)
        _FEAT = np.zeros((cards["card_id"].max() + 1, num.width), dtype=np.float32)
        _FEAT[cards["card_id"].to_numpy()] = num.to_numpy()
    return _FEAT


# ---------------------------------------------------------------------------
# Fetch layer — the only network touchpoint, injectable for tests
# ---------------------------------------------------------------------------
_last_request_t = 0.0


def _throttle() -> None:
    global _last_request_t
    wait = THROTTLE_S - (time.monotonic() - _last_request_t)
    if wait > 0:
        time.sleep(wait)
    _last_request_t = time.monotonic()


def _http_post(path: str, body: dict, retries: int = 4) -> dict:
    """POST to a Kaggle EpisodeService endpoint (listings only — replay downloads
    moved to the authenticated client, see ``_default_fetcher``). Throttled like
    the fetch path (M10: 22 unthrottled listings in a row earned a 429) and
    backing off on rate limits. Raises a runbook-pointing error offline."""
    try:
        import requests

        for attempt in range(retries):
            _throttle()
            resp = requests.post(f"{BASE_URL}/{path}", json=body, timeout=30)
            if resp.status_code == 429 and attempt < retries - 1:
                wait = 30.0 * (attempt + 1)
                print(f"[NET] 429 from {path}, backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
    except Exception as e:  # noqa: BLE001 — every failure mode gets the same remedy
        raise RuntimeError(
            f"[NET] cannot reach kaggle.com ({type(e).__name__}: {e}). This sandbox has "
            "no Kaggle access — run the M7.0 runbook in docs/M7.md on a machine with "
            "credentials, or download a replay manually (`kaggle competitions replay "
            "<id>`) and load it with `python -m rl.kaggle_ingest import-file ...`."
        ) from e


def _default_fetcher(episode_id: int) -> dict:
    """Download one replay via the authenticated Kaggle API client.

    Kaggle retired the unauthenticated ``EpisodeService/GetEpisodeReplay`` POST;
    replays are now served only through the official client (GET
    ``/api/v1/competitions/episodes/{id}/replay``), which needs
    ``~/.kaggle/kaggle.json`` credentials. ``ListEpisodes`` still lives on the
    old endpoint — see ``_http_post``.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import (
            ApiGetEpisodeReplayRequest,
        )

        api = KaggleApi()
        api.authenticate()
        request = ApiGetEpisodeReplayRequest()
        request.episode_id = int(episode_id)
        with api.build_kaggle_client() as kaggle:
            response = kaggle.competitions.competition_api_client.get_episode_replay(request)
            # Despite the FileDownload annotation, the SDK hands back the raw
            # requests.Response (FileDownload.prepare_from is the identity).
            response.raise_for_status()
            return json.loads(response.content)
    except Exception as e:  # noqa: BLE001 — every failure mode gets the same remedy
        raise RuntimeError(
            f"[NET] Kaggle replay download failed ({type(e).__name__}: {e}). Needs the "
            "`kaggle` package and ~/.kaggle/kaggle.json credentials — run the M7.0 "
            "runbook in docs/M7.md, or download a replay manually from the episode "
            "page and load it with `python -m rl.kaggle_ingest import-file ...`."
        ) from e


def fetch_agent_logs(episode_id: int, agent_index: int,
                     cache: Path | None = None) -> list:
    """Per-step agent logs for OUR seat (M19): a list (one entry per agent
    call) of ``[{"duration": s, "stdout": "...", "stderr": "..."}]``. The
    shipped agent writes one ``NN|{json}`` net-internals line per decision on
    stderr (submission/main.py) — this is the only channel that carries them;
    the replay JSON has none. Kaggle serves logs only for the caller's own
    team's agent (403 for the opponent seat). Immutable gzip cache, throttled
    on miss, same contract as ``fetch_episode``."""
    path = Path(cache or LOGS_DIR) / f"episode_{episode_id}_agent{agent_index}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    _throttle()
    try:
        import tempfile

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        with tempfile.TemporaryDirectory() as tmp_dir:
            api.competition_episode_agent_logs(int(episode_id), int(agent_index),
                                               path=tmp_dir)
            files = list(Path(tmp_dir).glob("*.json"))
            _require(len(files) == 1,
                     f"agent-logs download produced {len(files)} json files")
            raw = json.loads(files[0].read_text())
    except SchemaError:
        raise
    except Exception as e:  # noqa: BLE001 — every failure mode gets the same remedy
        raise RuntimeError(
            f"[NET] Kaggle agent-logs download failed ({type(e).__name__}: {e}). "
            "Needs the `kaggle` package + ~/.kaggle/kaggle.json, and works only "
            "for OUR OWN seat (opponent logs are 403-private)."
        ) from e
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # atomic: cache stays immutable/resumable
    tmp.write_bytes(gzip.compress(json.dumps(raw).encode()))
    tmp.replace(path)
    return raw


def _unwrap_replay(payload) -> dict:
    """Accept env.toJSON() dict, a {'replay': '<json>'} wrapper, or a raw JSON string."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "steps" not in payload and "replay" in payload:
        inner = payload["replay"]
        payload = json.loads(inner) if isinstance(inner, str) else inner
    _require(
        isinstance(payload, dict) and "steps" in payload,
        "replay payload has no 'steps'; "
        + (f"top-level keys = {sorted(payload)[:12]}" if isinstance(payload, dict)
           else f"payload is {type(payload).__name__}"),
    )
    return payload


def fetch_episode(episode_id: int, cache: Path | None = None, fetcher=None) -> dict:
    """Return an episode replay dict; immutable gzip cache, throttled on miss."""
    path = Path(cache or RAW_DIR) / f"episode_{episode_id}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    _throttle()
    raw = _unwrap_replay((fetcher or _default_fetcher)(episode_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # atomic: cache stays immutable/resumable
    tmp.write_bytes(gzip.compress(json.dumps(raw).encode()))
    tmp.replace(path)
    return raw


def import_file(path: Path, episode_id: int, cache: Path | None = None) -> dict:
    """Load a manually-downloaded replay (kaggle CLI / browser) into the cache."""
    raw = _unwrap_replay(Path(path).read_text())
    out = Path(cache or RAW_DIR) / f"episode_{episode_id}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps(raw).encode()))
    return raw


def _time_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):  # protobuf-style {"seconds": ...}
        v = _get(v, "seconds", default=v)
    return str(v)


def list_episodes(submission_id: int | None = None, team_id: int | None = None,
                  fetch_fn=None) -> pl.DataFrame:
    """List episodes for one submission or one team (exactly one required).

    Returns one row per episode: episode_id, create/end time, and per-seat
    submission_id / reward / updated_score. Not cached — listings go stale.
    """
    if (submission_id is None) == (team_id is None):
        raise ValueError("pass exactly one of submission_id / team_id")
    body = ({"submissionId": int(submission_id)} if submission_id is not None
            else {"teamId": int(team_id)})
    payload = (fetch_fn or _http_post)("ListEpisodes", body)
    if "episodes" not in payload and isinstance(payload.get("result"), dict):
        payload = payload["result"]
    episodes = _get(payload, "episodes", default=[])
    _require(isinstance(episodes, list),
             f"ListEpisodes response has no episode list; keys = {sorted(payload)[:12]}")

    rows = []
    for ep in episodes:
        row = {
            "episode_id": int(_get(ep, "id", "episodeId", default=0)),
            "create_time": _time_str(_get(ep, "createTime", "create_time")),
            "end_time": _time_str(_get(ep, "endTime", "end_time")),
        }
        agents = _get(ep, "agents", default=[]) or []
        for seat in (0, 1):
            a = agents[seat] if seat < len(agents) else {}
            sub = _get(a, "submissionId", "submission_id")
            reward = _get(a, "reward")
            score = _get(a, "updatedScore", "updated_score")
            team = _get(a, "teamId", "team_id")  # M10: the snowball channel —
            # our own listings reveal opponents' team ids, whose listings
            # reveal THEIR opponents (harvest targeting without a leaderboard)
            row[f"submission_id_{seat}"] = None if sub is None else int(sub)
            row[f"reward_{seat}"] = None if reward is None else float(reward)
            row[f"updated_score_{seat}"] = None if score is None else float(score)
            row[f"team_id_{seat}"] = None if team is None else int(team)
        rows.append(row)
    return pl.DataFrame(rows, schema=_LISTING_SCHEMA)


_LISTING_SCHEMA = {
    "episode_id": pl.Int64, "create_time": pl.Utf8, "end_time": pl.Utf8,
    "submission_id_0": pl.Int64, "submission_id_1": pl.Int64,
    "reward_0": pl.Float64, "reward_1": pl.Float64,
    "updated_score_0": pl.Float64, "updated_score_1": pl.Float64,
    "team_id_0": pl.Int64, "team_id_1": pl.Int64,
}


def targets(top_k: int = 30, min_score: float = 600.0, exclude=()) -> list[int]:
    """Snowball harvest targets: the highest-scoring opponent submission ids
    already recorded in episodes.parquet (their listings reveal their whole
    episode history, and those episodes reveal THEIR opponents). Prints a
    ready-to-paste `refresh --opp-subs ...` line. Pass our own submission ids
    as `exclude` — the our_seat flag only covers the fetched episode's seat,
    not every appearance."""
    df = pl.read_parquet(EPISODES_PQ)
    seats = [df.select(pl.col(f"submission_id_{s}").alias("sub"),
                       pl.col(f"updated_score_{s}").alias("score"),
                       (pl.col("our_seat") == s).alias("ours"))
             for s in (0, 1)]
    pool = (pl.concat(seats)
              .filter(~pl.col("ours").fill_null(False)
                      & ~pl.col("sub").is_in(list(exclude) or [-1])
                      & pl.col("sub").is_not_null() & pl.col("score").is_not_null())
              .group_by("sub").agg(pl.col("score").max())
              .filter(pl.col("score") >= min_score)
              .sort("score", descending=True).head(top_k))
    for r in pool.iter_rows(named=True):
        print(f"  {r['sub']:>10}  {r['score']:8.1f}", flush=True)
    subs = pool["sub"].to_list()
    print(f"refresh --opp-subs {' '.join(map(str, subs))}", flush=True)
    return subs


def leaderboard(top: int = 50) -> list[dict]:
    """Top leaderboard teams via the authenticated Kaggle client (M10 harvest
    targeting). Returns [{team_id, team_name, score}] best-first and prints a
    ready-to-paste `refresh --teams ...` line. Field names vary across kaggle
    package versions — extracted best-effort with a loud failure."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        entries = api.competition_leaderboard_view(COMPETITION)
    except Exception as e:  # noqa: BLE001 — every failure mode gets the same remedy
        raise RuntimeError(
            f"[NET] Kaggle leaderboard fetch failed ({type(e).__name__}: {e}). Needs "
            "the `kaggle` package and ~/.kaggle/kaggle.json credentials — or harvest "
            "team ids via the snowball channel instead (episodes.parquet team_id_* "
            "columns filled by `refresh`)."
        ) from e

    def _attr(obj, *names):
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return None

    rows = []
    for e in entries[:top]:
        team = _attr(e, "teamId", "team_id")
        rows.append({
            "team_id": None if team is None else int(team),
            "team_name": _attr(e, "teamName", "team_name", "teamNameNullable"),
            "score": float(_attr(e, "score") or 0.0),
        })
    if rows and all(r["team_id"] is None for r in rows):
        raise RuntimeError(
            "leaderboard entries carry no team id in this kaggle package version; "
            f"first entry fields: {sorted(vars(entries[0]))[:20]}")
    for r in rows:
        print(f"  {r['team_id']:>10}  {r['score']:8.1f}  {r['team_name']}", flush=True)
    ids = [str(r["team_id"]) for r in rows if r["team_id"] is not None]
    print(f"refresh --teams {' '.join(ids)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Parse layer — pure, schema-tolerant
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ParsedEpisode:
    episode_id: int
    decks: list[list[int] | None]        # per seat; None = not extractable
    deck_step: int | None                # step index where decks were found (off-by-one evidence)
    rewards: list[float | None]          # per seat; None = errored agent
    statuses: list[str | None]
    n_steps: int
    submission_ids: list[int | None]     # per seat, if serialized in the replay
    team_names: list[str | None]
    our_seat: int | None                 # seat whose submission id is in our_submission_ids
    has_opponent_obs: bool               # substantive observations for BOTH seats?
    has_visualize: bool
    warnings: list[str]


def _coerce_deck(action) -> list[int] | None:
    """A deck action in any plausible serialization -> 60 ints, else None."""
    if isinstance(action, dict):
        for key in ("action", "deck", "cards", "submission"):
            if key in action:
                return _coerce_deck(action[key])
        return None
    if not isinstance(action, (list, tuple)) or len(action) != DECK_SIZE:
        return None
    try:
        return [int(x) for x in action]
    except (TypeError, ValueError):
        return None


def _substantive_obs(agent_state: dict) -> bool:
    obs = agent_state.get("observation")
    return isinstance(obs, dict) and bool(set(obs) - _TRIVIAL_OBS_KEYS)


def parse_episode(raw: dict, our_submission_ids=(), episode_id: int | None = None) -> ParsedEpisode:
    """Extract decks, outcomes, and schema evidence from one replay dict.

    Tolerant by design: anything missing becomes a warning, never a crash — an
    outcomes-only schema is an anticipated go/no-go branch (docs/M7-plan.md §7).
    Hard assertions live in the `verify` CLI command instead.
    """
    if episode_id is None:
        episode_id = int(_get(raw, "id", "episodeId", default=-1))
    steps = raw.get("steps")
    _require(isinstance(steps, list) and steps,
             f"episode {episode_id}: 'steps' missing or empty; keys = {sorted(raw)[:12]}")
    _require(all(isinstance(s, list) and len(s) >= 2 for s in steps[:3]),
             f"episode {episode_id}: steps are not per-agent lists of >=2 entries; "
             f"steps[0] = {type(steps[0]).__name__}")
    warnings: list[str] = []

    # Decks: the env contract says each agent's FIRST action is its 60-card list,
    # but whether that lands at steps[0] or steps[1] is exactly the off-by-one
    # M7.0 verifies — scan the first three steps and record where we found it.
    decks: list[list[int] | None] = [None, None]
    deck_steps: list[int | None] = [None, None]
    for seat in (0, 1):
        for si, step in enumerate(steps[:3]):
            deck = _coerce_deck(step[seat].get("action")) if isinstance(step[seat], dict) else None
            if deck is not None:
                decks[seat], deck_steps[seat] = deck, si
                legal, reasons = validate_deck(deck)
                if not legal:
                    warnings.append(f"seat {seat} deck fails validate_deck: {reasons}")
                break
        if decks[seat] is None:
            warnings.append(f"seat {seat}: no 60-int deck action in steps 0-2")
    if deck_steps[0] is not None and deck_steps[1] is not None and deck_steps[0] != deck_steps[1]:
        warnings.append(f"seats found decks at different steps: {deck_steps}")
    deck_step = deck_steps[0] if deck_steps[0] is not None else deck_steps[1]

    # Rewards/statuses: top-level fields, falling back to the last step's per-agent state.
    last = steps[-1]
    rewards = raw.get("rewards")
    if not (isinstance(rewards, list) and len(rewards) >= 2):
        rewards = [last[s].get("reward") if isinstance(last[s], dict) else None for s in (0, 1)]
        warnings.append("no top-level 'rewards'; used last step's per-agent rewards")
    rewards = [None if r is None else float(r) for r in rewards[:2]]
    statuses = raw.get("statuses")
    if not (isinstance(statuses, list) and len(statuses) >= 2):
        statuses = [last[s].get("status") if isinstance(last[s], dict) else None for s in (0, 1)]
        warnings.append("no top-level 'statuses'; used last step's per-agent statuses")
    statuses = [None if s is None else str(s) for s in statuses[:2]]

    # Identity: best-effort — the listing metadata is the authoritative join key.
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    team_names = _get(info, "TeamNames", "teamNames", default=[None, None]) or [None, None]
    team_names = [None if t is None else str(t) for t in list(team_names)[:2]] + [None, None]
    team_names = team_names[:2]
    submission_ids: list[int | None] = [None, None]
    for seat in (0, 1):
        agent_info = steps[0][seat].get("info") if isinstance(steps[0][seat], dict) else None
        sub = _get(agent_info, "SubmissionId", "submissionId") if isinstance(agent_info, dict) else None
        submission_ids[seat] = None if sub is None else int(sub)
    ours = {int(s) for s in our_submission_ids}
    our_seat = next((s for s in (0, 1) if submission_ids[s] in ours), None) if ours else None

    # The two go/no-go schema facts (§5.2): are BOTH seats' observations populated
    # mid-game (unlocks BC-on-winners), and is the visualize blob embedded?
    has_opponent_obs = bool(steps[1:]) and all(
        any(_substantive_obs(step[seat]) for step in steps[1:] if isinstance(step[seat], dict))
        for seat in (0, 1)
    )
    first = steps[0][0] if isinstance(steps[0][0], dict) else {}
    has_visualize = any(
        isinstance(zone, dict) and "visualize" in zone
        for zone in (first, first.get("info"), first.get("observation"))
    )

    return ParsedEpisode(
        episode_id=episode_id, decks=decks, deck_step=deck_step, rewards=rewards,
        statuses=statuses, n_steps=len(steps), submission_ids=submission_ids,
        team_names=team_names, our_seat=our_seat, has_opponent_obs=has_opponent_obs,
        has_visualize=has_visualize, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Refresh — listings + fetch loop -> data/kaggle/episodes.parquet
# ---------------------------------------------------------------------------
_EPISODES_SCHEMA = {
    **_LISTING_SCHEMA,
    "team_0": pl.Utf8, "team_1": pl.Utf8,
    "status_0": pl.Utf8, "status_1": pl.Utf8,
    "n_steps": pl.Int64, "deck_step": pl.Int64, "our_seat": pl.Int64,
    "decks_extracted": pl.Int64,
    "has_opp_obs": pl.Boolean, "has_visualize": pl.Boolean,
    "fetched_at": pl.Float64,
}


def refresh(our_subs: list[int], top_team_ids=(), max_new: int = 500,
            fetcher=None, list_fn=None, opp_subs=()) -> pl.DataFrame:
    """List episodes for our submissions (+ optional opponent subs / teams),
    fetch+parse the new ones, and merge into episodes.parquet. Resumable: the raw
    cache is immutable, and a failed episode is skipped (it stays absent from the
    parquet, so the next refresh retries it against the cache).

    opp_subs (M10): opponent SUBMISSION ids to list — the snowball harvest
    channel. Kaggle retired the ListEpisodes team filter (probed 2026-07-15:
    only submissionId and ids[] are accepted, teamId 400s), so top-team
    harvesting walks the opponent graph instead: `targets` mines the parquet
    for the highest-scoring opponent subs, their episodes reveal THEIR
    opponents, repeat. top_team_ids is kept for the day the filter returns."""
    list_fn = list_fn or list_episodes
    listings = [list_fn(submission_id=s) for s in list(our_subs) + list(opp_subs)]
    listings += [list_fn(team_id=t) for t in top_team_ids]
    listing = pl.concat(listings).unique(subset="episode_id") if listings else \
        pl.DataFrame(schema=_LISTING_SCHEMA)

    known = (pl.read_parquet(EPISODES_PQ)["episode_id"].to_list()
             if EPISODES_PQ.exists() else [])
    todo = (listing.filter(~pl.col("episode_id").is_in(known))
                   .sort("episode_id", descending=True).head(max_new))
    print(f"refresh: {listing.height} listed, {todo.height} new (max_new={max_new})", flush=True)

    rows = []
    for k, meta in enumerate(todo.iter_rows(named=True), 1):
        eid = meta["episode_id"]
        try:
            ep = parse_episode(fetch_episode(eid, fetcher=fetcher), our_subs, episode_id=eid)
        except (SchemaError, RuntimeError) as e:
            print(f"[{k}/{todo.height}] episode {eid}: SKIP ({e})", flush=True)
            continue
        our_seat = ep.our_seat
        if our_seat is None:  # replay carried no submission ids -> infer from the listing
            our_seat = next((s for s in (0, 1) if meta[f"submission_id_{s}"] in set(our_subs)), None)
        rows.append({
            **meta,
            "team_0": ep.team_names[0], "team_1": ep.team_names[1],
            "status_0": ep.statuses[0], "status_1": ep.statuses[1],
            "n_steps": ep.n_steps, "deck_step": ep.deck_step, "our_seat": our_seat,
            "decks_extracted": sum(d is not None for d in ep.decks),
            "has_opp_obs": ep.has_opponent_obs, "has_visualize": ep.has_visualize,
            "fetched_at": time.time(),
        })
        print(f"[{k}/{todo.height}] episode {eid}: decks={rows[-1]['decks_extracted']}/2 "
              f"obs={ep.has_opponent_obs} warnings={len(ep.warnings)}", flush=True)

    new = pl.DataFrame(rows, schema=_EPISODES_SCHEMA)
    if EPISODES_PQ.exists():
        old = pl.read_parquet(EPISODES_PQ).filter(
            ~pl.col("episode_id").is_in(new["episode_id"].to_list()))
        # schema migration: rows written before a column existed get nulls
        old = old.with_columns([pl.lit(None, dtype=dt).alias(c)
                                for c, dt in _EPISODES_SCHEMA.items()
                                if c not in old.columns]).select(list(_EPISODES_SCHEMA))
        new = pl.concat([old, new])
    new = new.sort("episode_id")
    EPISODES_PQ.parent.mkdir(parents=True, exist_ok=True)
    new.write_parquet(EPISODES_PQ)
    print(f"episodes.parquet: {new.height} rows", flush=True)
    return new


# ---------------------------------------------------------------------------
# Harvest — raw cache -> data/kaggle/opp_decks.parquet (+ archetypes)
# ---------------------------------------------------------------------------
_OPP_SCHEMA = {
    "episode_id": pl.Int64, "seat": pl.Int64, "submission_id": pl.Int64, "team": pl.Utf8,
    "deck": pl.List(pl.Int32), "deck_hash": pl.Utf8, "archetype": pl.Utf8,
    "is_ours": pl.Boolean, "reward": pl.Float64, "score": pl.Float64,
    "won": pl.Boolean, "n_steps": pl.Int64,
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _core_vec(ids: list[int]) -> np.ndarray:
    """L2-normalized pooled FEAT of the deck's Pokémon core — the representation the
    98%-accuracy archetype probe validated (rl/encoders.py note). No longer used for
    archetype LABELS (see _assign_archetypes); rl/determinize.py matches revealed
    cards against meta decks with it."""
    v = _feat()[[i for i in ids if _ft[i]["is_pokemon"]]].sum(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _archetype_name(ids: list[int]) -> str:
    """Slug of the deck's top-2 Pokémon (by copies, then HP)."""
    mons = [i for i in ids if _ft[i]["is_pokemon"]]
    counts = Counter(_ft[i]["name"] for i in mons)
    hp = {_ft[i]["name"]: _ft[i]["hp"] for i in mons}
    top = sorted(counts, key=lambda n: (-counts[n], -(hp[n] or 0)))[:2]
    return "+".join(_slug(n) for n in top) or "no_pokemon"


def _assign_archetypes(uniq: list[tuple[str, list[int], int]]) -> dict[str, str]:
    """(deck_hash, ids, n_games) -> {deck_hash: archetype}. Each deck is named from
    its OWN top-2 Pokémon; decks sharing that core share the label. Replaces the
    stat-feature cosine cluster-join (ARCHETYPE_COS=0.95), which collapsed distinct
    decks into the most-played cluster and produced unstable labels (M24/M25)."""
    return {dh: _archetype_name(ids) for dh, ids, _n in uniq}


def harvest_decks(min_games: int = 3) -> pl.DataFrame:
    """Re-scan the raw cache into opp_decks.parquet; return the opponent-deck summary.

    The cache (not episodes.parquet) is the source of truth so parser fixes reprocess
    history. Returns one row per opponent deck_hash seen >= min_games times:
    deck_hash, archetype, n_games, wins, mean_score, deck.
    """
    meta = ({r["episode_id"]: r for r in pl.read_parquet(EPISODES_PQ).iter_rows(named=True)}
            if EPISODES_PQ.exists() else {})
    rows = []
    for path in sorted(RAW_DIR.glob("episode_*.json.gz")):
        eid = int(path.stem.split("_")[1].split(".")[0])
        ep = parse_episode(json.loads(gzip.decompress(path.read_bytes())), episode_id=eid)
        m = meta.get(eid, {})
        for seat in (0, 1):
            if ep.decks[seat] is None:
                continue
            reward, opp_reward = ep.rewards[seat], ep.rewards[1 - seat]
            rows.append({
                "episode_id": eid, "seat": seat,
                "submission_id": m.get(f"submission_id_{seat}") or ep.submission_ids[seat],
                "team": m.get(f"team_{seat}") or ep.team_names[seat],
                "deck": ep.decks[seat], "deck_hash": deck_hash(ep.decks[seat]),
                "archetype": None,  # assigned below over unique hashes
                "is_ours": m.get("our_seat") == seat,
                "reward": reward, "score": m.get(f"updated_score_{seat}"),
                "won": (reward > opp_reward) if None not in (reward, opp_reward) else None,
                "n_steps": ep.n_steps,
            })

    df = pl.DataFrame(rows, schema=_OPP_SCHEMA)
    if df.height:
        counts = Counter(df["deck_hash"].to_list())
        by_hash = {r["deck_hash"]: r["deck"] for r in df.iter_rows(named=True)}
        labels = _assign_archetypes([(h, by_hash[h], n) for h, n in counts.items()])
        df = df.with_columns(pl.col("deck_hash").replace_strict(labels).alias("archetype"))
    OPP_DECKS_PQ.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OPP_DECKS_PQ)
    print(f"opp_decks.parquet: {df.height} seat-rows, {df['deck_hash'].n_unique()} unique decks",
          flush=True)

    summary = (df.filter(~pl.col("is_ours").fill_null(False))
                 .group_by("deck_hash")
                 .agg(pl.first("archetype"), pl.len().alias("n_games"),
                      pl.col("won").sum().alias("wins"),
                      pl.col("score").mean().alias("mean_score"), pl.first("deck"))
                 .filter(pl.col("n_games") >= min_games)
                 .sort("n_games", descending=True))
    print(f"harvest: {summary.height} opponent decks with >= {min_games} games", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Meta field — frozen data/kaggle/meta_v<N>/ snapshot + collector-style specs
# ---------------------------------------------------------------------------
def build_meta_field(top_k: int = 8, dedupe_by: str = "archetype") -> list[OpponentSpec]:
    """Top-k harvested opponent decks, weighted by frequency x mean score (§5.3),
    frozen into a versioned snapshot and returned as ("generic", deck_csv_path) specs
    for the league anchor field and the collector opponent pool."""
    if dedupe_by not in ("archetype", "deck_hash"):
        raise ValueError(f"dedupe_by must be 'archetype' or 'deck_hash', got {dedupe_by!r}")
    df = pl.read_parquet(OPP_DECKS_PQ).filter(~pl.col("is_ours").fill_null(False))
    # Representative deck per group = its most-played exact list (top deck_hash).
    keys = ["deck_hash"] if dedupe_by == "deck_hash" else [dedupe_by, "deck_hash"]
    extra = [pl.first("archetype")] if dedupe_by == "deck_hash" else []
    rep = (df.group_by(keys)
             .agg(pl.len().alias("n"), pl.first("deck"), *extra)
             .sort("n", descending=True)
             .unique(subset=dedupe_by, keep="first", maintain_order=True)
             .drop("n"))
    groups = (df.group_by(dedupe_by)
                .agg(pl.len().alias("n_games"),
                     pl.col("score").fill_null(1.0).mean().alias("mean_score"))
                .with_columns((pl.col("n_games") * pl.col("mean_score")).alias("weight"))
                .join(rep, on=dedupe_by)
                .sort("weight", descending=True)
                .head(top_k))

    versions = [int(p.name.removeprefix("meta_v")) for p in KAGGLE_DIR.glob("meta_v*")
                if p.name.removeprefix("meta_v").isdigit()]
    snap = KAGGLE_DIR / f"meta_v{max(versions, default=0) + 1}"
    snap.mkdir(parents=True)

    specs, manifest = [], []
    for g in groups.iter_rows(named=True):
        ids = list(g["deck"])
        legal, reasons = validate_deck(ids)
        if not legal:  # snapshot decks feed real games — hard fail, unlike the parser
            raise ValueError(f"harvested deck {g['deck_hash'][:12]} is illegal: {reasons}")
        csv = snap / f"deck_{g['deck_hash'][:12]}.csv"
        _write_deck_csv(csv, ids)
        specs.append(("generic", str(csv)))
        manifest.append({"hash": g["deck_hash"], "archetype": g["archetype"],
                         "n_games": g["n_games"], "weight": g["weight"], "csv": csv.name})
    (snap / "manifest.json").write_text(json.dumps({
        "version": snap.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"top_k": top_k, "dedupe_by": dedupe_by, "labeler": ARCHETYPE_LABELER},
        "decks": manifest,
    }, indent=2))
    print(f"{snap.name}: froze {len(specs)} decks", flush=True)
    return specs


# ---------------------------------------------------------------------------
# Forensics — why did a submission win/lose, by opponent archetype
# ---------------------------------------------------------------------------
def forensics(submission_id: int) -> dict:
    """W/L(/D/err) for one of our submissions, split by opponent archetype, with
    game-length stats. Prints a table and returns the same as a dict. First-KO turn
    is deferred until the obs schema is confirmed (needs step-level parsing)."""
    eps = pl.read_parquet(EPISODES_PQ).filter(
        (pl.col("submission_id_0") == submission_id) | (pl.col("submission_id_1") == submission_id))
    opp = ({(r["episode_id"], r["seat"]): r["archetype"]
            for r in pl.read_parquet(OPP_DECKS_PQ).iter_rows(named=True)}
           if OPP_DECKS_PQ.exists() else {})

    games = []
    for r in eps.iter_rows(named=True):
        seat = 0 if r["submission_id_0"] == submission_id else 1
        mine, theirs = r[f"reward_{seat}"], r[f"reward_{1 - seat}"]
        outcome = ("err" if None in (mine, theirs)
                   else "win" if mine > theirs else "loss" if mine < theirs else "draw")
        games.append({"outcome": outcome, "n_steps": r["n_steps"],
                      "archetype": opp.get((r["episode_id"], 1 - seat), "unknown")})

    def _bucket(rows: list[dict]) -> dict:
        n = len(rows)
        w = sum(g["outcome"] == "win" for g in rows)
        steps = [g["n_steps"] for g in rows if g["n_steps"] is not None]
        return {"n": n, "wins": w, "losses": sum(g["outcome"] == "loss" for g in rows),
                "draws": sum(g["outcome"] == "draw" for g in rows),
                "errors": sum(g["outcome"] == "err" for g in rows),
                "wr": w / n if n else None,
                "mean_steps": sum(steps) / len(steps) if steps else None}

    by_arch = {a: _bucket([g for g in games if g["archetype"] == a])
               for a in sorted({g["archetype"] for g in games})}
    report = {
        "submission_id": submission_id, **_bucket(games),
        "by_archetype": by_arch,
        "mean_steps_by_outcome": {
            o: (lambda s: sum(s) / len(s) if s else None)(
                [g["n_steps"] for g in games if g["outcome"] == o and g["n_steps"] is not None])
            for o in ("win", "loss")},
        "first_ko": None,  # needs confirmed obs schema (M7.0 [NET] spike)
    }

    b = _bucket(games)
    print(f"submission {submission_id}: {b['n']} games, "
          f"{b['wins']}W-{b['losses']}L-{b['draws']}D ({b['errors']} err), "
          f"wr={b['wr']:.2f}" if b["n"] else f"submission {submission_id}: no episodes",
          flush=True)
    for arch, s in sorted(by_arch.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  vs {arch:<40} {s['wins']:>3}W {s['losses']:>3}L {s['draws']:>2}D  "
              f"wr={s['wr']:.2f}  steps={s['mean_steps']:.0f}"
              if s["mean_steps"] is not None else
              f"  vs {arch:<40} {s['wins']:>3}W {s['losses']:>3}L {s['draws']:>2}D",
              flush=True)
    return report


# ---------------------------------------------------------------------------
# verify — the M7.0 spike command (verification item 3: HARD asserts + schema report)
# ---------------------------------------------------------------------------
def verify(episode_id: int) -> ParsedEpisode:
    """Assert the schema bets on one cached episode and print the go/no-go report."""
    raw = fetch_episode(episode_id)
    ep = parse_episode(raw, episode_id=episode_id)
    for seat in (0, 1):
        assert ep.decks[seat] is not None, \
            f"seat {seat}: no 60-int deck action found in steps 0-2 — schema bet FAILED"
        legal, reasons = validate_deck(ep.decks[seat])
        assert legal, f"seat {seat} deck fails validate_deck: {reasons}"

    print(f"episode {episode_id}: schema VERIFIED", flush=True)
    print(f"  top-level keys : {sorted(raw)}", flush=True)
    print(f"  n_steps        : {ep.n_steps}", flush=True)
    print(f"  deck_step      : {ep.deck_step}   (the off-by-one answer)", flush=True)
    for seat in (0, 1):
        print(f"  seat {seat} deck    : hash={deck_hash(ep.decks[seat])[:12]} "
              f"rewards={ep.rewards[seat]} status={ep.statuses[seat]} "
              f"sub={ep.submission_ids[seat]} team={ep.team_names[seat]}", flush=True)
    print(f"  opponent obs   : {ep.has_opponent_obs}   (BC-on-winners go/no-go)", flush=True)
    print(f"  visualize blob : {ep.has_visualize}   (rl/replay.py forensics)", flush=True)
    for w in ep.warnings:
        print(f"  warning        : {w}", flush=True)
    print("record deck_step + obs verdict in docs/M7.md and docs/DECISIONS.md; "
          "paste the observed schema into this module's docstring.", flush=True)
    return ep


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list episodes for a submission or team")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--sub", type=int)
    g.add_argument("--team", type=int)

    s = sub.add_parser("fetch", help="fetch one episode into the cache")
    s.add_argument("--episode", type=int, required=True)

    s = sub.add_parser("agent-logs", help="fetch our agent's per-step logs for an episode")
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--agent-index", type=int, default=None,
                   help="seat of our agent (default: our_seat from episodes.parquet)")

    s = sub.add_parser("import-file", help="load a manually-downloaded replay JSON")
    s.add_argument("path", type=Path)
    s.add_argument("--episode", type=int, required=True)

    s = sub.add_parser("verify", help="hard-assert the schema bets on a cached episode")
    s.add_argument("--episode", type=int, required=True)

    s = sub.add_parser("leaderboard", help="top team ids/scores for harvest targeting")
    s.add_argument("--top", type=int, default=50)

    s = sub.add_parser("targets", help="snowball targets: top opponent subs in parquet")
    s.add_argument("--top-k", type=int, default=30)
    s.add_argument("--min-score", type=float, default=600.0)
    s.add_argument("--exclude", type=int, nargs="*", default=[],
                   help="our own submission ids (our_seat only covers fetched seats)")

    s = sub.add_parser("refresh", help="list + fetch + parse new episodes into parquet")
    s.add_argument("--subs", type=int, nargs="+", required=True)
    s.add_argument("--opp-subs", type=int, nargs="*", default=[],
                   help="opponent submission ids to snowball (see `targets`)")
    s.add_argument("--teams", type=int, nargs="*", default=[])
    s.add_argument("--max-new", type=int, default=500)

    s = sub.add_parser("harvest", help="raw cache -> opp_decks.parquet + summary")
    s.add_argument("--min-games", type=int, default=3)

    s = sub.add_parser("meta", help="freeze a meta_v<N> snapshot of top decks")
    s.add_argument("--top-k", type=int, default=8)
    s.add_argument("--dedupe-by", choices=["archetype", "deck_hash"], default="archetype")


    s = sub.add_parser("forensics", help="W/L by opponent archetype for a submission")
    s.add_argument("--sub", type=int, required=True)

    a = p.parse_args()
    if a.cmd == "list":
        print(list_episodes(submission_id=a.sub, team_id=a.team))
    elif a.cmd == "fetch":
        fetch_episode(a.episode)
        print(f"cached {RAW_DIR / f'episode_{a.episode}.json.gz'}", flush=True)
    elif a.cmd == "agent-logs":
        seat = a.agent_index
        if seat is None:
            eps = pl.read_parquet(EPISODES_PQ).filter(pl.col("episode_id") == a.episode)
            seat = None if eps.is_empty() else eps["our_seat"][0]
            if seat is None:
                raise SystemExit(f"episode {a.episode} not in {EPISODES_PQ} (or no "
                                 "our_seat) — pass --agent-index explicitly")
        fetch_agent_logs(a.episode, seat)
        print(f"cached {LOGS_DIR / f'episode_{a.episode}_agent{seat}.json.gz'}", flush=True)
    elif a.cmd == "import-file":
        import_file(a.path, a.episode)
        print(f"imported {a.path} -> episode {a.episode}", flush=True)
    elif a.cmd == "verify":
        verify(a.episode)
    elif a.cmd == "leaderboard":
        leaderboard(top=a.top)
    elif a.cmd == "targets":
        targets(top_k=a.top_k, min_score=a.min_score, exclude=a.exclude)
    elif a.cmd == "refresh":
        refresh(a.subs, opp_subs=a.opp_subs, top_team_ids=a.teams,
                max_new=a.max_new)
    elif a.cmd == "harvest":
        print(harvest_decks(min_games=a.min_games))
    elif a.cmd == "meta":
        for spec in build_meta_field(top_k=a.top_k, dedupe_by=a.dedupe_by):
            print(spec)
    elif a.cmd == "forensics":
        forensics(a.sub)


if __name__ == "__main__":
    _main()
