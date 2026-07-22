"""Offline tests for rl/kaggle_ingest.py against canned episode-replay fixtures.

The real Kaggle schema is only verifiable with network access (the M7.0 [NET] spike,
runbook in docs/M7.md); these tests pin the parser's tolerance contract, the cache /
throttle behavior, and the harvest -> archetype -> meta-field pipeline on synthetic
payloads shaped like kaggle_environments ``env.toJSON()`` output. No engine needed.
"""
import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("polars")
pytest.importorskip("numpy")

import rl.kaggle_ingest as ki  # noqa: E402
from rl.deck_search import validate_deck  # noqa: E402

DECKS = Path(__file__).resolve().parent.parent / "decks"
LUCARIO = [int(x) for x in (DECKS / "lucario.csv").read_text().split()]
IONO = [int(x) for x in (DECKS / "iono.csv").read_text().split()]
KYOGRE = [int(x) for x in (DECKS / "kyogre.csv").read_text().split()]


def make_replay(deck0=LUCARIO, deck1=IONO, *, deck_step=0, rewards=(1, -1),
                n_decisions=6, with_obs=False, with_visualize=False, wrap=None,
                episode_id=101, sub_ids=(None, None)):
    """Minimal env.toJSON()-shaped payload; the deck action lands at `deck_step`."""
    def agent(seat, si):
        a = {"action": None, "reward": 0, "status": "ACTIVE", "info": {}, "observation": {}}
        if si == deck_step:
            a["action"] = [deck0, deck1][seat]
        elif si > deck_step:
            a["action"] = [0]  # ordinary in-game decision
        if si == 0 and sub_ids[seat] is not None:
            a["info"]["SubmissionId"] = sub_ids[seat]
        if with_obs and si > deck_step:
            a["observation"] = {"remainingOverageTime": 60, "select": {"options": [1, 2]}}
        else:
            a["observation"] = {"remainingOverageTime": 60}
        return a

    steps = []
    for si in range(deck_step + 1 + n_decisions):
        step = [agent(0, si), agent(1, si)]
        if si == len(range(deck_step + 1 + n_decisions)) - 1:
            for seat in (0, 1):
                step[seat]["reward"] = rewards[seat]
                step[seat]["status"] = "DONE"
        steps.append(step)
    if with_visualize:
        steps[0][0]["visualize"] = {"turns": []}

    raw = {"id": episode_id, "name": "pokemon_tcg", "version": "1.0",
           "configuration": {}, "steps": steps, "rewards": list(rewards),
           "statuses": ["DONE", "DONE"], "info": {"TeamNames": ["us", "them"]}}
    if wrap == "replay-string":
        return {"replay": json.dumps(raw)}
    return raw


@pytest.fixture
def kdirs(tmp_path, monkeypatch):
    """Point every module path constant at tmp_path (resolved at call time)."""
    kag = tmp_path / "kaggle"
    monkeypatch.setattr(ki, "KAGGLE_DIR", kag)
    monkeypatch.setattr(ki, "RAW_DIR", kag / "raw")
    monkeypatch.setattr(ki, "EPISODES_PQ", kag / "episodes.parquet")
    monkeypatch.setattr(ki, "OPP_DECKS_PQ", kag / "opp_decks.parquet")
    return kag


def _seed_cache(episode_id, raw):
    ki.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (ki.RAW_DIR / f"episode_{episode_id}.json.gz").write_bytes(
        gzip.compress(json.dumps(raw).encode()))


# --- unwrap / schema errors -------------------------------------------------

def test_unwrap_accepts_dict_replay_string_and_raw_string():
    raw = make_replay()
    assert ki._unwrap_replay(raw)["id"] == 101
    assert ki._unwrap_replay(make_replay(wrap="replay-string"))["id"] == 101
    assert ki._unwrap_replay(json.dumps(raw))["id"] == 101


def test_unwrap_rejects_stepless_payload_with_keys_in_message():
    with pytest.raises(ki.SchemaError, match="no 'steps'.*wrongKey"):
        ki._unwrap_replay({"wrongKey": 1})


# --- parse_episode ----------------------------------------------------------

@pytest.mark.parametrize("deck_step", [0, 1])
def test_parse_finds_deck_and_records_deck_step(deck_step):
    ep = ki.parse_episode(make_replay(deck_step=deck_step))
    assert ep.decks == [LUCARIO, IONO]
    assert ep.deck_step == deck_step
    assert not any("no 60-int deck" in w for w in ep.warnings)


def test_parse_coerces_dict_wrapped_and_stringly_int_actions():
    raw = make_replay()
    raw["steps"][0][0]["action"] = {"action": [str(i) for i in LUCARIO]}
    ep = ki.parse_episode(raw)
    assert ep.decks[0] == LUCARIO


def test_parse_missing_deck_yields_none_plus_warning_not_raise():
    raw = make_replay()
    raw["steps"][0][1]["action"] = None  # seat 1 never submits a deck
    for step in raw["steps"][1:]:
        step[1]["action"] = [0]
    ep = ki.parse_episode(raw)
    assert ep.decks[1] is None
    assert any("seat 1: no 60-int deck" in w for w in ep.warnings)


def test_parse_illegal_60_list_kept_with_validate_warning():
    bad = [LUCARIO[0]] * 60  # 60 copies of one card: right length, illegal
    ep = ki.parse_episode(make_replay(deck0=bad))
    assert ep.decks[0] == bad
    assert any("fails validate_deck" in w for w in ep.warnings)


def test_parse_rewards_statuses_fall_back_to_last_step():
    raw = make_replay(rewards=(0, 1))
    del raw["rewards"], raw["statuses"]
    ep = ki.parse_episode(raw)
    assert ep.rewards == [0.0, 1.0]
    assert ep.statuses == ["DONE", "DONE"]
    assert any("last step" in w for w in ep.warnings)


def test_parse_detects_opponent_obs_and_visualize_flags():
    ep = ki.parse_episode(make_replay(with_obs=True, with_visualize=True))
    assert ep.has_opponent_obs and ep.has_visualize
    ep = ki.parse_episode(make_replay())  # blank obs (only remainingOverageTime)
    assert not ep.has_opponent_obs and not ep.has_visualize


def test_parse_identifies_our_seat_from_submission_ids():
    ep = ki.parse_episode(make_replay(sub_ids=(111, 222)), our_submission_ids=[222])
    assert ep.our_seat == 1
    assert ep.submission_ids == [111, 222]


# --- fetch / cache ----------------------------------------------------------

def test_fetch_cache_hit_never_calls_fetcher(kdirs):
    _seed_cache(7, make_replay(episode_id=7))
    def boom(_eid):
        raise AssertionError("network hit on cache hit")
    raw = ki.fetch_episode(7, fetcher=boom)
    assert raw["id"] == 7


def test_fetch_writes_gzip_and_throttles(kdirs, monkeypatch):
    sleeps = []
    monkeypatch.setattr(ki.time, "sleep", sleeps.append)
    monkeypatch.setattr(ki, "_last_request_t", ki.time.monotonic())  # pretend a recent request
    raw = ki.fetch_episode(8, fetcher=lambda eid: make_replay(episode_id=eid))
    assert raw["id"] == 8
    path = ki.RAW_DIR / "episode_8.json.gz"
    assert json.loads(gzip.decompress(path.read_bytes()))["id"] == 8
    assert sleeps and 0 < sleeps[0] <= ki.THROTTLE_S


def test_fetch_offline_error_mentions_runbook(kdirs, monkeypatch):
    monkeypatch.setattr(ki, "BASE_URL", "https://127.0.0.1:1/nowhere")
    with pytest.raises(RuntimeError, match=r"\[NET\].*docs/M7\.md"):
        ki.fetch_episode(9)


def test_import_file_bridges_manual_downloads(kdirs, tmp_path):
    f = tmp_path / "episode-5-replay.json"
    f.write_text(json.dumps(make_replay(episode_id=5, wrap="replay-string")))
    ki.import_file(f, 5)
    assert ki.fetch_episode(5, fetcher=None)["id"] == 5  # served from cache


# --- deck_hash --------------------------------------------------------------

def test_deck_hash_is_order_invariant_and_stable():
    assert ki.deck_hash(LUCARIO) == ki.deck_hash(list(reversed(LUCARIO)))
    assert ki.deck_hash(LUCARIO) != ki.deck_hash(IONO)
    assert len(ki.deck_hash(LUCARIO)) == 40


# --- refresh ----------------------------------------------------------------

def _fake_listing(episode_ids, subs=(111, 222)):
    import polars as pl
    return pl.DataFrame(
        [{"episode_id": e, "create_time": "1", "end_time": "2",
          "submission_id_0": subs[0], "submission_id_1": subs[1],
          "reward_0": 1.0, "reward_1": -1.0,
          "updated_score_0": 650.0, "updated_score_1": 700.0}
         for e in episode_ids],
        schema=ki._LISTING_SCHEMA)


def test_refresh_writes_episodes_parquet_and_is_idempotent(kdirs):
    import polars as pl
    list_fn = lambda submission_id=None, team_id=None: _fake_listing([1, 2])  # noqa: E731
    fetcher = lambda eid: make_replay(episode_id=eid)  # noqa: E731
    df = ki.refresh([111], list_fn=list_fn, fetcher=fetcher)
    assert df.height == 2
    assert df["our_seat"].to_list() == [0, 0]  # inferred from the listing
    assert df["decks_extracted"].to_list() == [2, 2]
    df2 = ki.refresh([111], list_fn=list_fn, fetcher=fetcher)  # nothing new
    assert df2.height == 2
    assert pl.read_parquet(ki.EPISODES_PQ).height == 2


# --- harvest + archetypes ---------------------------------------------------

def _seed_harvest(kdirs, n_lucario=3, n_kyogre=1):
    """Cache episodes (us=Lucario vs them=Iono xN, plus them=Kyogre) + parquet."""
    eids = []
    for i in range(n_lucario):
        _seed_cache(10 + i, make_replay(episode_id=10 + i))
        eids.append(10 + i)
    for i in range(n_kyogre):
        _seed_cache(50 + i, make_replay(deck1=KYOGRE, episode_id=50 + i))
        eids.append(50 + i)
    ki.refresh([111], list_fn=lambda **kw: _fake_listing(eids),
               fetcher=lambda eid: (_ for _ in ()).throw(AssertionError("cache miss")))
    return eids


def test_harvest_writes_opp_decks_and_min_games_filters_summary(kdirs):
    import polars as pl
    _seed_harvest(kdirs)
    summary = ki.harvest_decks(min_games=3)
    full = pl.read_parquet(ki.OPP_DECKS_PQ)
    assert full.height == 8  # 4 episodes x 2 seats, all decks extracted
    assert full.filter(pl.col("is_ours")).height == 4  # seat 0 = sub 111 = ours
    # summary: opponents only, Iono seen 3x passes, Kyogre 1x filtered out
    assert summary.height == 1
    assert summary["deck_hash"][0] == ki.deck_hash(IONO)
    assert summary["n_games"][0] == 3


def test_archetypes_named_from_own_top2_pokemon(kdirs):
    # same Pokémon core, one trainer swapped -> different hash, same archetype
    lucario_variant = list(LUCARIO)
    trainers = [i for i, c in enumerate(lucario_variant) if not ki._ft[c]["is_pokemon"]]
    swap_pool = [c for c in ki._ft if ki._ft[c]["is_item"] and c not in lucario_variant]
    lucario_variant[trainers[0]] = swap_pool[0]
    labels = ki._assign_archetypes([
        (ki.deck_hash(LUCARIO), LUCARIO, 5),
        (ki.deck_hash(lucario_variant), lucario_variant, 2),
        (ki.deck_hash(IONO), IONO, 3),
    ])
    assert labels[ki.deck_hash(LUCARIO)] == labels[ki.deck_hash(lucario_variant)]
    assert labels[ki.deck_hash(IONO)] != labels[ki.deck_hash(LUCARIO)]
    # the label is the deck's OWN top-2 core — independent of what else is in the pool
    # (the old cosine cluster-join let a big cluster absorb and rename smaller decks)
    assert labels[ki.deck_hash(IONO)] == ki._archetype_name(IONO)
    solo = ki._assign_archetypes([(ki.deck_hash(IONO), IONO, 1)])
    assert solo[ki.deck_hash(IONO)] == labels[ki.deck_hash(IONO)]


def test_build_meta_field_snapshot_and_specs(kdirs):
    _seed_harvest(kdirs)
    ki.harvest_decks(min_games=1)
    specs = ki.build_meta_field(top_k=8)
    assert specs and all(s[0] == "generic" for s in specs)
    snap = ki.KAGGLE_DIR / "meta_v1"
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["version"] == "meta_v1"
    assert len(manifest["decks"]) == len(specs)
    for spec in specs:  # every frozen deck re-validates and round-trips as csv
        ids = [int(x) for x in Path(spec[1]).read_text().split()]
        assert validate_deck(ids)[0]
    ki.build_meta_field(top_k=8)
    assert (ki.KAGGLE_DIR / "meta_v2").exists()


# --- forensics + bc-shards audit ---------------------------------------------

def test_forensics_by_archetype_counts(kdirs):
    _seed_harvest(kdirs)
    ki.harvest_decks(min_games=1)
    report = ki.forensics(111)
    assert report["n"] == 4 and report["wins"] == 4  # rewards fixture: seat 0 always wins
    assert sum(b["n"] for b in report["by_archetype"].values()) == 4
    assert report["first_ko"] is None  # deferred until the obs schema is confirmed


def test_bc_shards_returns_empty_when_no_obs(kdirs, capsys):
    _seed_harvest(kdirs)
    assert ki.bc_shards_from_opponents(min_score=600) == []
    assert "no encodable seats" in capsys.readouterr().out
