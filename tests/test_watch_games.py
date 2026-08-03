"""The replay bridge (scripts/watch_games.py).

The bridge rests on one contract: `rl.matchrunner.make_pilot` returns exactly
what `kaggle_environments` calls an agent — a one-argument callable that hands
back the decklist when `obs.select is None`. If either side of that drifts, the
bridge stops producing watchable games and `rl.eval.play_games` silently gets a
pilot it cannot drive, so it is pinned here.
"""
import inspect

import pytest

from scripts.watch_games import short_name


# --- the label helper (pure) -----------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("generic:lucario", "generic_lucario"),
    ("generic:decks/lucario.csv", "generic_lucario"),
    ("generic-scale:data/kaggle/grim_3121746f_deck.csv",
     "generic_scale_grim_3121746f_deck"),
    ("model:checkpoints/m39_retain_b.pt:alakazam_v2_h4",
     "model_alakazam_v2_h4"),
    ("solved:checkpoints/m39_bc_grim.pt:grim_live:800:400", "solved_400"),
])
def test_short_name_is_filesystem_safe(spec, expected):
    got = short_name(spec)
    assert got == expected
    assert got.replace("_", "").isalnum(), got
    assert len(got) <= 40


def test_short_name_never_returns_a_path_separator():
    """The label becomes a filename; a '/' would write outside replays/."""
    for spec in ("generic:decks/gen/deck_05810bff4f86.csv",
                 "ext:dist/qc_beds/wall:greattusk_wall"):
        assert "/" not in short_name(spec) and "\\" not in short_name(spec)


# --- the contract the bridge depends on ------------------------------------

def test_play_games_still_accepts_callables():
    """rl.eval.play_games' first two parameters must stay agent-shaped."""
    from rl.eval import play_games
    params = inspect.signature(play_games).parameters
    assert list(params)[:3] == ["agent_a", "agent_b", "n_games"]
    for opt in ("replay_prefix", "json_prefix", "names"):
        assert opt in params, f"play_games lost {opt}, the bridge needs it"


def test_make_pilot_returns_a_one_argument_callable():
    from rl.matchrunner import make_pilot, parse_spec
    fn, deck = make_pilot(parse_spec("generic:lucario"), "test_a")
    assert callable(fn)
    assert len(deck) == 60
    sig = inspect.signature(fn)
    required = [p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert len(required) == 1, (
        "kaggle_environments calls agents with a single observation; "
        f"this pilot wants {len(required)}")


def test_pilot_returns_the_deck_on_the_setup_prompt():
    """The kaggle contract: `obs.select is None` means "give me your deck"."""
    from rl.combat import _CARD
    if 743 not in _CARD:
        pytest.skip("needs the real card pool (conftest installs a 10-card stub)")
    from rl.matchrunner import make_pilot, parse_spec
    fn, deck = make_pilot(parse_spec("generic:lucario"), "test_b")
    assert fn({"select": None, "current": None, "logs": []}) == deck


def test_replay_saver_is_reachable_and_reports_missing_payloads():
    """save_replay returns None rather than writing a broken page when the env
    carries no `visualize` blob — that None is the bridge's only health check."""
    from rl.replay import save_replay

    class Env:
        steps = [[{}]]
        state = [type("S", (), {"reward": 1})(), type("S", (), {"reward": -1})()]

    assert save_replay(Env(), "should_not_be_written", ("a", "b")) is None
