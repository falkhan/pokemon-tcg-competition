"""Old-vs-new parity: rl/eval.py + rl/replay.py vs tcg/evaluation.py, and
rl/export.py + rl/gate.py vs tcg/shipping.py.

Real games / bundles need the engine and kaggle_environments — export,
parity_check, encoder_parity_check, gate_game, and bundle_isolation_check are
import-smoke only. What IS pinned: agent naming, the play_games bookkeeping
(scripted envs), replay page + manifest + index bytes (frozen clock), the
deck gate, deck-source resolution, and the isolation driver string.
"""
import datetime
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.eval as old_eval
import rl.export as old_export
import rl.gate as old_gate
import rl.replay as old_replay
import tcg.evaluation as new_eval
import tcg.shipping as new_shipping


# --- agent naming -------------------------------------------------------------

def named_callable():
    pass


@pytest.mark.parametrize("agent", [
    "random",
    "submission/main.py",
    "C:\\repo\\submission\\main.py",
    "path/to/agent.py",
    ".py",
    named_callable,
    object(),
])
def test_agent_name_parity(agent):
    assert old_eval._agent_name(agent) == new_eval.agent_name(agent)


# --- play_games with scripted environments -------------------------------------

class FakeEnv:
    """Just enough of a kaggle env: state rewards + a visualize payload."""

    def __init__(self, rewards, vis=None):
        self.state = [SimpleNamespace(reward=rewards[0], status="DONE"),
                      SimpleNamespace(reward=rewards[1], status="DONE")]
        self.steps = [[{"visualize": vis} if vis is not None else {}]] * 3

    def run(self, agents):
        self.ran = list(agents)


def make_factory(reward_script):
    calls = iter(reward_script)

    def make(name):
        assert name == "cabt"
        return FakeEnv(next(calls))
    return make


REWARD_SCRIPT = [(1, -1), (1, -1), (-1, 1), (0, 0), (None, 1), (1, None)]


@pytest.mark.parametrize("swap_slots", [True, False])
def test_play_games_parity(monkeypatch, swap_slots):
    monkeypatch.setattr(old_eval, "make", make_factory(REWARD_SCRIPT))
    old_result = old_eval.play_games("a", "b", len(REWARD_SCRIPT),
                                     swap_slots=swap_slots)
    monkeypatch.setattr(new_eval, "make", make_factory(REWARD_SCRIPT))
    new_result = new_eval.play_games("a", "b", len(REWARD_SCRIPT),
                                     swap_slots=swap_slots)
    assert old_result == new_result


# --- replay persistence ---------------------------------------------------------

class FrozenDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 7, 9, 12, 0, 0)


def frozen_datetime_module():
    module = SimpleNamespace(datetime=FrozenDatetime)
    return module


def test_save_replay_and_index_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(old_replay, "datetime", frozen_datetime_module())
    monkeypatch.setattr(new_eval, "datetime", frozen_datetime_module())

    vis = {"cards": ["Snorlax</script>", "Lucario"], "turns": 42}
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()

    games = [("g0", (1, -1)), ("g1", (-1, 1)), ("g2", (0, 0)), ("g0", (1, -1))]
    for name, rewards in games:
        env = FakeEnv(rewards, vis=vis)
        old_entry = old_replay.save_replay(env, name, ("alpha", "beta"),
                                           out_dir=old_dir)
        new_entry = new_eval.save_replay(env, name, ("alpha", "beta"),
                                         out_dir=new_dir)
        assert old_entry == new_entry

    for filename in ("g0.html", "g1.html", "g2.html", "manifest.json", "index.html"):
        assert (old_dir / filename).read_text(encoding="utf-8") == \
               (new_dir / filename).read_text(encoding="utf-8"), filename

    # The "</" script-injection guard actually fired.
    assert "</script>" not in json.loads(
        (new_dir / "manifest.json").read_text())[0]["name"]
    assert "Snorlax<\\/script>" in (new_dir / "g0.html").read_text(encoding="utf-8")


def test_save_replay_without_vis_returns_none(tmp_path):
    env = FakeEnv((1, -1), vis=None)
    assert old_replay.save_replay(env, "x", ("a", "b"), out_dir=tmp_path / "o") is None
    assert new_eval.save_replay(env, "x", ("a", "b"), out_dir=tmp_path / "n") is None


# --- shipping: deck gate, deck source, isolation driver -------------------------

def test_deck_check_parity(tmp_path, monkeypatch):
    from tcg.decks import load_deck
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    legal = load_deck("lucario")
    (bundle / "deck.csv").write_text("\n".join(map(str, legal)))
    assert old_gate.deck_check(str(bundle)) == new_shipping.deck_check(str(bundle)) == 60

    (bundle / "deck.csv").write_text("\n".join(map(str, legal[:59])))
    with pytest.raises(AssertionError):
        old_gate.deck_check(str(bundle))
    with pytest.raises(AssertionError):
        new_shipping.deck_check(str(bundle))


def test_deck_source_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(old_export, "ROOT", tmp_path)
    monkeypatch.setattr(new_shipping, "ROOT", tmp_path)
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "known.csv").write_text("1\n")
    (tmp_path / "deck.csv").write_text("2\n")
    for deck in ("known", "unknown"):
        assert old_export._deck_src(deck) == new_shipping.deck_source(deck)


def old_style_driver(base: str) -> str:
    """Transcription of the f-string driver built inline in rl/gate.py
    bundle_isolation_check — the golden reference for the template render."""
    from pathlib import Path
    bdir = Path(base).resolve()
    here = bdir.as_posix()
    root = bdir.parent.as_posix()
    return (
        "import sys,os\n"
        "nc=lambda p: os.path.normcase(os.path.abspath(p))\n"
        f"here=nc('{here}');root=nc('{root}')\n"
        "sys.path[:]=[p for p in sys.path if nc(p or os.getcwd())!=root]\n"
        f"sys.path.insert(0,'{here}')\n"
        "import main\n"
        "from cg.game import battle_start,battle_select,battle_finish\n"
        "od,_=battle_start(main.DECK,main.DECK);n=0\n"
        "while od['current']['result']<0 and n<5000:\n"
        "    od=battle_select([int(i) for i in main.agent(od)]);n+=1\n"
        "battle_finish()\n"
        "assert od['current']['result'] in (0,1,2),'game did not finish'\n"
        "assert 'polars' not in sys.modules and 'torch' not in sys.modules,'bundle pulled a heavy dep'\n"
        "rlf=nc(sys.modules['rl'].__file__)\n"
        "assert rlf.startswith(here),'imported project rl (%s), not bundled rl'%rlf\n"
        "print('ISO_OK',n)"
    )


def test_isolation_driver_render_matches_old_string(tmp_path):
    from pathlib import Path
    bundle = tmp_path / "submission_rules"
    bundle.mkdir()
    bundle_dir = Path(str(bundle)).resolve()
    rendered = new_shipping.ISOLATION_DRIVER_TEMPLATE.format(
        bundle_posix=bundle_dir.as_posix(),
        project_root_posix=bundle_dir.parent.as_posix())
    assert rendered == old_style_driver(str(bundle))


def test_export_constants_parity():
    assert (old_export.DEFAULT_CHECKPOINT, old_export.DEFAULT_DECK,
            old_export.DEFAULT_RULES_DECK) == \
           (new_shipping.DEFAULT_CHECKPOINT, new_shipping.DEFAULT_DECK,
            new_shipping.DEFAULT_RULES_DECK)
    assert old_gate.SUBMISSION_MAIN == new_shipping.SUBMISSION_MAIN


# --- v2 neural submission (M7.5): numpy forward vs torch OptionScorerV2 -------

def test_neural_v2_numpy_forward_parity(tmp_path):
    """The shipped main.py's score_options_v2 replays tcg.network.OptionScorerV2
    exactly — the ship-time parity_check's offline twin (random weights, so it
    runs without a trained checkpoint)."""
    import shutil

    torch = pytest.importorskip("torch")
    from tcg import encoders
    from tcg.network import OptionScorerV2, save_npz

    torch.manual_seed(0)
    model = OptionScorerV2()
    save_npz(model, str(tmp_path / "policy_weights.npz"))
    (tmp_path / "deck.csv").write_text("1\n" * 60)
    shutil.copy("submission/main.py", tmp_path / "main.py")
    sub = new_shipping.load_submission_module(str(tmp_path / "main.py"))

    rng = np.random.default_rng(7)
    state_ctx = rng.random(encoders.STATE_V2_DIM + encoders.N_CONTEXTS,
                           dtype=np.float32)
    state_ids = rng.integers(0, encoders.N_CARD_IDS, size=encoders.N_STATE_IDS)
    options = rng.random((7, encoders.OPTION_V2_DIM), dtype=np.float32)
    option_ids = rng.integers(0, encoders.N_CARD_IDS,
                              size=(7, encoders.N_OPTION_IDS))

    ours = sub.score_options_v2(state_ctx, state_ids, options, option_ids)
    with torch.no_grad():
        ref, _ = model(torch.from_numpy(state_ctx).unsqueeze(0),
                       torch.from_numpy(state_ids).long().unsqueeze(0),
                       torch.from_numpy(options).unsqueeze(0),
                       torch.from_numpy(option_ids).long().unsqueeze(0))
    assert np.allclose(ours, ref.squeeze(0).numpy(), rtol=1e-4, atol=1e-4)


def test_neural_v2_main_rejects_v1_weights(tmp_path):
    """A v1 npz behind the v2 main.py must fail LOUDLY at import, not play."""
    import shutil

    torch = pytest.importorskip("torch")
    from tcg.network import OptionScorer, save_npz

    torch.manual_seed(0)
    save_npz(OptionScorer(), str(tmp_path / "policy_weights.npz"))
    (tmp_path / "deck.csv").write_text("1\n" * 60)
    shutil.copy("submission/main.py", tmp_path / "main.py")
    with pytest.raises(ValueError, match="v1"):
        new_shipping.load_submission_module(str(tmp_path / "main.py"))
    assert old_gate.SUBMISSION_RULES_MAIN == new_shipping.SUBMISSION_RULES_MAIN
