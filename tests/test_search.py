"""Old-vs-new parity: rl/mcts.py + rl/hybrid.py vs tcg/search.py.

The engine's determinized-search API is scripted per test (the stub's
search_begin/search_step raise until monkeypatched). make_hybrid_agent /
make_mcts_agent end-to-end need a real teacher + forward model — import-smoke
only; their building blocks (evaluate, make_node, determinize, mcts_search)
are pinned here.
"""
import random
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

import rl.mcts as old
import tcg.search as new
from tests.builders import observation, option, player, pokemon, search_state
from tests.fake_cg import EnergyType, OptionType, SelectContext

FIGHTING = EnergyType.FIGHTING


def shared_model():
    torch.manual_seed(0)
    return old.OptionScorer()  # duck-typed; passed to BOTH old and new (bit-equal anyway)


def test_constants_parity():
    assert (old.FILLER_POKEMON, old.FILLER_ENERGY, old.MAX_ACTIONS, old.C_PUCT) == \
           (new.FILLER_POKEMON, new.FILLER_ENERGY, new.MAX_ACTIONS, new.C_PUCT)


def test_node_init_parity():
    state = object()
    priors = np.array([0.5, 0.3, 0.2])
    old_node = old.Node(state=state, to_move=0, value=0.1, terminal=False,
                        actions=[[0], [1], [2]], P=priors)
    new_node = new.Node(state=state, to_move=0, value=0.1, terminal=False,
                        actions=[[0], [1], [2]], P=priors)
    assert np.array_equal(old_node.N, new_node.N)
    assert np.array_equal(old_node.W, new_node.W)
    assert old_node.children == new_node.children == [None, None, None]
    assert old_node.N.dtype == new_node.N.dtype == np.float64


def select_namespace(n_options, max_count):
    return SimpleNamespace(option=[object()] * n_options, maxCount=max_count)


def test_enumerate_actions_parity_single_pick():
    probs = np.array([0.1, 0.6, 0.3])
    old_actions, old_priors = old._enumerate_actions(select_namespace(3, 1), probs)
    new_actions, new_priors = new.enumerate_actions(select_namespace(3, 1), probs)
    assert old_actions == new_actions == [[0], [1], [2]]
    assert np.array_equal(old_priors, new_priors)


def test_enumerate_actions_parity_multi_pick():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    old_actions, old_priors = old._enumerate_actions(select_namespace(4, 3), probs)
    new_actions, new_priors = new.enumerate_actions(select_namespace(4, 3), probs)
    assert old_actions == new_actions
    assert np.array_equal(old_priors, new_priors)
    assert np.isclose(new_priors.sum(), 1.0)


def test_enumerate_actions_parity_zero_prior_fallback():
    probs = np.zeros(3)
    old_actions, old_priors = old._enumerate_actions(select_namespace(3, 1), probs)
    new_actions, new_priors = new.enumerate_actions(select_namespace(3, 1), probs)
    assert old_actions == new_actions
    assert np.array_equal(old_priors, new_priors)
    assert np.allclose(new_priors, 1.0 / 3.0)


def live_state(search_id=0):
    me = player(active=pokemon(1, energies=[FIGHTING]), hand=[])
    opponent = player(active=pokemon(2, hp=60))
    obs = observation(me, opponent,
                      options=[option(OptionType.ATTACK, attack_id=101),
                               option(OptionType.END)],
                      context=SelectContext.MAIN)
    return search_state(obs, search_id=search_id)


@pytest.mark.parametrize("result,to_move", [(0, 0), (0, 1), (1, 0), (2, 0)])
def test_evaluate_parity_terminal(result, to_move):
    obs = observation(your_index=to_move, result=result)
    state = search_state(obs)
    model = shared_model()
    old_out = old.evaluate(state, model)
    new_out = new.evaluate(state, model)
    assert old_out[0] == new_out[0]
    assert old_out[1] == new_out[1]
    assert old_out[2] == new_out[2] == []
    assert np.array_equal(old_out[3], new_out[3])


def test_evaluate_and_make_node_parity_live():
    model = shared_model()
    old_value, old_to_move, old_actions, old_priors = old.evaluate(live_state(), model)
    new_value, new_to_move, new_actions, new_priors = new.evaluate(live_state(), model)
    assert old_value == new_value
    assert old_to_move == new_to_move
    assert old_actions == new_actions
    assert np.array_equal(old_priors, new_priors)

    old_node = old.make_node(live_state(), model)
    new_node = new.make_node(live_state(), model)
    assert (old_node.to_move, old_node.value, old_node.terminal, old_node.actions) == \
           (new_node.to_move, new_node.value, new_node.terminal, new_node.actions)
    assert np.array_equal(old_node.P, new_node.P)


@pytest.mark.parametrize("opp_deck", [None, list(range(1, 11)) * 6])
@pytest.mark.parametrize("hidden_active", [False, True])
def test_determinize_parity(monkeypatch, opp_deck, hidden_active):
    captured = {}

    def recorder_for(store, key):
        def recorder(obs, **kwargs):
            store[key] = kwargs
            return "SEARCH_STATE"
        return recorder

    monkeypatch.setattr(old, "search_begin", recorder_for(captured, "old"))
    monkeypatch.setattr(new, "search_begin", recorder_for(captured, "new"))

    def board():
        opponent = player(active=pokemon(2), deck_count=20, hand_count=5)
        if hidden_active:
            opponent.active = [None]
        return observation(player(active=pokemon(1), deck_count=25), opponent)

    deck = list(range(1, 11)) * 6
    random.seed(7)
    assert old.determinize(board(), deck, opp_deck) == "SEARCH_STATE"
    random.seed(7)
    assert new.determinize(board(), deck, opp_deck) == "SEARCH_STATE"
    assert captured["old"] == captured["new"]


def scripted_engine():
    """A deterministic 2-ply game tree keyed on (search_id, action tuple).

    Root (id 0, my move, 2 options) -> two mid states (opponent's move) ->
    terminals. Terminal results make option 0 the winning line for player 0.
    """
    me = player(active=pokemon(1, energies=[FIGHTING]))
    opponent = player(active=pokemon(2, hp=60))

    def live(your_index, search_id):
        return search_state(
            observation(me, opponent, your_index=your_index,
                        options=[option(OptionType.ATTACK, attack_id=101),
                                 option(OptionType.END)],
                        context=SelectContext.MAIN),
            search_id=search_id)

    def terminal(result, search_id):
        return search_state(observation(me, opponent, result=result),
                            search_id=search_id)

    tree = {
        (0, (0,)): live(your_index=1, search_id=1),   # my attack -> opponent moves
        (0, (1,)): live(your_index=1, search_id=2),   # my END    -> opponent moves
        (1, (0,)): terminal(result=0, search_id=3),   # I win
        (1, (1,)): terminal(result=0, search_id=4),
        (2, (0,)): terminal(result=1, search_id=5),   # opponent wins
        (2, (1,)): terminal(result=2, search_id=6),   # draw
    }

    def step(search_id, action):
        return tree[(search_id, tuple(action))]

    return step


def test_mcts_search_parity(monkeypatch):
    step = scripted_engine()
    monkeypatch.setattr(old, "search_step", step)
    monkeypatch.setattr(new, "search_step", step)
    model = shared_model()

    old_root = old.make_node(root_state(), model)
    new_root = new.make_node(root_state(), model)
    old_visits = old.mcts_search(old_root, model, n_sims=25)
    new_visits = new.mcts_search(new_root, model, n_sims=25)
    assert np.array_equal(old_visits, new_visits)
    assert np.array_equal(old_root.W, new_root.W)
    assert np.array_equal(old_root.N, new_root.N)
    # The winning line (action 0) must dominate visits — sanity that the
    # scripted tree actually exercises select/expand/backprop.
    assert new_visits[0] > new_visits[1]


def root_state():
    me = player(active=pokemon(1, energies=[FIGHTING]))
    opponent = player(active=pokemon(2, hp=60))
    return search_state(
        observation(me, opponent, your_index=0,
                    options=[option(OptionType.ATTACK, attack_id=101),
                             option(OptionType.END)],
                    context=SelectContext.MAIN),
        search_id=0)
