"""Unit pins for rl/mcts.py (determinized MCTS building blocks).

Ported from the retired old-vs-new parity suite (tests/test_search.py) when
tcg/search.py was removed — the ``old == new`` assertions became direct pins.
The engine's determinized-search API is scripted per test (the stub's
search_begin/search_step raise until monkeypatched). make_mcts_agent
end-to-end needs a real teacher + forward model — import-smoke only; its
building blocks (evaluate, make_node, determinize, mcts_search) are pinned
here.
"""
import random
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

import rl.mcts as mcts
from tests.builders import observation, option, player, pokemon, search_state
from tests.fake_cg import EnergyType, OptionType, SelectContext

FIGHTING = EnergyType.FIGHTING


def seeded_model():
    torch.manual_seed(0)
    return mcts.OptionScorer()


def test_constants():
    assert (mcts.FILLER_POKEMON, mcts.FILLER_ENERGY, mcts.MAX_ACTIONS,
            mcts.C_PUCT) == (1072, 1, 64, 1.5)


def test_node_init():
    priors = np.array([0.5, 0.3, 0.2])
    node = mcts.Node(state=object(), to_move=0, value=0.1, terminal=False,
                     actions=[[0], [1], [2]], P=priors)
    assert np.array_equal(node.N, np.zeros(3))
    assert np.array_equal(node.W, np.zeros(3))
    assert node.children == [None, None, None]
    assert node.N.dtype == node.W.dtype == np.float64


def select_namespace(n_options, max_count):
    return SimpleNamespace(option=[object()] * n_options, maxCount=max_count)


def test_enumerate_actions_single_pick():
    probs = np.array([0.1, 0.6, 0.3])
    actions, priors = mcts._enumerate_actions(select_namespace(3, 1), probs)
    assert actions == [[0], [1], [2]]
    assert np.allclose(priors, probs / probs.sum())


def test_enumerate_actions_multi_pick():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    actions, priors = mcts._enumerate_actions(select_namespace(4, 3), probs)
    # Size-maxCount combinations in itertools order; prior = mean of member
    # option-probs, renormalised.
    combos = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    assert actions == [list(c) for c in combos]
    expected = np.array([probs[list(c)].mean() for c in combos])
    assert np.allclose(priors, expected / expected.sum())
    assert np.isclose(priors.sum(), 1.0)


def test_enumerate_actions_zero_prior_fallback():
    actions, priors = mcts._enumerate_actions(select_namespace(3, 1),
                                              np.zeros(3))
    assert actions == [[0], [1], [2]]
    assert np.allclose(priors, 1.0 / 3.0)


def live_state(search_id=0):
    me = player(active=pokemon(1, energies=[FIGHTING]), hand=[])
    opponent = player(active=pokemon(2, hp=60))
    obs = observation(me, opponent,
                      options=[option(OptionType.ATTACK, attack_id=101),
                               option(OptionType.END)],
                      context=SelectContext.MAIN)
    return search_state(obs, search_id=search_id)


@pytest.mark.parametrize("result,to_move,expected_value", [
    (0, 0, 1.0),    # player 0 won, player 0 to move
    (0, 1, -1.0),   # player 0 won, player 1 to move
    (1, 0, -1.0),   # player 1 won, player 0 to move
    (2, 0, 0.0),    # draw
])
def test_evaluate_terminal(result, to_move, expected_value):
    obs = observation(your_index=to_move, result=result)
    value, out_to_move, actions, priors = mcts.evaluate(search_state(obs),
                                                        seeded_model())
    assert value == expected_value
    assert out_to_move == to_move
    assert actions == []
    assert priors.size == 0


def test_evaluate_and_make_node_live():
    value, to_move, actions, priors = mcts.evaluate(live_state(),
                                                    seeded_model())
    assert to_move == 0
    assert actions == [[0], [1]]
    assert np.isclose(priors.sum(), 1.0)
    assert (priors > 0).all()
    assert isinstance(value, float)

    node = mcts.make_node(live_state(), seeded_model())  # same seeded weights
    assert (node.to_move, node.value, node.terminal, node.actions) == \
           (to_move, value, False, actions)
    assert np.array_equal(node.P, priors)


def test_make_node_terminal():
    node = mcts.make_node(search_state(observation(result=0)), seeded_model())
    assert node.terminal
    assert node.actions == []
    assert node.children == []


@pytest.mark.parametrize("opp_deck", [None, list(range(1, 11)) * 6])
@pytest.mark.parametrize("hidden_active", [False, True])
def test_determinize(monkeypatch, opp_deck, hidden_active):
    captured = {}

    def recorder(obs, **kwargs):
        captured.update(kwargs)
        return "SEARCH_STATE"

    monkeypatch.setattr(mcts, "search_begin", recorder)

    opponent = player(active=pokemon(2), deck_count=20, hand_count=5)
    if hidden_active:
        opponent.active = [None]
    board = observation(player(active=pokemon(1), deck_count=25), opponent)

    deck = list(range(1, 11)) * 6
    random.seed(7)
    assert mcts.determinize(board, deck, opp_deck) == "SEARCH_STATE"

    # Our own unseen zones are sampled from our known decklist.
    assert len(captured["your_deck"]) == 25
    assert len(captured["your_prize"]) == 6
    assert set(captured["your_deck"]) <= set(deck)
    # A hidden opponent active must be replaced by a Basic Pokémon filler.
    assert captured["opponent_active"] == \
           ([mcts.FILLER_POKEMON] if hidden_active else [])
    if opp_deck is None:
        # Crude M2 placeholders: Snorlax deck, basic-energy prizes/hand.
        assert captured["opponent_deck"] == [mcts.FILLER_POKEMON] * 20
        assert captured["opponent_prize"] == [mcts.FILLER_ENERGY] * 6
        assert captured["opponent_hand"] == [mcts.FILLER_ENERGY] * 5
    else:
        # Realistic modelling: sampled from the assumed opponent decklist.
        assert len(captured["opponent_deck"]) == 20
        assert len(captured["opponent_prize"]) == 6
        assert len(captured["opponent_hand"]) == 5
        assert set(captured["opponent_deck"]) <= set(opp_deck)


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


def root_state():
    me = player(active=pokemon(1, energies=[FIGHTING]))
    opponent = player(active=pokemon(2, hp=60))
    return search_state(
        observation(me, opponent, your_index=0,
                    options=[option(OptionType.ATTACK, attack_id=101),
                             option(OptionType.END)],
                    context=SelectContext.MAIN),
        search_id=0)


def test_mcts_search(monkeypatch):
    monkeypatch.setattr(mcts, "search_step", scripted_engine())
    model = seeded_model()

    root = mcts.make_node(root_state(), model)
    visits = mcts.mcts_search(root, model, n_sims=25)
    assert visits is root.N
    # Every simulation passes through the root exactly once.
    assert root.N.sum() == 25
    # The winning line (action 0) must dominate visits — sanity that the
    # scripted tree actually exercises select/expand/backprop.
    assert visits[0] > visits[1]
    # From the root player's perspective action 0 leads only to wins,
    # action 1 only to losses/draws.
    assert root.W[0] > root.W[1]
