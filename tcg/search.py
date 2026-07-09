"""Determinized MCTS + the hybrid rule/value agent built on it.

Merges the old ``rl/mcts.py`` and ``rl/hybrid.py`` — the hybrid agent is a
1-ply consumer of the same determinize/evaluate machinery.

MCTS (docs/M2.md pivot): AlphaGo-Zero-style — one two-headed net (policy =
move priors, value = leaf eval), MCTS as the lookahead the value/policy net
can't express on its own. Because the game is imperfect-information, we
DETERMINIZE once per real move (guess the opponent's hidden cards via
search_begin), then search that concrete game with the engine's forward model.

Hybrid (M5): the Lucario rule agent is near-optimal offline (M1-M5).
``make_hybrid_agent`` wraps it with our trained value head as a conservative
*blunder-guard*: at high-stakes MAIN decisions, do a 1-ply lookahead over the
legal options with the value head, and override the rule agent's pick ONLY if
a different option is clearly better (by ``margin``). High margin => defers to
the rule agent almost always (safe); the value head can only help at the edges.
(M5 verdict: the value head can't rank options well enough to help — kept for
the record and future value heads.)
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch

from cg.api import SelectContext, search_begin, search_step, search_end, to_observation_class

from tcg.encoders import encode_context, encode_option, encode_state
from tcg.network import OptionScorer
from tcg.teachers import load_teacher

# Fillers for hidden opponent cards at determinization (mirrors the official sample).
FILLER_POKEMON = 1072   # Snorlax (a Basic; keeps search_begin's setup rules happy)
FILLER_ENERGY = 1       # Basic {G} Energy
MAX_ACTIONS = 64        # cap on enumerated multi-select combinations per node
C_PUCT = 1.5            # exploration constant in PUCT
UNEVALUATED = -1e9      # argmax floor for options the hybrid hasn't evaluated


@dataclass
class Node:
    """One search-tree node = one concrete game state, plus per-action edge stats.

    All values are stored from THIS node's ``to_move`` perspective, so selection
    can read Q = W/N directly with no sign juggling (the sign flip lives in
    backprop). N/W/P are the canonical AlphaZero edge-stat names.
    """
    state: object                 # cg.api.SearchState
    to_move: int                  # which player chooses here (obs.current.yourIndex)
    value: float                  # leaf value (net or terminal), to_move's perspective
    terminal: bool
    actions: list                 # list[list[int]] — candidate option-index picks
    P: np.ndarray                 # prior per action  (softmax of policy head)
    # Sized by len(actions), so they're built in __post_init__ rather than passed.
    N: np.ndarray = field(init=False)         # visit count per action
    W: np.ndarray = field(init=False)         # summed value per action (to_move perspective)
    children: list = field(init=False)        # list[Node | None], lazily expanded

    def __post_init__(self):
        n_actions = len(self.actions)
        self.N = np.zeros(n_actions, dtype=np.float64)
        self.W = np.zeros(n_actions, dtype=np.float64)
        self.children = [None] * n_actions


def enumerate_actions(select, probs: np.ndarray):
    """Turn the engine's (minCount, maxCount) selection into candidate actions.

    Common case maxCount==1 -> one action per option. Multi-select -> combinations
    of size maxCount (capped), prior = mean of member option-probs. Returns
    (actions: list[list[int]], action_priors: np.ndarray summing to 1)."""
    n_options = len(select.option)
    pick_count = max(1, select.maxCount)
    if pick_count == 1:
        actions = [[i] for i in range(n_options)]
        priors = probs.copy()
    else:
        combos = list(combinations(range(n_options), pick_count))[:MAX_ACTIONS]
        actions = [list(combo) for combo in combos]
        priors = np.array([probs[list(combo)].mean() for combo in combos],
                          dtype=np.float64)
    prior_sum = priors.sum()
    priors = (priors / prior_sum if prior_sum > 0
              else np.full(len(actions), 1.0 / len(actions)))
    return actions, priors


@torch.no_grad()
def evaluate(state, model: OptionScorer):
    """Neural (or terminal) evaluation of a search state.

    Returns (value, to_move, actions, priors). ``value`` is from ``to_move``'s
    perspective: +1 this player has won, -1 lost, else the value head's estimate."""
    observation = state.observation
    current = observation.current
    to_move = current.yourIndex
    if current.result >= 0:                             # terminal
        value = (0.0 if current.result == 2
                 else (1.0 if current.result == to_move else -1.0))
        return value, to_move, [], np.array([])

    state_ctx = np.concatenate([
        encode_state(current),
        encode_context(observation.select.context)]).astype(np.float32)
    option_vectors = np.stack([encode_option(option, observation)
                               for option in observation.select.option]).astype(np.float32)
    logits, value = model(torch.from_numpy(state_ctx).unsqueeze(0),
                          torch.from_numpy(option_vectors).unsqueeze(0))
    probs = torch.softmax(logits.squeeze(0), dim=0).numpy()
    actions, priors = enumerate_actions(observation.select, probs)
    return float(value), to_move, actions, priors


def make_node(state, model: OptionScorer) -> Node:
    """Evaluate a state and wrap it as a fresh Node ready for search."""
    value, to_move, actions, priors = evaluate(state, model)
    terminal = state.observation.current.result >= 0
    return Node(state=state, to_move=to_move, value=value, terminal=terminal,
                actions=actions, P=priors)


def determinize(observation, deck: list[int], opp_deck: list[int] | None = None):
    """Guess the opponent's hidden cards and open a concrete search game (root state).

    We know our own deck list, so sample our unseen zones from it. For the opponent:
    if ``opp_deck`` is given, sample their hidden zones from that realistic decklist;
    otherwise fall back to crude placeholders (Snorlax + basic energy). Modelling the
    opponent with a real deck makes the lookahead reason about the game actually being
    played. One determinization per real move."""
    import random
    current = observation.current
    my_index = current.yourIndex
    mine = current.players[my_index]
    opponent = current.players[1 - my_index]
    need_active = len(opponent.active) > 0 and opponent.active[0] is None

    if opp_deck is None:
        opponent_deck = [FILLER_POKEMON] * opponent.deckCount
        opponent_prize = [FILLER_ENERGY] * len(opponent.prize)
        opponent_hand = [FILLER_ENERGY] * opponent.handCount
    else:
        opponent_deck = random.sample(opp_deck, opponent.deckCount)
        opponent_prize = random.sample(opp_deck, len(opponent.prize))
        opponent_hand = random.sample(opp_deck, opponent.handCount)

    return search_begin(
        observation,
        your_deck=random.sample(deck, mine.deckCount),
        your_prize=random.sample(deck, len(mine.prize)),
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=[FILLER_POKEMON] if need_active else [],  # must be a Basic Pokémon
    )


def mcts_search(root: Node, model: OptionScorer, n_sims: int) -> np.ndarray:
    """Run n_sims MCTS simulations from ``root``; return root visit counts (root.N).

    Each simulation is Select -> Expand -> Backprop (Evaluate happens inside
    make_node when a leaf is created):

      1. SELECT: from ``root``, descend by PUCT until you hit either a terminal node
         or an action whose child is None (unexpanded). Record the (node, action)
         path you took.
             exploitation = node.W[a] / node.N[a]     if node.N[a] > 0 else 0.0
             exploration  = C_PUCT * node.P[a] * sqrt(node.N.sum()) / (1 + node.N[a])
             a = argmax(exploitation + exploration)   # all from node.to_move's perspective
      2. EXPAND: at the chosen unexpanded action, advance the engine
             child_state = search_step(node.state.searchId, node.actions[a])
             node.children[a] = make_node(child_state, model)
         The leaf is that new child (or the terminal node you landed on).
      3. BACKPROP: let v = leaf.value, p = leaf.to_move. For every (node, a) on the
         path, add from THAT node's perspective:
             node.N[a] += 1
             node.W[a] += v if node.to_move == p else -v
         (negamax: a position good for one player is bad for the other.)

    A node with no actions (terminal) is never selected into. Read: the official
    sample's search loop (reference/reinforcement-learning-and-mcts-sample-code.ipynb)
    and AlphaGo Zero.
    """
    for _ in range(n_sims):
        node, path = root, []
        while not node.terminal:
            exploitation = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            exploration = C_PUCT * node.P * np.sqrt(node.N.sum()) / (1 + node.N)
            action_index = int(np.argmax(exploitation + exploration))
            path.append((node, action_index))
            if node.children[action_index] is None:
                child_state = search_step(node.state.searchId,
                                          node.actions[action_index])
                node.children[action_index] = make_node(child_state, model)
                node = node.children[action_index]
                break
            node = node.children[action_index]
        leaf_value, leaf_to_move = node.value, node.to_move
        for visited, action_index in path:
            visited.N[action_index] += 1
            visited.W[action_index] += (leaf_value if visited.to_move == leaf_to_move
                                        else -leaf_value)
    return root.N


def make_mcts_agent(model: OptionScorer, deck: list[int], n_sims: int = 16,
                    opp_deck: list[int] | None = None):
    """Return an agent(obs_dict)->list[int] that picks moves by MCTS.

    opp_deck: assumed opponent decklist for determinization (realistic modelling);
    None uses crude Snorlax/energy fillers."""
    def agent(obs_dict: dict) -> list[int]:
        observation = to_observation_class(obs_dict)
        if observation.select is None:
            return deck
        if observation.search_begin_input is None:    # no forward model available here
            raise RuntimeError("search_begin_input missing; MCTS needs the forward model")

        root = make_node(determinize(observation, deck, opp_deck), model)
        try:
            if len(root.actions) <= 1:                 # nothing to search
                pick = (root.actions[0] if root.actions
                        else list(range(observation.select.maxCount)))
            else:
                visits = mcts_search(root, model, n_sims)
                pick = root.actions[int(np.argmax(visits))]
        finally:
            search_end()
        return [int(i) for i in pick]
    return agent


def make_hybrid_agent(deck: list[int], value_ckpt: str = "checkpoints/hybrid_value.pt",
                      opp_deck: list[int] | None = None, margin: float = 0.20,
                      instance: str = "hyb"):
    """Rule-agent pilot + value-head 1-ply override. Returns agent(obs_dict)->list[int].

    Genuinely ours (rule priors + our neural value), and the validation gate (must
    beat the stock rule agent) protects the submission from the value head's known
    noisiness."""
    rule = load_teacher(f"{instance}_rule", agent="lucario", deck="lucario")
    model = OptionScorer()
    model.load_state_dict(torch.load(value_ckpt, map_location="cpu"))
    model.eval()

    def agent(obs_dict: dict) -> list[int]:
        observation = to_observation_class(obs_dict)
        if observation.select is None:
            return deck
        rule_picks = rule(obs_dict)
        select = observation.select
        # Only reconsider single-pick MAIN decisions with real choice + a forward model.
        if (select.context != SelectContext.MAIN or select.maxCount != 1
                or len(select.option) < 2 or observation.search_begin_input is None):
            return rule_picks
        try:
            root = make_node(determinize(observation, deck, opp_deck), model)
            rule_index = int(rule_picks[0])
            option_values = np.full(len(select.option), UNEVALUATED, dtype=np.float32)
            for i in range(len(select.option)):
                child = search_step(root.state.searchId, [i])
                value, to_move, _, _ = evaluate(child, model)
                option_values[i] = value if to_move == root.to_move else -value  # my perspective
            best_index = int(np.argmax(option_values))
            override = (best_index != rule_index
                        and option_values[best_index] > option_values[rule_index] + margin)
        finally:
            search_end()
        return [best_index] if override else rule_picks

    return agent
