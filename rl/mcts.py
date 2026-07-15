"""Determinized MCTS with OptionScorer as priors + leaf value (docs/M2.md pivot).

AlphaGo-Zero-style: one two-headed net (policy = move priors, value = leaf eval),
MCTS as the lookahead the value/policy net can't express on its own. Because the game
is imperfect-information, we DETERMINIZE once per real move (guess the opponent's hidden
cards via search_begin), then search that concrete game with the engine's forward model.

Division of labor (as with GAE/PPO):
  Claude scaffold (done):  determinize(), evaluate(), make_node(), the agent wrapper,
                           action enumeration, engine plumbing.
  PIOTR (the core):        mcts_search() — Select (PUCT) / Expand / Backprop. Stubbed
                           below with the full spec. This is the heart of MCTS.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch

from cg.api import search_begin, search_step, search_end, to_observation_class
from .encoders import (COMBAT_SLICE, N_COMBAT, encode_context, encode_option,
                       encode_state)
from .policy import OptionScorer

# Fillers for hidden opponent cards at determinization (mirrors the official sample).
FILLER_POKEMON = 1072   # Snorlax (a Basic; keeps search_begin's setup rules happy)
FILLER_ENERGY = 1       # Basic {G} Energy
MAX_ACTIONS = 64        # cap on enumerated multi-select combinations per node
C_PUCT = 1.5            # exploration constant in PUCT


@dataclass
class Node:
    """One search-tree node = one concrete game state, plus per-action edge stats.

    All values are stored from THIS node's `to_move` perspective, so selection can
    read Q = W/N directly with no sign juggling (the sign flip lives in backprop).
    """
    state: object                 # cg.api.SearchState
    to_move: int                  # which player chooses here (obs.current.yourIndex)
    value: float                  # leaf value (net or terminal), to_move's perspective
    terminal: bool
    actions: list                 # list[list[int]] — candidate option-index picks
    P: np.ndarray                 # prior per action  (softmax of policy head)
    N: np.ndarray = field(default=None)   # visit count per action
    W: np.ndarray = field(default=None)   # summed value per action (to_move perspective)
    children: list = field(default=None)  # list[Node | None], lazily expanded

    def __post_init__(self):
        n = len(self.actions)
        self.N = np.zeros(n, dtype=np.float64)
        self.W = np.zeros(n, dtype=np.float64)
        self.children = [None] * n


def _enumerate_actions(select, probs: np.ndarray):
    """Turn the engine's (minCount, maxCount) selection into candidate actions.

    Common case maxCount==1 -> one action per option. Multi-select -> combinations
    of size maxCount (capped), prior = mean of member option-probs. Returns
    (actions: list[list[int]], action_priors: np.ndarray summing to 1)."""
    n = len(select.option)
    k = max(1, select.maxCount)
    if k == 1:
        actions = [[i] for i in range(n)]
        pri = probs.copy()
    else:
        combos = list(combinations(range(n), k))[:MAX_ACTIONS]
        actions = [list(c) for c in combos]
        pri = np.array([probs[list(c)].mean() for c in combos], dtype=np.float64)
    s = pri.sum()
    pri = pri / s if s > 0 else np.full(len(actions), 1.0 / len(actions))
    return actions, pri


@torch.no_grad()
def evaluate(state, model: OptionScorer):
    """Neural (or terminal) evaluation of a search state.

    Returns (value, to_move, actions, priors). `value` is from `to_move`'s
    perspective: +1 this player has won, -1 lost, else the value head's estimate."""
    obs = state.observation
    st = obs.current
    to_move = st.yourIndex
    if st.result >= 0:                                  # terminal
        v = 0.0 if st.result == 2 else (1.0 if st.result == to_move else -1.0)
        return v, to_move, [], np.array([])

    sc = np.concatenate([encode_state(st), encode_context(obs.select.context)])
    # Pre-M3 checkpoints (bc_v1, the M8.4b instrument stack) are exactly
    # N_COMBAT narrower — derive the cut from the model itself (matchrunner's
    # shim, signature-free so mcts_search stays untouched).
    if model.state_enc[0].in_features == sc.size - N_COMBAT:
        sc = np.delete(sc, np.s_[COMBAT_SLICE[0]:COMBAT_SLICE[1]])
    sc = sc.astype(np.float32)
    opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
    logits, value = model(torch.from_numpy(sc).unsqueeze(0), torch.from_numpy(opts).unsqueeze(0))
    probs = torch.softmax(logits.squeeze(0), dim=0).numpy()
    actions, priors = _enumerate_actions(obs.select, probs)
    return float(value), to_move, actions, priors


def make_node(state, model: OptionScorer) -> Node:
    """Evaluate a state and wrap it as a fresh Node ready for search."""
    value, to_move, actions, priors = evaluate(state, model)
    terminal = state.observation.current.result >= 0
    return Node(state=state, to_move=to_move, value=value, terminal=terminal,
                actions=actions, P=priors)


def determinize(obs, deck: list[int], opp_deck: list[int] | None = None,
                meta=None):
    """Guess the opponent's hidden cards and open a concrete search game (root state).

    We know our own deck list, so sample our unseen zones from it. For the
    opponent, in preference order: `meta` (M8.4a — a list of
    rl.determinize.MetaDeck; the L3 archetype determinizer infers their list
    from revealed cards and samples the remaining pool), else `opp_deck` (a
    fixed realistic decklist), else the crude M2 placeholders (Snorlax +
    basic energy) that measurably misled the search. One determinization per
    real move."""
    import random
    st = obs.current
    me = st.yourIndex
    mine, opp = st.players[me], st.players[1 - me]
    need_active = len(opp.active) > 0 and opp.active[0] is None

    if meta is not None:
        from rl.determinize import determinize_kwargs
        kw = determinize_kwargs(obs, meta, random.Random(random.random()))
        return search_begin(
            obs,
            your_deck=random.sample(deck, mine.deckCount),
            your_prize=random.sample(deck, len(mine.prize)), **kw)

    if opp_deck is None:
        o_deck = [FILLER_POKEMON] * opp.deckCount
        o_prize = [FILLER_ENERGY] * len(opp.prize)
        o_hand = [FILLER_ENERGY] * opp.handCount
    else:
        o_deck = random.sample(opp_deck, opp.deckCount)
        o_prize = random.sample(opp_deck, len(opp.prize))
        o_hand = random.sample(opp_deck, opp.handCount)

    return search_begin(
        obs,
        your_deck=random.sample(deck, mine.deckCount),
        your_prize=random.sample(deck, len(mine.prize)),
        opponent_deck=o_deck,
        opponent_prize=o_prize,
        opponent_hand=o_hand,
        opponent_active=[FILLER_POKEMON] if need_active else [],  # must be a Basic Pokémon
    )


# ---------------------------------------------------------------------------
# THE CORE — Piotr's part
# ---------------------------------------------------------------------------

def mcts_search(root: Node, model: OptionScorer, n_sims: int) -> np.ndarray:
    """Run n_sims MCTS simulations from `root`; return root visit counts (root.N).

    Each simulation is Select -> Expand -> Backprop (Evaluate happens inside
    make_node when a leaf is created):

      1. SELECT: from `root`, descend by PUCT until you hit either a terminal node
         or an action whose child is None (unexpanded). Record the (node, action)
         path you took.
             q = node.W[a] / node.N[a]     if node.N[a] > 0 else 0.0
             u = C_PUCT * node.P[a] * sqrt(node.N.sum()) / (1 + node.N[a])
             a = argmax(q + u)             # all from `node.to_move`'s perspective
      2. EXPAND: at the chosen unexpanded action, advance the engine
             child_state = search_step(node.state.searchId, node.actions[a])
             node.children[a] = make_node(child_state, model)
         The leaf is that new child (or the terminal node you landed on).
      3. BACKPROP: let v = leaf.value, p = leaf.to_move. For every (node, a) on the
         path, add from THAT node's perspective:
             node.N[a] += 1
             node.W[a] += v if node.to_move == p else -v
         (negamax: a position good for one player is bad for the other.)

    Guard the edge cases: a node with no actions (terminal) is never selected into;
    if root has 1 action, you can early-out. Read: the official sample's search loop
    (reference/reinforcement-learning-and-mcts-sample-code.ipynb) and AlphaGo Zero.
    """
    for _ in range(n_sims):
        node, path = root, []
        while not node.terminal:
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            u = C_PUCT * node.P * np.sqrt(node.N.sum()) / (1 + node.N)
            a = int(np.argmax(q + u))
            path.append((node, a))
            if node.children[a] is None:
                child_state = search_step(node.state.searchId, node.actions[a])
                node.children[a] = make_node(child_state, model)
                node = node.children[a]
                break
            node = node.children[a]
        v, p = node.value, node.to_move
        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += v if nd.to_move == p else -v
    return root.N


# ---------------------------------------------------------------------------
# Agent wrapper (infra, done) — plugs MCTS into eval / the submission
# ---------------------------------------------------------------------------

def make_mcts_agent(model: OptionScorer, deck: list[int], n_sims: int = 16,
                    opp_deck: list[int] | None = None, meta=None):
    """Return an agent(obs_dict)->list[int] that picks moves by MCTS.

    opp_deck: assumed opponent decklist for determinization (realistic modelling);
    meta (M8.4): rl.determinize.load_meta() list — the L3 archetype
    determinizer infers the opponent's list per decision; None uses crude
    Snorlax/energy fillers."""
    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        if obs.search_begin_input is None:            # no forward model available here
            raise RuntimeError("search_begin_input missing; MCTS needs the forward model")

        root = make_node(determinize(obs, deck, opp_deck, meta=meta), model)
        try:
            if len(root.actions) <= 1:                 # nothing to search
                pick = root.actions[0] if root.actions else list(range(obs.select.maxCount))
            else:
                visits = mcts_search(root, model, n_sims)
                pick = root.actions[int(np.argmax(visits))]
        finally:
            search_end()
        return [int(i) for i in pick]
    return agent
