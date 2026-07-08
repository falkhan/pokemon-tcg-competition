"""Hybrid agent: strong rule-agent play + neural value-head override (M5).

The Lucario rule agent is near-optimal offline (M1-M5). This wraps it with our trained
value head as a conservative *blunder-guard*: at high-stakes MAIN decisions, do a 1-ply
lookahead over the legal options with the value head, and override the rule agent's pick
ONLY if a different option is clearly better (by `margin`). High margin => defers to the
rule agent almost always (safe); the value head can only help at the edges.

Genuinely ours (rule priors + our neural value), and the validation gate (must beat the
stock rule agent) protects the submission from the value head's known noisiness.
"""
import numpy as np
import torch

from cg.api import to_observation_class, search_step, search_end, SelectContext
from rl.policy import OptionScorer
from rl.teacher import load_teacher
from rl.mcts import determinize, make_node, evaluate


def make_hybrid_agent(deck: list[int], value_ckpt: str = "checkpoints/hybrid_value.pt",
                      opp_deck: list[int] | None = None, margin: float = 0.20,
                      instance: str = "hyb"):
    """Rule-agent pilot + value-head 1-ply override. Returns agent(obs_dict)->list[int]."""
    rule = load_teacher(f"{instance}_rule", agent="lucario", deck="lucario")
    model = OptionScorer(); model.load_state_dict(torch.load(value_ckpt, map_location="cpu"))
    model.eval()

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        rule_picks = rule(obs_dict)
        sel = obs.select
        # Only reconsider single-pick MAIN decisions with real choice + a forward model.
        if (sel.context != SelectContext.MAIN or sel.maxCount != 1
                or len(sel.option) < 2 or obs.search_begin_input is None):
            return rule_picks
        try:
            root = make_node(determinize(obs, deck, opp_deck), model)
            rule_i = int(rule_picks[0])
            vals = np.full(len(sel.option), -1e9, dtype=np.float32)
            for i in range(len(sel.option)):
                child = search_step(root.state.searchId, [i])
                v, to_move, _, _ = evaluate(child, model)
                vals[i] = v if to_move == root.to_move else -v   # my perspective
            best_i = int(np.argmax(vals))
            override = best_i != rule_i and vals[best_i] > vals[rule_i] + margin
        finally:
            search_end()
        return [best_i] if override else rule_picks

    return agent
