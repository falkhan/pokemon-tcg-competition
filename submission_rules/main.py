"""Kaggle entry point for the RULE-BASED submission: the deck-agnostic generic pilot
(rl/generic_pilot.py) + the within-turn combo solver (rl/turn_solver.py, M7.4a) driving a
bundled deck. No neural net, no torch, no numpy needed — just the bundled `cg` engine and
the pure-Python pilot + combat core (rl/combat.py).

Built by `python -m tcg.shipping export --agent rules`, which bundles cg/, a minimal rl/ package
(combat.py + generic_pilot.py + turn_solver.py), and deck.csv alongside this file. Because
we ship the ACTUAL pilot modules (not hand-copied forks), the submission runs exactly the
code we test.
"""
import os
import sys

# Make the bundled cg/ and rl/ importable whether run locally or on Kaggle
# (Kaggle exec's this file from /kaggle_simulations/agent/).
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
for _d in [_HERE, "/kaggle_simulations/agent", "submission_rules", "."]:
    if _d and os.path.exists(os.path.join(_d, "deck.csv")):
        _BASE = _d
        break
else:
    raise FileNotFoundError("rule-agent artifacts not found (deck.csv)")

if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from rl.turn_solver import make_solver_pilot

DECK = [int(x) for x in open(os.path.join(_BASE, "deck.csv")) if x.strip()]

# `agent` MUST be the last callable defined in this module (the Kaggle harness calls it).
# M7.4a/M7.5: generic pilot wrapped with the within-turn combo solver (greedy fallback
# on any solver error — the wrapper never costs the G1 crash gate).
agent = make_solver_pilot(DECK)
