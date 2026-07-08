"""Kaggle entry point for the RULE-BASED submission: the deck-agnostic generic pilot
(rl/generic_pilot.py) driving a bundled deck. No neural net, no torch, no numpy needed —
just the bundled `cg` engine and the pure-Python pilot + combat core (rl/combat.py).

Built by `python -m rl.export --agent rules`, which bundles cg/, a minimal rl/ package
(combat.py + generic_pilot.py), and deck.csv alongside this file. Because we ship the
ACTUAL pilot module (not a hand-copied fork), the submission runs exactly the code we test.
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

from rl.generic_pilot import make_generic_pilot

DECK = [int(x) for x in open(os.path.join(_BASE, "deck.csv")) if x.strip()]

# `agent` MUST be the last callable defined in this module (the Kaggle harness calls it).
agent = make_generic_pilot(DECK)
