"""Isolated loader for the rule-based teacher agent (sample-agent/main.py).

Two quirks of the teacher file require this loader (docs/M1-plan.md §A1):

1. It loads its deck as "../deck.csv" relative to the CURRENT WORKING DIRECTORY,
   so we chdir into sample-agent/ for the duration of the import.
2. It keeps module-level mutable state (plan, pre_turn, ability_used). Two players
   must therefore be two SEPARATE module instances, or their globals interleave
   mid-game and corrupt decisions. Each load_teacher() call execs a fresh module.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHER_PATHS = {                          # rule-based agent per archetype
    "lucario": ROOT / "sample-agent" / "main.py",
    "iono": ROOT / "sample-agent-iono" / "main.py",
    "tuned": ROOT / "sample-agent-tuned" / "main.py",   # parameterized Lucario (M5)
}
DECK_PATHS = {name: ROOT / "decks" / f"{name}.csv"
              for name in ("kyogre", "lucario", "iono")}
DECK_PATHS["tuned"] = DECK_PATHS["lucario"]   # the tuned agent pilots the Lucario deck


def load_teacher(instance_name: str, agent: str = "lucario", deck: str | None = None,
                 weights: dict | None = None):
    """Exec a fresh, isolated instance of a rule-based agent. Returns its callable.

    instance_name must be unique per live instance (globals are module-level state).
    agent: which rule brain ("lucario" or "iono").
    deck:  which deck it pilots ("kyogre"/"lucario"/"iono"); defaults to the agent's
           own archetype. The brain's card-specific heuristics only fire on matching
           cards, so a mismatched deck is piloted on generic fallback scores (this is
           exactly how the Lucario brain played the Kyogre deck through all of M1).
    """
    path = TEACHER_PATHS[agent]
    deck = deck or agent
    module_name = f"_teacher_{instance_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    cwd = os.getcwd()
    os.chdir(path.parent)                  # so the agent's "../deck.csv" import resolves
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)

    # Override the deck the agent returns at setup (its heuristics are deck-independent).
    module.my_deck = [int(x) for x in DECK_PATHS[deck].read_text().split() if x.strip()]
    # Override tunable heuristic weights (M5 parameter search on the "tuned" agent).
    if weights:
        for k, v in weights.items():
            setattr(module, k, v)
    return module.agent
