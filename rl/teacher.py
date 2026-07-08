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

TEACHER_PATH = Path(__file__).resolve().parent.parent / "sample-agent" / "main.py"


def load_teacher(instance_name: str):
    """Exec a fresh, isolated instance of the teacher module. Returns its agent callable.

    instance_name must be unique per live instance (e.g. "teacher_p0", "teacher_p1").
    """
    module_name = f"_teacher_{instance_name}"
    spec = importlib.util.spec_from_file_location(module_name, TEACHER_PATH)
    module = importlib.util.module_from_spec(spec)

    cwd = os.getcwd()
    os.chdir(TEACHER_PATH.parent)          # so the teacher's "../deck.csv" resolves
    try:
        sys.modules[module_name] = module  # so dataclass/module internals resolve
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)

    return module.agent
