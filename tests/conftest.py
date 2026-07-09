"""Installs the fake ``cg`` engine BEFORE any ``rl``/``tcg`` module is imported.

Both packages build their card tables at import time from ``cg.api``, and the
real engine is gitignored — so the stub must be in sys.modules first. Plain
top-level imports (not fixtures) guarantee the ordering.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tests.fake_cg  # noqa: E402,F401  (side effect: registers cg / cg.api / cg.game)
import tests.fake_kaggle_environments  # noqa: E402,F401  (registers kaggle_environments)
