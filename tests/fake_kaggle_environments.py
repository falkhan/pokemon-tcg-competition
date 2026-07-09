"""A fake ``kaggle_environments`` module, installed into sys.modules on import.

The real package isn't needed for tests: ``rl/eval.py`` / ``tcg/evaluation.py``
(and the gates) only bind its ``make`` name at import time. Tests monkeypatch
the module-bound ``make`` in the consuming module with a factory returning
scripted FakeEnv objects; calling this placeholder directly is an error.
"""
import sys
import types


def make(name):
    raise NotImplementedError("monkeypatch the consuming module's make")


def _install():
    module = types.ModuleType("kaggle_environments")
    module.make = make
    sys.modules.setdefault("kaggle_environments", module)


_install()
