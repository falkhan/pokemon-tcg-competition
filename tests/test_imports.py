"""Every tcg module imports cleanly under the test stubs, and ``import tcg``
itself stays side-effect free (no submodule imports, no engine calls)."""
import importlib
import sys

import pytest


def test_import_tcg_is_side_effect_free():
    for name in [name for name in sys.modules if name.startswith("tcg")]:
        del sys.modules[name]
    import tcg  # noqa: F401
    assert not any(name.startswith("tcg.") for name in sys.modules)


CG_ONLY_MODULES = ["tcg.models", "tcg.constants", "tcg.library", "tcg.combat",
                   "tcg.pilot", "tcg.decks", "tcg.teachers"]
# The rule-submission bundle ships these rl/ modules verbatim: pure Python on
# the cg API only (rl/gate.py bundle_isolation_check is the on-engine pin;
# this import tier is the offline half — M7.4a).
RL_CG_ONLY_MODULES = ["rl.combat", "rl.generic_pilot", "rl.turn_solver"]
NUMPY_MODULES = ["tcg.selfplay", "tcg.value_training"]
# M7 modules live in rl/ only (no tcg twin — M7-plan risk 6)
M7_MODULES = ["rl.matchrunner", "rl.kaggle_ingest", "rl.deck_build", "rl.league"]
POLARS_MODULES = ["tcg.encoders"] + M7_MODULES
TORCH_MODULES = ["tcg.network", "tcg.behavior_cloning", "tcg.ppo"]
KAGGLE_ENV_MODULES = ["tcg.evaluation", "tcg.shipping"]


@pytest.mark.parametrize("name", CG_ONLY_MODULES)
def test_cg_only_modules_import(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", RL_CG_ONLY_MODULES)
def test_rl_bundle_modules_import_without_heavy_deps(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", NUMPY_MODULES)
def test_numpy_modules_import(name):
    pytest.importorskip("numpy")
    importlib.import_module(name)


@pytest.mark.parametrize("name", POLARS_MODULES)
def test_polars_modules_import(name):
    pytest.importorskip("numpy")
    pytest.importorskip("polars")
    importlib.import_module(name)


@pytest.mark.parametrize("name", TORCH_MODULES)
def test_torch_modules_import(name):
    pytest.importorskip("numpy")
    pytest.importorskip("polars")
    pytest.importorskip("torch")
    importlib.import_module(name)


@pytest.mark.parametrize("name", KAGGLE_ENV_MODULES)
def test_kaggle_env_modules_import(name):
    pytest.importorskip("numpy")
    pytest.importorskip("polars")
    importlib.import_module(name)
