"""Export a submission bundle.

Two agent kinds:
  --agent neural  (default): a (checkpoint, deck) pair -> numpy-only net bundle in submission/.
      A policy and the deck it was trained on are ONE artifact — a policy trained on Lucario
      plays Kyogre badly and vice-versa (M1 lesson) — so the deck travels with the weights.
  --agent rules: the deck-agnostic generic pilot (rl/generic_pilot.py) + a deck -> a torch-free
      bundle in submission_rules/ (bundles cg/ and a minimal rl/ package; ships the ACTUAL pilot
      module we test — no drift-prone hand-copy). See docs/M6.md.

  python -m rl.export --agent rules --deck lucario
  python -m rl.export --checkpoint bc_lucario.pt --deck lucario
  python -m rl.export                      # defaults below (neural, kyogre)
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
SUBMISSION_RULES = ROOT / "submission_rules"

# Current shipping pair = our best agent so far. bc_lucario was WORSE (BC can't
# capture the Lucario expert's hidden-state lookahead — see docs/M2 findings), so
# bc_v1+Kyogre remains the champion until PPO/search beats it.
DEFAULT_CHECKPOINT = "bc_v1.pt"
DEFAULT_DECK = "kyogre"
# Rule agent ships our strongest tested deck (M6.0 legibility work).
DEFAULT_RULES_DECK = "lucario"


def _copy_cg(dest: Path) -> None:
    """Kaggle provides numpy but NOT the cg engine bindings -- bundle cg/ with the agent."""
    shutil.copytree(ROOT / "cg", dest / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)


def _deck_src(deck: str) -> Path:
    # M24: fail loud on an unresolvable deck name (parity twin of
    # tcg/shipping.py::deck_source — the silent root-deck fallback nearly
    # shipped the kyogre fossil a third time via a typo'd name).
    src = ROOT / "decks" / f"{deck}.csv"
    if not src.exists():
        raise SystemExit(
            f"deck '{deck}' not found at {src} — refusing the root-deck "
            f"fallback (M18.1/M22c/M24 lesson: verify deck identity)")
    return src


def export_rules(deck: str = DEFAULT_RULES_DECK) -> None:
    """Bundle the rule-based generic pilot: cg/, a minimal rl/ package (only the pure-Python
    combat core + pilot — no torch/polars), and the deck. main.py is committed source."""
    SUBMISSION_RULES.mkdir(exist_ok=True)
    _copy_cg(SUBMISSION_RULES)

    rl_pkg = SUBMISSION_RULES / "rl"
    rl_pkg.mkdir(exist_ok=True)
    for name in ("__init__.py", "combat.py", "generic_pilot.py", "turn_solver.py"):
        shutil.copy(str(ROOT / "rl" / name), str(rl_pkg / name))

    shutil.copy(str(_deck_src(deck)), str(SUBMISSION_RULES / "deck.csv"))
    print(f"exported RULE agent (generic pilot) paired with deck '{deck}' -> {SUBMISSION_RULES}")


def export(checkpoint: str = DEFAULT_CHECKPOINT, deck: str = DEFAULT_DECK) -> None:
    import numpy as np
    import torch
    from rl.encoders import FEAT
    from rl.policy import OptionScorer, save_npz

    model = OptionScorer()
    ckpt_path = ROOT / "checkpoints" / checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"exporting checkpoint {checkpoint} paired with deck '{deck}'")
    else:
        torch.manual_seed(0)
        print(f"checkpoint {checkpoint} not found -- exporting seeded random weights")

    save_npz(model, str(SUBMISSION / "policy_weights.npz"))
    np.save(str(SUBMISSION / "card_features.npy"), FEAT)
    shutil.copy(str(_deck_src(deck)), str(SUBMISSION / "deck.csv"))
    _copy_cg(SUBMISSION)


if __name__ == "__main__":
    # SUPERSEDED by tcg/shipping.py (M6), which is what build_submission.sh
    # and build_submission.ps1 actually run. This module survives only as the
    # parity twin pinned by tests/test_evaluation_shipping.py — its importable
    # API stays live for those tests, but the CLI must not be used to build a
    # real bundle: export() below copies weights, card features, deck and cg/,
    # but NOT the rl/ package into submission/ (tcg/shipping.py does). Running
    # it would ship a bundle whose rl/plan.py and rl/encoders.py are whatever
    # was last left in submission/rl/ — silently, with no error. Given the
    # M18.1/M22c fossil-deck history, fail loudly instead.
    raise SystemExit(
        "rl.export is superseded and does NOT sync submission/rl/ — a bundle "
        "built with it ships stale encoders/plan modules. Use:\n"
        "  ./build_submission.sh --checkpoint <ckpt> --deck <deck>\n"
        "  (or: python -m tcg.shipping export ...)")

    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["neural", "rules"], default="neural")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--deck", default=None,
                   help="deck name in decks/ (defaults: neural=kyogre, rules=lucario)")
    args = p.parse_args()
    if args.agent == "rules":
        export_rules(args.deck or DEFAULT_RULES_DECK)
    else:
        export(args.checkpoint, args.deck or DEFAULT_DECK)
