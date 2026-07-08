"""Export a (checkpoint, deck) pair into the numpy-only submission bundle.

A policy and the deck it was trained on are ONE artifact — a policy trained on
Lucario plays Kyogre badly and vice-versa (M1 lesson). So export always ships a
matching pair, and the deck travels with the weights.

  python -m rl.export --checkpoint bc_lucario.pt --deck lucario
  python -m rl.export                      # defaults below
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from rl.encoders import FEAT
from rl.policy import OptionScorer, save_npz

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"

# Current shipping pair = our best agent so far. bc_lucario was WORSE (BC can't
# capture the Lucario expert's hidden-state lookahead — see docs/M2 findings), so
# bc_v1+Kyogre remains the champion until PPO/search beats it.
DEFAULT_CHECKPOINT = "bc_v1.pt"
DEFAULT_DECK = "kyogre"


def export(checkpoint: str = DEFAULT_CHECKPOINT, deck: str = DEFAULT_DECK) -> None:
    model = OptionScorer()
    ckpt_path = ROOT / "checkpoints" / checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"exporting checkpoint {checkpoint} paired with deck '{deck}'")
    else:
        torch.manual_seed(0)
        print(f"checkpoint {checkpoint} not found -- exporting seeded random weights")

    deck_src = ROOT / "decks" / f"{deck}.csv"
    if not deck_src.exists():                      # fall back to the root deck
        deck_src = ROOT / "deck.csv"

    save_npz(model, str(SUBMISSION / "policy_weights.npz"))
    np.save(str(SUBMISSION / "card_features.npy"), FEAT)
    shutil.copy(str(deck_src), str(SUBMISSION / "deck.csv"))

    # Kaggle provides numpy but NOT the cg engine bindings -- bundle cg/ with the agent.
    shutil.copytree(ROOT / "cg", SUBMISSION / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--deck", default=DEFAULT_DECK)
    args = p.parse_args()
    export(args.checkpoint, args.deck)
