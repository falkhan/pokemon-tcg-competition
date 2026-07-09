"""Supervised value-head training on self-play outcomes (docs/DECISIONS.md 2026-07-08).

Same pipeline as the old ``rl/value_train.py``. Why this exists: PPO could not
train a useful critic on this sparse-reward, mirror-heavy problem (3 attempts
stalled or degraded the policy). But the *information* to predict the winner is
in the encoder (opponent archetype is observable). Training the value head
DIRECTLY — supervised regression to game outcomes, policy frozen — works:
bc_v1's original value head scored 0.62 sign-accuracy (≈ chance); a supervised
head hits ~0.87 on the same frozen features. This decouples "get a good critic"
from "don't wreck the policy," and it's the value-target step of an
AlphaGo-Zero-lite loop (self-play -> outcomes -> value head -> MCTS).

torch is imported inside the functions on purpose, so the module itself stays
importable (and its CLI printable) without the training stack loaded.

  python -m tcg.value_training collect --checkpoint checkpoints/bc_v1.pt --games 600
  python -m tcg.value_training train   --checkpoint bc_v1.pt --out bc_v1_value.pt
"""
import argparse
from pathlib import Path

import numpy as np

from tcg.decks import ROOT, load_deck
from tcg.selfplay import write_shard

DATA = ROOT / "data" / "value_train.npz"

TRAIN_FRACTION = 0.85    # train/val split over shuffled states
BATCH_SIZE = 512
OPPONENT_ROTATION = 3    # cycle through the 3 opponents below, one per game


def collect(checkpoint: str, learn_deck: str = "kyogre", n_games: int = 600,
            out: Path = DATA) -> None:
    """Play the policy (greedy) vs the opponent pool; record (state_ctx, final outcome)
    for the learner seat. Outcome is +1 win / -1 loss / 0 draw from the learner's view."""
    import random

    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish

    from tcg.encoders import encode_context, encode_option, encode_state
    from tcg.network import OptionScorer
    from tcg.teachers import load_teacher

    model = OptionScorer()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    learner_deck = load_deck(learn_deck)
    opponents = [(None, learner_deck),                # None = uniform-random legal moves
                 (load_teacher("vpl", agent="lucario", deck="lucario"),
                  load_deck("lucario")),
                 (load_teacher("vpi", agent="iono", deck="iono"), load_deck("iono"))]

    def pick(observation):
        state_ctx = np.concatenate([
            encode_state(observation.current),
            encode_context(observation.select.context)]).astype(np.float32)
        option_vectors = np.stack([encode_option(option, observation)
                                   for option in observation.select.option]).astype(np.float32)
        picks = model.act(state_ctx, option_vectors,
                          k=observation.select.maxCount, greedy=True)
        return state_ctx, [int(i) for i in picks]

    states, outcomes = [], []
    for game in range(n_games):
        opponent_act, opponent_deck = opponents[game % OPPONENT_ROTATION]
        learn_seat = game % 2
        deck_p0, deck_p1 = ((learner_deck, opponent_deck) if learn_seat == 0
                            else (opponent_deck, learner_deck))
        obs_dict, _ = battle_start(deck_p0, deck_p1)
        pending_states = []
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            observation = to_observation_class(obs_dict)
            if seat == learn_seat:
                state_ctx, picks = pick(observation)
                pending_states.append(state_ctx)
                obs_dict = battle_select(picks)
            else:
                picks = (random.sample(range(len(obs_dict["select"]["option"])),
                                       obs_dict["select"]["maxCount"])
                         if opponent_act is None else opponent_act(obs_dict))
                obs_dict = battle_select([int(i) for i in picks])
        result = obs_dict["current"]["result"]
        battle_finish()
        outcome = 0.0 if result == 2 else (1.0 if result == learn_seat else -1.0)
        states += pending_states
        outcomes += [outcome] * len(pending_states)

    out.parent.mkdir(parents=True, exist_ok=True)
    write_shard(out, {"states": states, "outcomes": outcomes},
                int32_columns=frozenset(), float32_columns=frozenset({"outcomes"}))
    print(f"collected {len(states)} states -> {out}")


def train(checkpoint: str, out: str = "bc_v1_value.pt", data: Path = DATA,
          epochs: int = 40, lr: float = 1e-3) -> None:
    """Freeze policy+body; train ONLY the value head to regress outcomes. Saves a
    checkpoint = original policy/body + the new value head."""
    import torch
    import torch.nn.functional as F

    from tcg.network import OptionScorer

    loaded = np.load(data)
    states = torch.from_numpy(loaded["states"])
    outcomes = torch.from_numpy(loaded["outcomes"])
    n_rows = len(states)
    perm = np.random.default_rng(0).permutation(n_rows)
    train_rows = perm[:int(TRAIN_FRACTION * n_rows)]
    val_rows = perm[int(TRAIN_FRACTION * n_rows):]

    model = OptionScorer()
    model.load_state_dict(torch.load(ROOT / "checkpoints" / checkpoint,
                                     map_location="cpu"))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("value_head")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr)

    def sign_accuracy(rows):
        with torch.no_grad():
            predicted = model.value_head(model.state_enc(states[rows])).squeeze(-1)
        return (((predicted > 0).float() == (outcomes[rows] > 0).float())
                .float().mean().item())

    baseline = sign_accuracy(val_rows)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(train_rows), BATCH_SIZE):
            batch_rows = train_rows[i:i + BATCH_SIZE]
            with torch.no_grad():
                features = model.state_enc(states[batch_rows])
            loss = F.mse_loss(model.value_head(features).squeeze(-1),
                              outcomes[batch_rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    torch.save(model.state_dict(), ROOT / "checkpoints" / out)
    print(f"value-head sign-acc {baseline:.2f} -> {sign_accuracy(val_rows):.2f}  "
          f"saved checkpoints/{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--checkpoint", default="checkpoints/bc_v1.pt")
    collect_parser.add_argument("--deck", default="kyogre")
    collect_parser.add_argument("--games", type=int, default=600)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--checkpoint", default="bc_v1.pt")
    train_parser.add_argument("--out", default="bc_v1_value.pt")
    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.checkpoint, args.deck, args.games)
    else:
        train(args.checkpoint, args.out)
