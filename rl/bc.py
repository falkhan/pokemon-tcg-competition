"""Behavior cloning from the rule-based Mega Lucario teacher (docs/M1-plan.md).

Two halves:
  COLLECTION (below, done)   — teacher self-play via the direct engine loop, decisions
                               encoded immediately and saved as sharded .npz files.
  TRAINING   (Piotr's part)  — dataset/collate, masked cross-entropy, train loop.
                               Signatures + notes at the bottom of this file.

Collect:  python -m rl.bc collect --games 1000
"""
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
from rl.policy import OptionScorer

from .encoders import (N_CONTEXTS, OPTION_DIM, STATE_DIM,
                       encode_context, encode_option, encode_state)
from .teacher import load_teacher

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bc"
ROOT = Path(__file__).resolve().parent.parent


def _load_deck(name: str | None = None) -> list[int]:
    path = (ROOT / "deck.csv") if name is None else (ROOT / "decks" / f"{name}.csv")
    return [int(x) for x in path.read_text().split() if x.strip()]


def collect_games(n_games: int, out_dir: Path = DATA_DIR, shard_size: int = 200,
                  log_every: int = 25, agent: str = "lucario", deck: str = "lucario") -> None:
    """Rule-agent self-play on a chosen (agent, deck); record every decision of BOTH players.

    Pair the agent with its OWN deck (agent="lucario", deck="lucario") so its
    card-specific heuristics fire — that's the expert-quality demonstration the
    M1 mismatch was missing (M1 cloned the Lucario brain on the Kyogre deck =
    generic play). See docs/M2 round-robin: Lucario is the strongest archetype.

    Shard layout (ragged options stored flat + per-decision lengths):
      states    (D, STATE_DIM + N_CONTEXTS) float32   state ++ context one-hot
      options   (sum_N, OPTION_DIM)         float32   all decisions' options, concatenated
      n_options (D,)                        int32     options per decision -> slice offsets
      labels    (D,)                        int32     teacher's first pick (index into that
                                                      decision's options)
      game_ids  (D,)                        int32     for the by-game train/val split
      results   (D,)                        float32   +1 win / -1 loss / 0 draw for the
                                                      deciding player (value-head target)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    deck_ids = _load_deck(deck)
    # Two ISOLATED teacher instances -- they hold per-player mutable globals.
    teachers = [load_teacher("p0", agent=agent, deck=deck),
                load_teacher("p1", agent=agent, deck=deck)]

    shard: dict[str, list] = {k: [] for k in
                              ("states", "options", "n_options", "labels", "game_ids", "results")}
    shard_idx = sum(1 for _ in out_dir.glob("shard_*.npz"))  # append after existing shards
    wins = [0, 0, 0]

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            options=np.concatenate(shard["options"]),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            labels=np.array(shard["labels"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    for game in range(n_games):
        obs_dict, start_data = battle_start(deck_ids, deck_ids)
        if start_data.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected the deck (errorType={start_data.errorType})")

        game_decisions: list[tuple] = []  # (row_index_data..., player) until result known
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            picks = teachers[player](obs_dict)

            obs = to_observation_class(obs_dict)
            state_ctx = np.concatenate([encode_state(obs.current),
                                        encode_context(obs.select.context)])
            opts = np.stack([encode_option(o, obs) for o in obs.select.option])
            picks = picks[:obs.select.maxCount]
            game_decisions.append((state_ctx, opts, picks[0], player))

            obs_dict = battle_select([int(i) for i in picks])

        result = obs_dict["current"]["result"]  # 0/1 = winner index, 2 = draw
        battle_finish()
        wins[result] += 1

        for state_ctx, opts, label, player in game_decisions:
            shard["states"].append(state_ctx)
            shard["options"].append(opts)
            shard["n_options"].append(len(opts))
            shard["labels"].append(label)
            shard["game_ids"].append(game)
            shard["results"].append(0.0 if result == 2 else (1.0 if result == player else -1.0))

        if (game + 1) % shard_size == 0:
            flush()
        if (game + 1) % log_every == 0:
            print(f"[{game + 1}/{n_games}] p0/p1/draw = {wins[0]}/{wins[1]}/{wins[2]}", flush=True)

    flush()
    print(f"done: {n_games} games -> {shard_idx} shards in {out_dir}")


# ---------------------------------------------------------------------------
# TRAINING HALF -- Piotr's part (see docs/M1-plan.md §A3 for the full breakdown)
# ---------------------------------------------------------------------------
# Suggested shape:
#
# class BCDataset(torch.utils.data.Dataset):
#     """Loads all shard_*.npz; __getitem__(i) -> (state_ctx, options[N_i, 53], label).
#     Precompute per-shard option offsets from n_options (np.cumsum) to slice the
#     flat options array. Keep game_ids around for the split."""
#
# def collate(batch):
#     """Pad options to the batch max N; return (states, options, pad_mask, labels).
#     pad_mask True where padded -- the loss must set those logits to -inf."""
#
# def train(epochs=..., lr=3e-4, batch_size=256, val_fraction=0.1):
#     """Split BY GAME_ID (not by row!). AdamW. Per epoch: masked cross-entropy on
#     labels (+ optional 0.5 * Huber(value, result)); log loss + top-1 accuracy to
#     TensorBoard; save checkpoints/bc_v0.pt when val accuracy improves.
#     Sanity ritual before the real run: overfit ~500 decisions to ~100% accuracy."""
#
# ---------------------------------------------------------------------------

class BCDataset(torch.utils.data.Dataset):
    def __init__(self):
        states, options, n_options, labels, game_ids, results, starts_list = [], [], [], [], [], [], []

        option_base = 0     # loaded rows of options
        game_base = 0       # for making game_ids globally unique

        for path in sorted(Path(ROOT / "data" / "bc").glob("*.npz")):
            print(f"loading {path.name}", flush=True)
            shard = np.load(path)

            starts = np.cumsum(shard["n_options"]) - shard["n_options"]
            starts_list.append(starts + option_base)
            game_ids.append(shard["game_ids"] + game_base)
            states.append(shard["states"])
            options.append(shard["options"])
            n_options.append(shard["n_options"])
            labels.append(shard["labels"])
            results.append(shard["results"])

            option_base += len(shard["options"])
            game_base += int(shard["game_ids"].max()) + 1

        self.states = np.concatenate(states)
        self.starts = np.concatenate(starts_list)
        self.game_ids = np.concatenate(game_ids)
        self.options = np.concatenate(options)
        self.n_options = np.concatenate(n_options)
        self.labels = np.concatenate(labels)
        self.results = np.concatenate(results)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        s = self.starts[i]
        menu = self.options[s: s + self.n_options[i]]  # slice this decision's rows out
        return self.states[i], menu, self.labels[i], self.results[i]


def collate(batch):
    B = len(batch)
    maxN = max(menu.shape[0] for _, menu, _, _ in batch)

    states = torch.zeros(B, STATE_DIM + N_CONTEXTS)
    options = torch.zeros(B, maxN, OPTION_DIM)
    valid = torch.zeros(B, maxN, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    results = torch.zeros(B, dtype=torch.float32)

    for i, (state, menu, label, result) in enumerate(batch):
        n = menu.shape[0]
        states[i] = torch.from_numpy(state)
        options[i, :n] = torch.from_numpy(menu)
        valid[i, :n] = True
        labels[i] = int(label)
        results[i] = float(result)

    return states, options, valid, labels, results

def train(epochs=10, lr=3e-4, batch_size=256, name="bc_v1"):
    ds = BCDataset()

    # Split by game
    rng = np.random.default_rng(0)
    unique_games = np.unique(ds.game_ids)
    val_games = set(rng.choice(unique_games, int(0.1 * len(unique_games)), replace=False).tolist())
    is_val = np.isin(ds.game_ids, list(val_games))
    train_ds = torch.utils.data.Subset(ds, np.nonzero(~is_val)[0])
    val_ds = torch.utils.data.Subset(ds, np.nonzero(is_val)[0])
    print(f"train {len(train_ds)}, val {len(val_ds)}")


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = OptionScorer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for states, options, valid, labels, results in train_dl:
            opt.zero_grad()

            logits, value = model(states, options)
            logits = logits.masked_fill(~valid, -1e9)
            policy_loss = F.cross_entropy(logits, labels)
            value_loss = F.huber_loss(value, results)
            loss = policy_loss + 0.5 * value_loss

            loss.backward()
            opt.step()
            running += loss.item()
        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for states, options, valid, labels, results in val_dl:
                logits, _ = model(states, options)
                logits = logits.masked_fill(~valid, -1e9)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)
        acc = correct / total
        print(f"epoch {epoch}: train_loss {running/len(train_dl):.3f}  val_acc {acc:.3f}")

        # Checkpoint the best model
        if acc > best_acc:
            best_acc = acc
            Path(ROOT / "checkpoints").mkdir(exist_ok = True)
            torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run teacher self-play and write shards")
    c.add_argument("--games", type=int, default=1000)
    c.add_argument("--shard-size", type=int, default=200)
    c.add_argument("--agent", type=str, default="lucario")
    c.add_argument("--deck", type=str, default="lucario")
    t = sub.add_parser("train", help="train the BC policy on collected shards")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--name", type=str, default="bc_v1")
    args = p.parse_args()

    if args.cmd == "collect":
        collect_games(args.games, shard_size=args.shard_size,
                      agent=args.agent, deck=args.deck)
    elif args.cmd == "train":
        train(epochs=args.epochs, name=args.name)
