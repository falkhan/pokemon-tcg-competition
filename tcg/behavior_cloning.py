"""Behavior cloning from the rule-based Mega Lucario teacher (docs/M1-plan.md).

Same pipeline as the old ``rl/bc.py``, in two halves:

  COLLECTION  collect_games() — teacher self-play via the direct engine loop,
              decisions encoded immediately and saved as sharded .npz files.
  TRAINING    BCDataset / collate / train() — masked cross-entropy on the
              teacher's picks (+ a Huber value loss on game outcomes).

Collect:  python -m tcg.behavior_cloning collect --games 1000
Train:    python -m tcg.behavior_cloning train --epochs 10
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

from tcg.decks import ROOT, load_deck
from tcg.encoders import (N_CONTEXTS, OPTION_DIM, STATE_DIM, STATE_V2_DIM,
                          encode_context, encode_option, encode_option_v2,
                          encode_state, encode_state_v2)
from tcg.network import MASKED_LOGIT, OptionScorer, OptionScorerV2
from tcg.selfplay import write_shard
from tcg.teachers import load_teacher

BC_DATA_DIR = ROOT / "data" / "bc"
BC_DATA_DIR_V2 = ROOT / "data" / "bc_v2"

BC_INT32_COLUMNS = frozenset({"n_options", "labels", "game_ids"})
BC_FLOAT32_COLUMNS = frozenset({"results"})
BC_V2_INT32_COLUMNS = frozenset({"n_options", "labels", "game_ids", "deck_idx"})

VALUE_LOSS_WEIGHT = 0.5   # policy cross-entropy carries the training; value rides along
VAL_GAME_FRACTION = 0.1   # validation split, BY GAME (rows of one game correlate)


def collect_games(n_games: int, out_dir: Path = BC_DATA_DIR, shard_size: int = 200,
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
    deck_ids = load_deck(deck)
    # Two ISOLATED teacher instances -- they hold per-player mutable globals.
    teachers = [load_teacher("p0", agent=agent, deck=deck),
                load_teacher("p1", agent=agent, deck=deck)]

    shard: dict[str, list] = {name: [] for name in
                              ("states", "options", "n_options", "labels",
                               "game_ids", "results")}
    shard_idx = sum(1 for _ in out_dir.glob("shard_*.npz"))  # append after existing shards
    wins = [0, 0, 0]

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        write_shard(out_dir / f"shard_{shard_idx:04d}.npz", shard,
                    BC_INT32_COLUMNS, BC_FLOAT32_COLUMNS)
        shard_idx += 1
        for values in shard.values():
            values.clear()

    for game in range(n_games):
        obs_dict, start_data = battle_start(deck_ids, deck_ids)
        if start_data.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected the deck (errorType={start_data.errorType})")

        game_decisions: list[tuple] = []  # (state_ctx, options, label, player) until result known
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            picks = teachers[player](obs_dict)

            observation = to_observation_class(obs_dict)
            state_ctx = np.concatenate([encode_state(observation.current),
                                        encode_context(observation.select.context)])
            option_vectors = np.stack([encode_option(option, observation)
                                       for option in observation.select.option])
            picks = picks[:observation.select.maxCount]
            game_decisions.append((state_ctx, option_vectors, picks[0], player))

            obs_dict = battle_select([int(i) for i in picks])

        result = obs_dict["current"]["result"]  # 0/1 = winner index, 2 = draw
        battle_finish()
        wins[result] += 1

        for state_ctx, option_vectors, label, player in game_decisions:
            shard["states"].append(state_ctx)
            shard["options"].append(option_vectors)
            shard["n_options"].append(len(option_vectors))
            shard["labels"].append(label)
            shard["game_ids"].append(game)
            shard["results"].append(0.0 if result == 2 else (1.0 if result == player else -1.0))

        if (game + 1) % shard_size == 0:
            flush()
        if (game + 1) % log_every == 0:
            print(f"[{game + 1}/{n_games}] p0/p1/draw = {wins[0]}/{wins[1]}/{wins[2]}",
                  flush=True)

    flush()
    print(f"done: {n_games} games -> {shard_idx} shards in {out_dir}")


def load_population(path) -> list[list[int]]:
    """data/league/population.json ({"decks": [name|csv, ...]}) -> id lists."""
    import json
    from rl.matchrunner import resolve_deck
    return [resolve_deck(d) for d in json.loads(Path(path).read_text())["decks"]]


def collect_games_v2(n_games: int, decks_file, out_dir: Path = BC_DATA_DIR_V2,
                     shard_size: int = 200, log_every: int = 25, seed: int = 0) -> None:
    """M7.3 collection: GENERIC-pilot self-play across the deck population.

    Each seat samples its OWN deck per game — both seats are the teacher (the
    generic pilot's scoring is a function of observable state, no hidden
    AttackPlan to alias), so both are recorded, and the deck diversity is what
    makes the resulting pilot deck-conditioned. Decisions are encoded with
    encoders v2. Extra shard columns vs v1:
      state_ids  (D, N_STATE_IDS)  int32   board card ids (embedding sites)
      option_ids (sum_N, 2)        int32   acted/target card ids per option
      deck_idx   (D,)              int32   the deciding seat's population index
                                           (G5-style held-out-deck splits)
    """
    import random

    from tcg.pilot import make_generic_pilot

    population = load_population(decks_file)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard: dict[str, list] = {name: [] for name in
                              ("states", "state_ids", "options", "option_ids",
                               "n_options", "labels", "game_ids", "results",
                               "deck_idx")}
    shard_idx = sum(1 for _ in out_dir.glob("shard_*.npz"))
    wins = [0, 0, 0]

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        write_shard(out_dir / f"shard_{shard_idx:04d}.npz", shard,
                    BC_V2_INT32_COLUMNS, BC_FLOAT32_COLUMNS)
        shard_idx += 1
        for values in shard.values():
            values.clear()

    for game in range(n_games):
        deck_indices = [rng.randrange(len(population)) for _ in range(2)]
        decks = [population[deck_indices[0]], population[deck_indices[1]]]
        pilots = [make_generic_pilot(d) for d in decks]

        obs_dict, start_data = battle_start(decks[0], decks[1])
        if start_data.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected a deck (errorType={start_data.errorType})")

        game_decisions: list[tuple] = []
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            picks = pilots[player](obs_dict)

            observation = to_observation_class(obs_dict)
            state_num, state_ids = encode_state_v2(observation.current, decks[player])
            state_ctx = np.concatenate([state_num,
                                        encode_context(observation.select.context)])
            pairs = [encode_option_v2(option, observation)
                     for option in observation.select.option]
            option_vectors = np.stack([numeric for numeric, _ in pairs])
            option_ids = np.stack([ids for _, ids in pairs])
            picks = picks[:observation.select.maxCount]
            game_decisions.append((state_ctx, state_ids, option_vectors, option_ids,
                                   picks[0], player))
            obs_dict = battle_select([int(i) for i in picks])

        result = obs_dict["current"]["result"]
        battle_finish()
        wins[result] += 1

        for state_ctx, state_ids, option_vectors, option_ids, label, player \
                in game_decisions:
            shard["states"].append(state_ctx)
            shard["state_ids"].append(state_ids)
            shard["options"].append(option_vectors)
            shard["option_ids"].append(option_ids)
            shard["n_options"].append(len(option_vectors))
            shard["labels"].append(label)
            shard["game_ids"].append(game)
            shard["results"].append(0.0 if result == 2 else (1.0 if result == player else -1.0))
            shard["deck_idx"].append(deck_indices[player])

        if (game + 1) % shard_size == 0:
            flush()
        if (game + 1) % log_every == 0:
            print(f"[{game + 1}/{n_games}] p0/p1/draw = {wins[0]}/{wins[1]}/{wins[2]}",
                  flush=True)

    flush()
    print(f"done: {n_games} games -> {shard_idx} shards in {out_dir}")


class BCDataset(torch.utils.data.Dataset):
    """All shard_*.npz concatenated; __getitem__(i) -> (state_ctx, menu, label, result).

    Per-shard option offsets are precomputed from n_options (np.cumsum) to slice
    the flat options array; game_ids are remapped to stay globally unique across
    shards, for the by-game train/val split.
    """

    def __init__(self, data_dir: Path = BC_DATA_DIR):
        states, options, n_options, labels = [], [], [], []
        game_ids, results, starts_list = [], [], []

        option_base = 0     # loaded rows of options
        game_base = 0       # for making game_ids globally unique

        for path in sorted(Path(data_dir).glob("*.npz")):
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
        start = self.starts[i]
        menu = self.options[start: start + self.n_options[i]]  # this decision's rows
        return self.states[i], menu, self.labels[i], self.results[i]


def collate(batch):
    """Pad options to the batch max N; the loss masks padded logits via `valid`."""
    batch_size = len(batch)
    max_options = max(menu.shape[0] for _, menu, _, _ in batch)

    states = torch.zeros(batch_size, STATE_DIM + N_CONTEXTS)
    options = torch.zeros(batch_size, max_options, OPTION_DIM)
    valid = torch.zeros(batch_size, max_options, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    results = torch.zeros(batch_size, dtype=torch.float32)

    for i, (state, menu, label, result) in enumerate(batch):
        n = menu.shape[0]
        states[i] = torch.from_numpy(state)
        options[i, :n] = torch.from_numpy(menu)
        valid[i, :n] = True
        labels[i] = int(label)
        results[i] = float(result)

    return states, options, valid, labels, results


class BCDatasetV2(torch.utils.data.Dataset):
    """v2 shards: v1 fields + parallel state_ids/option_ids + deck_idx."""

    def __init__(self, data_dir: Path = BC_DATA_DIR_V2):
        cols = {k: [] for k in ("states", "state_ids", "options", "option_ids",
                                "n_options", "labels", "game_ids", "results",
                                "deck_idx")}
        starts_list = []
        option_base = 0
        game_base = 0
        for path in sorted(Path(data_dir).glob("*.npz")):
            print(f"loading {path.name}", flush=True)
            shard = np.load(path)
            starts = np.cumsum(shard["n_options"]) - shard["n_options"]
            starts_list.append(starts + option_base)
            cols["game_ids"].append(shard["game_ids"] + game_base)
            for k in cols:
                if k != "game_ids":
                    cols[k].append(shard[k])
            option_base += len(shard["options"])
            game_base += int(shard["game_ids"].max()) + 1
        for k, v in cols.items():
            setattr(self, k, np.concatenate(v))
        self.starts = np.concatenate(starts_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        start = self.starts[i]
        n = self.n_options[i]
        return (self.states[i], self.state_ids[i], self.options[start:start + n],
                self.option_ids[start:start + n], self.labels[i], self.results[i],
                self.deck_idx[i])


def collate_v2(batch):
    """collate() with the parallel id arrays padded alongside the menus."""
    batch_size = len(batch)
    max_options = max(menu.shape[0] for _, _, menu, _, _, _, _ in batch)

    states = torch.zeros(batch_size, STATE_V2_DIM + N_CONTEXTS)
    state_ids = torch.zeros(batch_size, batch[0][1].shape[0], dtype=torch.long)
    options = torch.zeros(batch_size, max_options, OPTION_DIM)
    option_ids = torch.zeros(batch_size, max_options, 2, dtype=torch.long)
    valid = torch.zeros(batch_size, max_options, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    results = torch.zeros(batch_size, dtype=torch.float32)
    deck_idx = torch.zeros(batch_size, dtype=torch.long)

    for i, (state, sids, menu, oids, label, result, didx) in enumerate(batch):
        n = menu.shape[0]
        states[i] = torch.from_numpy(state)
        state_ids[i] = torch.from_numpy(sids.astype(np.int64))
        options[i, :n] = torch.from_numpy(menu)
        option_ids[i, :n] = torch.from_numpy(oids.astype(np.int64))
        valid[i, :n] = True
        labels[i] = int(label)
        results[i] = float(result)
        deck_idx[i] = int(didx)

    return states, state_ids, options, option_ids, valid, labels, results, deck_idx


def train_v2(epochs: int = 10, lr: float = 3e-4, batch_size: int = 256,
             name: str = "osv2_bc", data_dir: Path = BC_DATA_DIR_V2) -> None:
    """Train OptionScorerV2 on generic-teacher v2 shards. Reports overall AND
    per-deck val accuracy (the G5 deck-conditioning signal). BC should MATCH
    the teacher, not beat it — the M7.3 gate is fidelity >= 0.80 val top-1."""
    dataset = BCDatasetV2(data_dir)

    rng = np.random.default_rng(0)
    unique_games = np.unique(dataset.game_ids)
    val_games = set(rng.choice(unique_games,
                               max(1, int(VAL_GAME_FRACTION * len(unique_games))),
                               replace=False).tolist())
    is_val = np.isin(dataset.game_ids, list(val_games))
    train_dataset = torch.utils.data.Subset(dataset, np.nonzero(~is_val)[0])
    val_dataset = torch.utils.data.Subset(dataset, np.nonzero(is_val)[0])
    print(f"train {len(train_dataset)}, val {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_v2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_v2)

    model = OptionScorerV2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for states, sids, options, oids, valid, labels, results, _ in train_loader:
            optimizer.zero_grad()
            logits, value = model(states, sids, options, oids)
            logits = logits.masked_fill(~valid, MASKED_LOGIT)
            loss = (F.cross_entropy(logits, labels)
                    + VALUE_LOSS_WEIGHT * F.huber_loss(value, results))
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        per_deck: dict[int, list[int]] = {}
        with torch.no_grad():
            for states, sids, options, oids, valid, labels, results, didx in val_loader:
                logits, _ = model(states, sids, options, oids)
                logits = logits.masked_fill(~valid, MASKED_LOGIT)
                hit = (logits.argmax(dim=1) == labels)
                correct += hit.sum().item()
                total += len(labels)
                for d, h in zip(didx.tolist(), hit.tolist()):
                    per_deck.setdefault(d, [0, 0])
                    per_deck[d][0] += h
                    per_deck[d][1] += 1
        accuracy = correct / max(1, total)
        by_deck = " ".join(f"d{d}:{c / n:.2f}" for d, (c, n) in sorted(per_deck.items()))
        print(f"epoch {epoch}: train_loss {running / len(train_loader):.3f}  "
              f"val_acc {accuracy:.3f}  [{by_deck}]")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Path(ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")


def train(epochs: int = 10, lr: float = 3e-4, batch_size: int = 256,
          name: str = "bc_v1") -> None:
    """Masked cross-entropy on the teacher's picks; checkpoint on best val accuracy.

    Sanity ritual before a real run: overfit ~500 decisions to ~100% accuracy.
    """
    dataset = BCDataset()

    # Split BY GAME (not by row!) — rows within a game are heavily correlated.
    rng = np.random.default_rng(0)
    unique_games = np.unique(dataset.game_ids)
    val_games = set(rng.choice(unique_games,
                               int(VAL_GAME_FRACTION * len(unique_games)),
                               replace=False).tolist())
    is_val = np.isin(dataset.game_ids, list(val_games))
    train_dataset = torch.utils.data.Subset(dataset, np.nonzero(~is_val)[0])
    val_dataset = torch.utils.data.Subset(dataset, np.nonzero(is_val)[0])
    print(f"train {len(train_dataset)}, val {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate)

    model = OptionScorer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for states, options, valid, labels, results in train_loader:
            optimizer.zero_grad()

            logits, value = model(states, options)
            logits = logits.masked_fill(~valid, MASKED_LOGIT)
            policy_loss = F.cross_entropy(logits, labels)
            value_loss = F.huber_loss(value, results)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            loss.backward()
            optimizer.step()
            running += loss.item()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for states, options, valid, labels, results in val_loader:
                logits, _ = model(states, options)
                logits = logits.masked_fill(~valid, MASKED_LOGIT)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)
        accuracy = correct / total
        print(f"epoch {epoch}: train_loss {running / len(train_loader):.3f}  "
              f"val_acc {accuracy:.3f}")

        # Checkpoint the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Path(ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect_parser = sub.add_parser("collect", help="run teacher self-play and write shards")
    collect_parser.add_argument("--games", type=int, default=1000)
    collect_parser.add_argument("--shard-size", type=int, default=200)
    collect_parser.add_argument("--teacher", type=str, default="rule",
                                choices=["rule", "generic"],
                                help="generic = M7.3 deck-population collection")
    collect_parser.add_argument("--agent", type=str, default="lucario")
    collect_parser.add_argument("--deck", type=str, default="lucario")
    collect_parser.add_argument("--decks", type=str,
                                default="data/league/population.json",
                                help="deck population file (generic teacher only)")
    train_parser = sub.add_parser("train", help="train the BC policy on collected shards")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--name", type=str, default=None)
    train_parser.add_argument("--arch", type=str, default="v1", choices=["v1", "v2"])
    args = parser.parse_args()

    if args.cmd == "collect":
        if args.teacher == "generic":
            collect_games_v2(args.games, args.decks, shard_size=args.shard_size)
        else:
            collect_games(args.games, shard_size=args.shard_size,
                          agent=args.agent, deck=args.deck)
    elif args.cmd == "train":
        if args.arch == "v2":
            train_v2(epochs=args.epochs, name=args.name or "osv2_bc")
        else:
            train(epochs=args.epochs, name=args.name or "bc_v1")
