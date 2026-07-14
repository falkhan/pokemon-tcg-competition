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

from .encoders import (N_CONTEXTS, OPTION_DIM, STATE_DIM, STATE_V2_DIM,
                       encode_context, encode_option, encode_option_v2,
                       encode_state, encode_state_v2)
from .policy import OptionScorerV2
from .teacher import load_teacher

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bc"
DATA_DIR_V2 = Path(__file__).resolve().parent.parent / "data" / "bc_v2"
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


def load_population(path) -> list[list[int]]:
    """data/league/population.json ({"decks": [name|csv, ...]}) -> id lists."""
    import json
    from rl.matchrunner import resolve_deck
    return [resolve_deck(d) for d in json.loads(Path(path).read_text())["decks"]]


def _teacher_pilot(teacher: str, deck_ids: list[int], instance: str):
    """Build one seat's teacher. "generic" = the M7.3 default; "solver" /
    "solver-dev" (M8.2) route through matchrunner.make_pilot so BC can clone
    the solver pilot's play — its overrides fire on ~1% of prompts, the rest
    stays the observable generic scoring (low aliasing risk; the M8.2 leg-B
    fidelity probe measures it)."""
    if teacher == "generic":
        from rl.generic_pilot import make_generic_pilot
        return make_generic_pilot(deck_ids)
    from rl.matchrunner import make_pilot
    fn, _ = make_pilot((teacher, deck_ids), instance)
    return fn


def collect_games_v2(n_games: int, decks_file, out_dir: Path = DATA_DIR_V2,
                     shard_size: int = 200, log_every: int = 25, seed: int = 0,
                     teacher: str = "generic", driver: str | None = None,
                     driver_mix: float = 1.0) -> None:
    """M7.3 collection: teacher self-play across the deck population.

    Each seat samples its OWN deck per game — both seats are the teacher (the
    generic pilot's scoring is a function of observable state, no hidden
    AttackPlan to alias), so both are recorded, and the deck diversity is what
    makes the resulting pilot deck-conditioned. Decisions are encoded with
    encoders v2. teacher: "generic" (default) | "solver" | "solver-dev"
    (M8.2 — keep different teachers in different out_dir!). Extra shard
    columns vs v1:
      state_ids  (D, N_STATE_IDS)  int32   board card ids (embedding sites)
      option_ids (sum_N, 2)        int32   acted/target card ids per option
      deck_idx   (D,)              int32   the deciding seat's population index
                                           (G5-style held-out-deck splits)

    driver (M8.6, DAgger): a checkpoint path — game ACTIONS come from this
    model pilot while LABELS stay the teacher's pick at every visited prompt,
    so training covers the states the STUDENT actually reaches (the ~12pp
    clone-vs-teacher gap is BC's compounding drift; this is the standard fix).
    driver_mix is the per-decision probability the driver acts (else the
    teacher does) — 1.0 = pure student trajectories. Keep DAgger shards in
    their own out_dir and TRAIN ON THE UNION with the teacher's shards.
    """
    import random

    population = load_population(decks_file)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    driver_cache: dict[tuple, object] = {}

    def _driver_pilot(deck_ids: list[int]):
        """One model pilot per distinct deck (torch.load once, not per game —
        the pilot is stateless so games can share it)."""
        key = tuple(deck_ids)
        if key not in driver_cache:
            from rl.matchrunner import make_pilot
            driver_cache[key] = make_pilot(("model", driver, deck_ids),
                                           f"dagger{len(driver_cache)}")[0]
        return driver_cache[key]

    columns = ("states", "state_ids", "options", "option_ids",
               "n_options", "labels", "game_ids", "results", "deck_idx")
    shard: dict[str, list] = {k: [] for k in columns}
    shard_idx = sum(1 for _ in out_dir.glob("shard_*.npz"))
    wins = [0, 0, 0]

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            state_ids=np.stack(shard["state_ids"]),
            options=np.concatenate(shard["options"]),
            option_ids=np.concatenate(shard["option_ids"]),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            labels=np.array(shard["labels"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    for game in range(n_games):
        picks_idx = [rng.randrange(len(population)) for _ in range(2)]
        decks = [population[picks_idx[0]], population[picks_idx[1]]]
        pilots = [_teacher_pilot(teacher, d, f"bc{game}_{seat}")
                  for seat, d in enumerate(decks)]

        obs_dict, start_data = battle_start(decks[0], decks[1])
        if start_data.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected a deck (errorType={start_data.errorType})")

        game_decisions: list[tuple] = []
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            picks = pilots[player](obs_dict)

            obs = to_observation_class(obs_dict)
            state_num, state_ids = encode_state_v2(obs.current, decks[player])
            state_ctx = np.concatenate([state_num, encode_context(obs.select.context)])
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([num for num, _ in pairs])
            opt_ids = np.stack([ids for _, ids in pairs])
            picks = picks[:obs.select.maxCount]
            game_decisions.append((state_ctx, state_ids, opts, opt_ids,
                                   picks[0], player))
            exec_picks = picks                      # teacher acts (pre-M8.6 path)
            if driver is not None and rng.random() < driver_mix:
                exec_picks = _driver_pilot(decks[player])(obs_dict)
                exec_picks = exec_picks[:obs.select.maxCount]
            obs_dict = battle_select([int(i) for i in exec_picks])

        result = obs_dict["current"]["result"]
        battle_finish()
        wins[result] += 1

        for state_ctx, state_ids, opts, opt_ids, label, player in game_decisions:
            shard["states"].append(state_ctx)
            shard["state_ids"].append(state_ids)
            shard["options"].append(opts)
            shard["option_ids"].append(opt_ids)
            shard["n_options"].append(len(opts))
            shard["labels"].append(label)
            shard["game_ids"].append(game)
            shard["results"].append(0.0 if result == 2 else (1.0 if result == player else -1.0))
            shard["deck_idx"].append(picks_idx[player])

        if (game + 1) % shard_size == 0:
            flush()
        if (game + 1) % log_every == 0:
            print(f"[{game + 1}/{n_games}] p0/p1/draw = {wins[0]}/{wins[1]}/{wins[2]}",
                  flush=True)

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

class BCDatasetV2(torch.utils.data.Dataset):
    """v2 shards: v1 fields + parallel state_ids/option_ids + deck_idx.

    data_dir may be one directory or a list of them (M8.6: DAgger trains on
    the UNION of the teacher's shards and the student-visited correction
    shards — game ids are re-based across shards, so mixing dirs is safe)."""

    def __init__(self, data_dir: Path | str | list = DATA_DIR_V2):
        dirs = ([data_dir] if isinstance(data_dir, (str, Path))
                else list(data_dir))
        cols = {k: [] for k in ("states", "state_ids", "options", "option_ids",
                                "n_options", "labels", "game_ids", "results",
                                "deck_idx")}
        starts_list = []
        option_base = 0
        game_base = 0
        for path in [p for d in dirs for p in sorted(Path(d).glob("*.npz"))]:
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
        s = self.starts[i]
        n = self.n_options[i]
        return (self.states[i], self.state_ids[i], self.options[s:s + n],
                self.option_ids[s:s + n], self.labels[i], self.results[i],
                self.deck_idx[i])


def collate_v2(batch):
    B = len(batch)
    maxN = max(menu.shape[0] for _, _, menu, _, _, _, _ in batch)

    states = torch.zeros(B, STATE_V2_DIM + N_CONTEXTS)
    state_ids = torch.zeros(B, batch[0][1].shape[0], dtype=torch.long)
    options = torch.zeros(B, maxN, OPTION_DIM)
    option_ids = torch.zeros(B, maxN, 2, dtype=torch.long)
    valid = torch.zeros(B, maxN, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    results = torch.zeros(B, dtype=torch.float32)
    deck_idx = torch.zeros(B, dtype=torch.long)

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


def train_v2(epochs=10, lr=3e-4, batch_size=256, name="osv2_bc",
             data_dir: Path = DATA_DIR_V2):
    """Train OptionScorerV2 on generic-teacher v2 shards. Reports overall AND
    per-deck val accuracy (the G5 deck-conditioning signal). BC should MATCH
    the teacher, not beat it — the M7.3 gate is fidelity >= 0.80 val top-1."""
    ds = BCDatasetV2(data_dir)

    rng = np.random.default_rng(0)
    unique_games = np.unique(ds.game_ids)
    val_games = set(rng.choice(unique_games, max(1, int(0.1 * len(unique_games))),
                               replace=False).tolist())
    is_val = np.isin(ds.game_ids, list(val_games))
    train_ds = torch.utils.data.Subset(ds, np.nonzero(~is_val)[0])
    val_ds = torch.utils.data.Subset(ds, np.nonzero(is_val)[0])
    print(f"train {len(train_ds)}, val {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_v2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_v2)

    model = OptionScorerV2()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for states, sids, options, oids, valid, labels, results, _ in train_dl:
            opt.zero_grad()
            logits, value = model(states, sids, options, oids)
            logits = logits.masked_fill(~valid, -1e9)
            loss = F.cross_entropy(logits, labels) + 0.5 * F.huber_loss(value, results)
            loss.backward()
            opt.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        per_deck: dict[int, list[int]] = {}
        with torch.no_grad():
            for states, sids, options, oids, valid, labels, results, didx in val_dl:
                logits, _ = model(states, sids, options, oids)
                logits = logits.masked_fill(~valid, -1e9)
                hit = (logits.argmax(dim=1) == labels)
                correct += hit.sum().item()
                total += len(labels)
                for d, h in zip(didx.tolist(), hit.tolist()):
                    per_deck.setdefault(d, [0, 0])
                    per_deck[d][0] += h
                    per_deck[d][1] += 1
        acc = correct / max(1, total)
        by_deck = " ".join(f"d{d}:{c / n:.2f}" for d, (c, n) in sorted(per_deck.items()))
        print(f"epoch {epoch}: train_loss {running / len(train_dl):.3f}  "
              f"val_acc {acc:.3f}  [{by_deck}]")

        if acc > best_acc:
            best_acc = acc
            Path(ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")


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
    c.add_argument("--teacher", type=str, default="rule",
                   choices=["rule", "generic", "solver", "solver-dev"],
                   help="generic/solver/solver-dev = deck-population collection "
                        "(encoders v2); solver* clones the search pilot (M8.2)")
    c.add_argument("--agent", type=str, default="lucario")
    c.add_argument("--deck", type=str, default="lucario")
    c.add_argument("--decks", type=str, default="data/league/population.json",
                   help="deck population file (v2 teachers only)")
    c.add_argument("--out", type=str, default=None,
                   help="shard output dir (default data/bc_v2; keep different "
                        "teachers in different dirs)")
    c.add_argument("--driver", type=str, default=None, metavar="CKPT",
                   help="DAgger (M8.6): this model checkpoint PLAYS the games "
                        "while labels stay the teacher's picks — collects "
                        "teacher corrections on student-visited states "
                        "(v2 teachers only; use a dedicated --out)")
    c.add_argument("--driver-mix", type=float, default=1.0,
                   help="per-decision probability the --driver acts instead "
                        "of the teacher (default 1.0 = pure student play)")
    t = sub.add_parser("train", help="train the BC policy on collected shards")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--name", type=str, default=None)
    t.add_argument("--arch", type=str, default="v1", choices=["v1", "v2"])
    t.add_argument("--data", type=str, default=None,
                   help="shard dir for --arch v2 (default data/bc_v2); "
                        "comma-separate to train on a union, e.g. teacher "
                        "shards + DAgger correction shards (M8.6)")
    args = p.parse_args()

    if args.cmd == "collect":
        if args.teacher in ("generic", "solver", "solver-dev"):
            collect_games_v2(args.games, args.decks, shard_size=args.shard_size,
                             teacher=args.teacher, driver=args.driver,
                             driver_mix=args.driver_mix,
                             out_dir=Path(args.out) if args.out else DATA_DIR_V2)
        else:
            if args.driver:
                raise SystemExit("--driver needs a v2 teacher (generic/solver*)")
            collect_games(args.games, shard_size=args.shard_size,
                          agent=args.agent, deck=args.deck)
    elif args.cmd == "train":
        if args.arch == "v2":
            train_v2(epochs=args.epochs, name=args.name or "osv2_bc",
                     data_dir=([Path(x) for x in args.data.split(",")]
                               if args.data else DATA_DIR_V2))
        else:
            train(epochs=args.epochs, name=args.name or "bc_v1")
