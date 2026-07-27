"""Supervised value-head training on self-play outcomes (docs/DECISIONS.md 2026-07-08).

Why this exists: PPO could not train a useful critic on this sparse-reward, mirror-heavy
problem (3 attempts stalled or degraded the policy). But the *information* to predict the
winner is in the encoder (opponent archetype is observable). Training the value head
DIRECTLY — supervised regression to game outcomes, policy frozen — works: bc_v1's original
value head scored 0.62 sign-accuracy (≈ chance); a supervised head hits ~0.87 on the same
frozen features. This decouples "get a good critic" from "don't wreck the policy," and it's
the value-target step of an AlphaGo-Zero-lite loop (self-play -> outcomes -> value head ->
MCTS).

  python -m rl.value_train collect --checkpoint checkpoints/bc_v1.pt --games 600
  python -m rl.value_train train   --checkpoint checkpoints/bc_v1.pt --out bc_v1_value.pt
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "value_train.npz"
DECK_DIR = ROOT / "decks"


def _deck(name: str) -> list[int]:
    return [int(x) for x in (DECK_DIR / f"{name}.csv").read_text().split() if x.strip()]


def _search_visited(obs, my_deck, meta, rng, model_pick, learn_seat,
                    encode_fn, plies: int = 4, samples: int = 2) -> list[tuple]:
    """M8.4(b): the states SEARCH will visit from this decision — K determinized
    rollouts of a few plies on the engine forward model, opponent hidden zones
    from the L3 archetype determinizer (rl/determinize.py, the M8.4a
    instrument). Returns (state_ctx, perspective_sign) pairs: a visited state
    encodes from ITS yourIndex's view, so the outcome label must flip when the
    visited mover is the opponent (the mcts negamax convention).

    This is the documented-but-never-executed fix for the measured MCTS
    failure: the value head was OOD on determinized/deep search states
    (DECISIONS.md 2026-07-08) — so train it on exactly those states."""
    import random as _random

    from cg.api import search_begin, search_end, search_step
    from rl.determinize import determinize_kwargs

    st = obs.current
    mine = st.players[st.yourIndex]
    visited = []
    for _ in range(samples):
        kw = determinize_kwargs(obs, meta, rng)
        try:
            state = search_begin(
                obs, your_deck=rng.sample(my_deck, mine.deckCount),
                your_prize=rng.sample(my_deck, len(mine.prize)), **kw)
            for _ in range(plies):
                o = state.observation
                if o.current.result >= 0 or o.select is None:
                    break
                if rng.random() < 0.5 and o.current.yourIndex == learn_seat:
                    action = model_pick(o)          # the policy's own line
                else:
                    action = _random.Random(rng.random()).sample(
                        range(len(o.select.option)), o.select.maxCount)
                state = search_step(state.searchId, [int(i) for i in action])
                o = state.observation
                if o.current.result >= 0 or o.select is None:
                    break
                sign = 1.0 if o.current.yourIndex == learn_seat else -1.0
                visited.append((encode_fn(o), sign))
        finally:
            search_end()
    return visited


def collect(checkpoint: str, learn_deck: str = "kyogre", n_games: int = 600,
            out: Path = DATA, search_plies: int = 0, search_samples: int = 2,
            meta_version: str = "meta_v1", seed: int = 71) -> None:
    """Play the policy (greedy) vs the opponent pool; record (state_ctx, final outcome)
    for the learner seat. Outcome is +1 win / -1 loss / 0 draw from the learner's view.

    search_plies > 0 (M8.4b): every learner decision ALSO records
    `search_samples` determinized rollouts of that many plies (L3 opponent
    determinization), outcome-labeled with the perspective sign — the
    value-on-search-states dataset."""
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.encoders import (COMBAT_SLICE, N_COMBAT, N_CONTEXTS, STATE_DIM,
                             encode_state, encode_context, encode_option)
    from rl.policy import OptionScorer
    from rl.teacher import load_teacher
    import random

    sd = torch.load(checkpoint, map_location="cpu")
    is_v3 = "plan_enc.0.weight" in sd          # M28 Track E: v3 lineage
    ld = _deck(learn_deck)
    state_ids: list = []
    if is_v3:
        # M28: the shipped lineage is OptionScorerV3. Encoding differs in three
        # ways from v1 — the state carries id vectors for learnable embeddings,
        # the option encoder is encode_option_v2, and the trunk takes a plan
        # (fed as ZEROS: docs/M27.md Probe 4 measured the plan block inert here).
        from rl.encoders import encode_option_v2, encode_state_v3
        from rl.plan import PLAN_DIM
        from rl.plan_iter import _n_ids_of
        from rl.policy import OptionScorerV3, option_dim_of
        model = OptionScorerV3(n_state_ids=_n_ids_of(sd),
                               option_dim=option_dim_of(sd))
        model.load_state_dict(sd)
        model.eval()
        cut = None
    else:
        # Dimension-aware load (the matchrunner shim): bc_v1 predates the M3
        # combat features — slice the combat block out of current encodings.
        in_dim = sd["state_enc.0.weight"].shape[1]
        expected = STATE_DIM + N_CONTEXTS
        if in_dim == expected:
            cut = None
        elif in_dim == expected - N_COMBAT:
            cut = COMBAT_SLICE
        else:
            raise ValueError(f"{checkpoint}: state dim {in_dim} matches neither "
                             f"{expected} nor the pre-M3 {expected - N_COMBAT}")
        model = OptionScorer(state_ctx_dim=in_dim)
        model.load_state_dict(sd)
        model.eval()
    opps = [(None, ld),
            (load_teacher("vpl", agent="lucario", deck="lucario"), _deck("lucario")),
            (load_teacher("vpi", agent="iono", deck="iono"), _deck("iono"))]
    meta = None
    rng = random.Random(seed)
    if search_plies > 0:
        from rl.determinize import load_meta
        meta = load_meta(meta_version)

    def _sc(obs):
        """(state_ctx, state_ids) — ids are None on the v1 path."""
        if is_v3:
            num, sids = encode_state_v3(obs.current, ld)
            return (np.concatenate([num, encode_context(obs.select.context)]
                                   ).astype(np.float32), sids)
        sc = np.concatenate([encode_state(obs.current),
                             encode_context(obs.select.context)]).astype(np.float32)
        return (np.delete(sc, np.s_[cut[0]:cut[1]]) if cut is not None else sc), None

    def pick(obs):
        sc, sids = _sc(obs)
        if is_v3:
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([n for n, _ in pairs]).astype(np.float32)
            oids = np.stack([i for _, i in pairs])
            acts = model.act(sc, np.zeros(PLAN_DIM, np.float32), sids, opts, oids,
                             obs.select.maxCount, greedy=True)
        else:
            opts = np.stack([encode_option(o, obs)
                             for o in obs.select.option]).astype(np.float32)
            acts = model.act(sc, opts, k=obs.select.maxCount, greedy=True)
        return (sc, sids), [int(i) for i in acts]

    states, outcomes, game_ids = [], [], []
    for g in range(n_games):
        oa, od = opps[g % 3]
        learn_seat = g % 2
        d0, d1 = (ld, od) if learn_seat == 0 else (od, ld)
        obs_dict, _ = battle_start(d0, d1)
        pend = []                                # (state_ctx, perspective_sign)
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            obs = to_observation_class(obs_dict)
            if seat == learn_seat:
                scp, picks = pick(obs); pend.append((scp, 1.0))
                if search_plies > 0 and getattr(obs, "search_begin_input", None) is not None:
                    pend += _search_visited(obs, ld, meta, rng,
                                            lambda o: pick(o)[1], learn_seat,
                                            _sc, plies=search_plies,
                                            samples=search_samples)
                obs_dict = battle_select(picks)
            else:
                p = random.sample(range(len(obs_dict["select"]["option"])), obs_dict["select"]["maxCount"]) \
                    if oa is None else oa(obs_dict)
                obs_dict = battle_select([int(i) for i in p])
        res = obs_dict["current"]["result"]; battle_finish()
        y = 0.0 if res == 2 else (1.0 if res == learn_seat else -1.0)
        states += [sc for (sc, _), _ in pend]
        state_ids += [sids for (_, sids), _ in pend]
        outcomes += [y * sign for _, sign in pend]
        game_ids += [g] * len(pend)

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {"states": np.stack(states).astype(np.float32),
            "outcomes": np.array(outcomes, dtype=np.float32),
            # M28: every state of a game shares that game's outcome label, so a
            # per-STATE split leaks the label across the split and inflates
            # sign-accuracy. train() splits on this instead.
            "game_ids": np.array(game_ids, dtype=np.int64)}
    if is_v3:                       # M28: v3 trunks need the embedding ids
        cols["state_ids"] = np.stack(state_ids).astype(np.int64)
    np.savez_compressed(out, **cols)
    print(f"collected {len(states)} states over {len(set(game_ids))} games -> {out}"
          f"{' (v3, with state_ids)' if is_v3 else ''}")


def train(checkpoint: str, out: str = "bc_v1_value.pt", data: Path = DATA,
          epochs: int = 40, lr: float = 1e-3) -> None:
    """Freeze policy+body; train ONLY the value head to regress outcomes. Saves a
    checkpoint = original policy/body + the new value head."""
    import torch
    import torch.nn.functional as F
    from rl.policy import OptionScorer

    d = np.load(data)
    S = torch.from_numpy(d["states"]); Y = torch.from_numpy(d["outcomes"])
    # Split by GAME, not by state (M28): every state of a game carries that
    # game's outcome as its label, so a per-state split puts near-duplicate
    # rows with the SAME label on both sides and inflates sign-accuracy.
    if "game_ids" in d:
        games = np.unique(d["game_ids"])
        rng = np.random.default_rng(0)
        val_games = set(rng.choice(games, max(1, int(0.15 * len(games))),
                                   replace=False).tolist())
        is_val = np.isin(d["game_ids"], list(val_games))
        tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
        split = f"{len(games)} games ({len(val_games)} held out)"
    else:
        n = len(S); perm = np.random.default_rng(0).permutation(n)
        tr, va = perm[:int(0.85 * n)], perm[int(0.85 * n):]
        split = "per-STATE split (legacy npz, no game_ids — LEAKY)"
    print(f"value data: {len(S)} states, split by {split}")

    sd = torch.load(ROOT / "checkpoints" / checkpoint, map_location="cpu")
    is_v3 = "plan_enc.0.weight" in sd
    if is_v3:
        from rl.plan import PLAN_DIM
        from rl.plan_iter import _n_ids_of
        from rl.policy import OptionScorerV3, option_dim_of
        if "state_ids" not in d:
            raise ValueError(f"{data} has no state_ids — recollect with the v3 "
                             "checkpoint (M28 Track E)")
        IDS = torch.from_numpy(d["state_ids"]).long()
        model = OptionScorerV3(n_state_ids=_n_ids_of(sd),
                               option_dim=option_dim_of(sd))
    else:
        IDS = None
        model = OptionScorer(state_ctx_dim=S.shape[1])
    model.load_state_dict(sd)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("value_head")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def feats_of(idx):
        """Frozen trunk features. v3's trunk also consumes the plan (ZEROS —
        Probe 4 measured the plan block inert here) and the id embeddings."""
        if is_v3:
            return model._trunk(S[idx], S.new_zeros(len(idx), PLAN_DIM), IDS[idx])
        return model.state_enc(S[idx])

    def sign_acc(idx):
        with torch.no_grad():
            v = model.value_head(feats_of(idx)).squeeze(-1)
        # drawn games carry outcome 0 and have no sign to predict
        keep = Y[idx] != 0
        if not keep.any():
            return float("nan")
        return ((v[keep] > 0).float() == (Y[idx][keep] > 0).float()).float().mean().item()

    base = sign_acc(va)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(tr), 512):
            b = tr[i:i + 512]
            with torch.no_grad():
                f = feats_of(b)
            loss = F.mse_loss(model.value_head(f).squeeze(-1), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    torch.save(model.state_dict(), ROOT / "checkpoints" / out)
    print(f"value-head sign-acc {base:.3f} -> {sign_acc(va):.3f}  "
          f"(n_val={len(va)}, {'v3' if is_v3 else 'v1'})  saved checkpoints/{out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--checkpoint", default="checkpoints/bc_v1.pt")
    c.add_argument("--deck", default="kyogre"); c.add_argument("--games", type=int, default=600)
    c.add_argument("--out", default=None, help="npz path (default data/value_train.npz)")
    c.add_argument("--search-plies", type=int, default=0,
                   help="M8.4b: also record K-ply determinized search rollouts "
                        "per learner decision (L3 opponent determinization)")
    c.add_argument("--search-samples", type=int, default=2)
    c.add_argument("--meta", default="meta_v1")
    t = sub.add_parser("train"); t.add_argument("--checkpoint", default="bc_v1.pt")
    t.add_argument("--out", default="bc_v1_value.pt")
    t.add_argument("--data", default=None, help="npz path (default data/value_train.npz)")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.checkpoint, a.deck, a.games,
                out=Path(a.out) if a.out else DATA,
                search_plies=a.search_plies, search_samples=a.search_samples,
                meta_version=a.meta)
    else:
        train(a.checkpoint, a.out, data=Path(a.data) if a.data else DATA)
