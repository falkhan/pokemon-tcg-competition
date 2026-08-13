"""Self-play PPO — the campaign's live RL trainer (M2 origin, M20 KL anchor,
M21 opponent curriculum + plan surrogate, M23 vs_teacher promotion, M43 value
shaping).

Split with rl/collector.py: this module owns the iteration loop, GAE
(compute_gae), the clipped update (ppo_update), evaluation + promotion, and
checkpoints; the collector owns rollout workers, reward shaping, and the
opponent pool. tcg/ppo.py is a frozen M8-era parity twin of the pure
functions only (pinned by tests/test_ppo.py), not the pipeline.

Run one iteration end-to-end:
  python -m rl.ppo --iterations 1
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.collector import collect
from rl.policy import (OptionScorer, OptionScorerV2, OptionScorerV3,
                       resolve_device)

ROOT = Path(__file__).resolve().parent.parent
PPO_DIR = ROOT / "data" / "ppo"
CKPT_DIR = ROOT / "checkpoints"

GAMMA = 0.99          # discount: how much future reward is worth now
LAM = 0.95            # GAE lambda: bias/variance dial for advantage estimates
CLIP_EPS = 0.2        # PPO ratio clip
ENTROPY_COEF = 0.001  # was 0.01 -- too high: the 25-iter run showed entropy RISING
                      # (0.82 -> 1.0), i.e. the bonus overpowered the weak advantage signal
                      # and diffused the policy toward random. Lowered 10x so the policy
                      # gradient can actually sharpen the policy.
VALUE_COEF = 0.5


# ---------------------------------------------------------------------------
# Shard loading + trajectory slicing (infra, done)
# ---------------------------------------------------------------------------

def load_shards(ppo_dir: Path = PPO_DIR) -> dict[str, np.ndarray]:
    """Concatenate all ppo_shard_*.npz into flat arrays (same trick as BCDataset:
    ragged options stay flat, `starts` gives each decision's slice offset).
    game_ids are remapped to be globally unique across shards."""
    cols: dict[str, list] = {}
    game_base = 0
    option_base = 0
    cand_base = 0
    for path in sorted(ppo_dir.glob("ppo_shard_*.npz")):
        z = np.load(path)
        starts = np.cumsum(z["n_options"]) - z["n_options"]
        cols.setdefault("starts", []).append(starts + option_base)
        cols.setdefault("game_ids", []).append(z["game_ids"] + game_base)
        for k in ("states", "options", "n_options", "actions", "logprobs",
                  "values", "rewards", "players"):
            cols.setdefault(k, []).append(z[k])
        for k in ("state_ids", "option_ids", "deck_idx",    # encoders-v2 shards (M7.4b)
                  "plans"):                                 # v3 plan vectors (M20)
            if k in z:
                cols.setdefault(k, []).append(z[k])
        if "plan_cands" in z:                               # M21 plan-PPO shards
            ncands = z["n_plan_cands"]
            cols.setdefault("cand_starts", []).append(
                np.cumsum(ncands) - ncands + cand_base)
            for k in ("plan_cands", "n_plan_cands", "plan_actions",
                      "plan_logprobs"):
                cols.setdefault(k, []).append(z[k])
            cand_base += len(z["plan_cands"])
        option_base += len(z["options"])
        game_base += int(z["game_ids"].max()) + 1
    return {k: np.concatenate(v) for k, v in cols.items()}


def trajectory_slices(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Row indices of each (game, seat) trajectory, in decision order.

    GAE runs over one agent's consecutive decisions; the shards interleave both
    seats chronologically, so this de-interleaves them. Row order within a shard
    is already chronological."""
    key = data["game_ids"].astype(np.int64) * 2 + data["players"]
    order = np.argsort(key, kind="stable")           # stable keeps chronology
    slices, prev, start = [], None, 0
    for pos, idx in enumerate(order):
        if key[idx] != prev:
            if prev is not None:
                slices.append(order[start:pos])
            prev, start = key[idx], pos
    slices.append(order[start:])
    return slices


_SUPPORTER_IDS: set | None = None


def _supporter_ids() -> set:
    global _SUPPORTER_IDS
    if _SUPPORTER_IDS is None:
        import polars as pl
        df = pl.read_parquet(ROOT / "data" / "cards_features.parquet")
        _SUPPORTER_IDS = set(
            df.filter(pl.col("is_supporter") == 1)["card_id"].to_list())
    return _SUPPORTER_IDS


def advantage_by_type(data: dict[str, np.ndarray],
                      advantages: np.ndarray) -> dict[str, tuple]:
    """M23 audit S4 (Piotr-approved logging): GAE-advantage mean/std/n per
    chosen-option type; PLAY split supporter/other by card identity. Read:
    supporter advantages ~0 or noise-drowned while attack advantages are
    clean = credit is not reaching card-economy actions."""
    from rl.encoders import N_OPTION_TYPES
    rows = data["starts"] + data["actions"]
    kinds = data["options"][rows, :N_OPTION_TYPES].argmax(axis=1)
    classes = {"attach": kinds == 8, "attack": kinds == 13, "end": kinds == 14}
    play = kinds == 7
    if "option_ids" in data:
        sup = np.isin(data["option_ids"][rows, 0], list(_supporter_ids()))
        classes["play_supporter"] = play & sup
        classes["play_other"] = play & ~sup
    else:
        classes["play"] = play
    return {name: (float(advantages[m].mean()), float(advantages[m].std()),
                   int(m.sum()))
            for name, m in classes.items() if m.any()}


def collate_ppo(data: dict[str, np.ndarray], rows: np.ndarray,
                advantages: np.ndarray, returns: np.ndarray):
    """Build one padded/masked minibatch from row indices (same padding idea as
    rl/bc.py collate). Returns tensors ready for ppo_update; the two id slots
    are None for v1 shards and padded long tensors for encoders-v2 shards."""
    B = len(rows)
    n_opts = data["n_options"][rows]
    maxN = int(n_opts.max())
    opt_dim = data["options"].shape[1]
    has_ids = "state_ids" in data

    states = torch.from_numpy(data["states"][rows])
    options = torch.zeros(B, maxN, opt_dim)
    valid = torch.zeros(B, maxN, dtype=torch.bool)
    state_ids = (torch.from_numpy(data["state_ids"][rows].astype(np.int64))
                 if has_ids else None)
    plans = (torch.from_numpy(data["plans"][rows]).float()
             if "plans" in data else None)               # v3 shards (M20)
    option_ids = (torch.zeros(B, maxN, data["option_ids"].shape[1], dtype=torch.long)
                  if has_ids else None)
    for i, r in enumerate(rows):
        s, n = data["starts"][r], n_opts[i]
        options[i, :n] = torch.from_numpy(data["options"][s:s + n])
        if has_ids:
            option_ids[i, :n] = torch.from_numpy(
                data["option_ids"][s:s + n].astype(np.int64))
        valid[i, :n] = True

    return (states, state_ids, options, option_ids, plans, valid,
            torch.from_numpy(data["actions"][rows]).long(),
            torch.from_numpy(data["logprobs"][rows]).float(),
            torch.from_numpy(advantages[rows]).float(),
            torch.from_numpy(returns[rows]).float())


def collate_plan(data: dict[str, np.ndarray], rows: np.ndarray):
    """M21 plan-PPO minibatch slice: the subset of `rows` that carry a plan
    decision (n_plan_cands > 0), padded. Returns None when the shard has no
    plan columns or the batch has no plan rows; else
    (sub_idx, plan_cands, plan_valid, plan_actions, plan_logprobs) where
    sub_idx indexes back into the batch for states/advantages."""
    if "plan_cands" not in data:
        return None
    ncands = data["n_plan_cands"][rows]
    sub = np.nonzero(ncands > 0)[0]
    if not len(sub):
        return None
    maxM = int(ncands[sub].max())
    plan_dim = data["plan_cands"].shape[1]
    cands = torch.zeros(len(sub), maxM, plan_dim)
    valid = torch.zeros(len(sub), maxM, dtype=torch.bool)
    for j, bi in enumerate(sub):
        r = rows[bi]
        cs, m = data["cand_starts"][r], data["n_plan_cands"][r]
        cands[j, :m] = torch.from_numpy(data["plan_cands"][cs:cs + m])
        valid[j, :m] = True
    return (torch.from_numpy(sub).long(), cands, valid,
            torch.from_numpy(data["plan_actions"][rows][sub]).long(),
            torch.from_numpy(data["plan_logprobs"][rows][sub]).float())


# ---------------------------------------------------------------------------
# THE CORE — Piotr's part
# ---------------------------------------------------------------------------

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float = GAMMA, lam: float = LAM) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over ONE trajectory (one game, one seat).

    Inputs are that trajectory's per-decision rewards and value predictions, in
    order. The episode ends after the last decision (terminal value = 0).

    Returns (advantages, returns) for the trajectory, where
      delta_t   = r_t + gamma * V(s_{t+1}) - V(s_t)        # TD error; V(s_T) = 0
      A_t       = delta_t + gamma * lam * A_{t+1}          # computed BACKWARD
      return_t  = A_t + V(s_t)                             # value-head target

    Hints: walk the arrays back-to-front with a running accumulator (exactly like
    the reversed() loop in the official MCTS sample's value labeling). Loss ~2.0
    moment here: with lam=1, gamma=1 this reduces to "sum of future rewards minus
    V" — sanity-check your implementation against that by hand on a 3-step toy.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _forward(model, states, state_ids, options, option_ids, plans):
    """Dispatch the right forward signature: v1 (2 args), v2 (4), v3 (5 — the
    plan tensor, M20). Duck-typed on plan_enc so either twin's V3 class works
    (the parity tests drive rl models through tcg's update and vice versa)."""
    if hasattr(model, "plan_enc"):
        if plans is None:                      # v2-era shards under a v3 net
            plans = torch.zeros(len(states), model.plan_dim,
                                device=states.device)
        return model(states, plans, state_ids, options, option_ids)
    if state_ids is None:
        return model(states, options)
    return model(states, state_ids, options, option_ids)


def _to_device(device, *tensors):
    """Move a collated batch to the training device (None = leave on CPU,
    the historical behavior — keeps the tcg/ppo.py parity twin comparable)."""
    if device is None:
        return tensors
    return tuple(t.to(device) if t is not None else None for t in tensors)


def ppo_update(model: OptionScorer, opt: torch.optim.Optimizer,
               data: dict[str, np.ndarray], advantages: np.ndarray,
               returns: np.ndarray, epochs: int = 4, batch_size: int = 256,
               entropy_coef: float = ENTROPY_COEF,
               ref_model=None, kl_coef: float = 0.0,
               plan_coef: float = 0.0, device=None) -> dict:
    """The clipped PPO update over all collected decisions.

    Per minibatch (use collate_ppo above; shuffle rows each epoch):
      1. logits, value = model(states, options); mask invalid logits to -1e9.
      2. new_logprob   = log_softmax(logits)[action]       # gather along dim 1
      3. ratio         = exp(new_logprob - old_logprob)    # how far policy moved
      4. policy_loss   = -min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()
      5. entropy       = -(softmax * log_softmax).sum(dim over REAL options).mean()
      6. value_loss    = F.huber_loss(value, returns_batch)
      7. loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
      8. zero_grad / backward / clip_grad_norm_(model.parameters(), 0.5) / step

    Normalize advantages ONCE globally before training: (A - A.mean()) / (A.std()+1e-8).
    Return a dict of mean losses/entropy for TensorBoard.
    Read FIRST: the "37 implementation details of PPO" post (ARCHITECTURE.md §9).
    """
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(data["actions"])
    logs = {"policy_loss": [], "value_loss": [], "entropy": [], "ratio": []}
    if kl_coef:
        logs["kl"] = []
    if plan_coef:
        logs["plan_loss"] = []
        logs["plan_entropy"] = []

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            rows = perm[i:i + batch_size]
            states, state_ids, options, option_ids, plans, valid, actions, old_lp, adv_b, ret_b = \
                _to_device(device, *collate_ppo(data, rows, adv, returns))
            logits, value = _forward(model, states, state_ids, options,
                                     option_ids, plans)
            logits = logits.masked_fill(~valid, -1e9)
            logp = torch.log_softmax(logits, dim=1)
            new_lp = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv_b
            clipped = torch.clamp(ratio, 1- CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(unclipped, clipped).mean()

            probs = logp.exp()
            entropy = -(probs * logp).masked_fill(~valid, 0.0).sum(dim=1).mean()
            value_loss = F.huber_loss(value, ret_b)

            # M7.4b fix: this was `policy_loss * VALUE_COEF * value_loss` for
            # every stalled PPO run (M2-M4) — a multiplicative loss whose policy
            # gradient scales with the value error (and flips sign with it)
            # instead of the documented additive PPO objective.
            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy

            if plan_coef:
                # M21 plan-PPO: same clipped surrogate over the plan head for
                # the rows that carry a plan decision. The plan action shares
                # its first-MAIN option row's advantage (same state, zero
                # reward in between — no separate GAE step, compute_gae is
                # untouched). NOT covered by the KL anchor below: the anchor
                # would pin the plan head to the BC prior, which is exactly
                # the never-gusts behavior this term exists to escape; a plan
                # entropy bonus guards against collapse instead.
                pb = collate_plan(data, rows)
                if pb is not None:
                    sub, cands, pvalid, pactions, p_old_lp = \
                        _to_device(device, *pb)
                    plan_logits = model.plan_logits(states[sub],
                                                    None if state_ids is None
                                                    else state_ids[sub], cands)
                    plan_logits = plan_logits.masked_fill(~pvalid, -1e9)
                    p_logp = torch.log_softmax(plan_logits, dim=1)
                    p_new_lp = p_logp.gather(
                        1, pactions.unsqueeze(1)).squeeze(1)
                    p_ratio = torch.exp(p_new_lp - p_old_lp)
                    p_adv = adv_b[sub]
                    p_unclipped = p_ratio * p_adv
                    p_clipped = torch.clamp(p_ratio, 1 - CLIP_EPS,
                                            1 + CLIP_EPS) * p_adv
                    plan_loss = -torch.min(p_unclipped, p_clipped).mean()
                    p_probs = p_logp.exp()
                    plan_entropy = -(p_probs * p_logp).masked_fill(
                        ~pvalid, 0.0).sum(dim=1).mean()
                    loss = loss + plan_coef * plan_loss \
                        - entropy_coef * plan_entropy
                    logs["plan_loss"].append(plan_loss.item())
                    logs["plan_entropy"].append(plan_entropy.item())

            if ref_model is not None and kl_coef:
                # M20 KL-anchor (the M8-plan leg C, finally spent): a rubber
                # band to the frozen start policy. KL(pi_new || pi_ref) over
                # each decision's REAL options — the defect penalty pushes
                # down over-attach, this term pulls everything else back
                # toward the anchor (the strength-loss mode M19 measured in
                # BC reweighting, addressed structurally).
                with torch.no_grad():
                    ref_logits, _ = _forward(ref_model, states, state_ids,
                                             options, option_ids, plans)
                    ref_logp = torch.log_softmax(
                        ref_logits.masked_fill(~valid, -1e9), dim=1)
                kl = (probs * (logp - ref_logp)).masked_fill(~valid, 0.0) \
                    .sum(dim=1).mean()
                loss = loss + kl_coef * kl
                logs["kl"].append(kl.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy.item())
            logs["ratio"].append(ratio.mean().item())
    # drop keys that never collected a value (e.g. plan_coef on but the data
    # carries no plan columns) — np.mean([]) is a nan trap
    return {k: float(np.mean(v)) for k, v in logs.items() if v}

# ---------------------------------------------------------------------------
# Iteration loop (infra, done)
# ---------------------------------------------------------------------------

def warm_start_value(model, value_ckpt: Path) -> list[str]:
    """Load ONLY the value head from a rl/value_train.py checkpoint — the
    supervised critic is the one proven training success (MSE 1.0->0.35,
    sign-acc 0.87); starting PPO from it beats trusting PPO to learn a critic
    from sparse ±1 returns (M7-plan §3.1e). value_head keys are shape-identical
    across OptionScorer and OptionScorerV2 (both operate on the hidden-dim body
    output); for v2 bodies the fit is approximate but a far better prior than
    random. Returns the loaded key names."""
    sd = torch.load(value_ckpt, map_location="cpu")
    value_keys = {k: v for k, v in sd.items() if k.startswith("value_head.")}
    model.load_state_dict(value_keys, strict=False)
    return sorted(value_keys)


def _load_model(ckpt_path: Path):
    """Build the right architecture for a checkpoint: OptionScorerV3 for
    plan-conditioned checkpoints (M11+, dims sniffed from the weights — the
    rl/matchrunner formula), OptionScorerV2 when the state dict carries the
    id-embedding table (M7.3), else v1."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if "plan_enc.0.weight" in sd:                       # M20: v3 support
        from rl.encoders import (EMBED_DIM, N_CONTEXTS, STATE_V2_DIM,
                                 V4_EXTRA_DIM)
        from rl.plan import PLAN_DIM
        from rl.policy import option_dim_of
        extra = V4_EXTRA_DIM if "enc_ver" in sd else 0  # M21: v4 sniff
        n_ids = (sd["state_enc.0.weight"].shape[1] - STATE_V2_DIM - N_CONTEXTS
                 - extra - PLAN_DIM) // EMBED_DIM
        model = OptionScorerV3(n_state_ids=n_ids, option_dim=option_dim_of(sd),
                               extra_dim=extra)
    elif "embedding.weight" in sd:
        model = OptionScorerV2()
    else:
        model = OptionScorer()
    model.load_state_dict(sd)
    return model


def _as_kaggle_agent(ckpt_path: Path, deck, instance: str):
    """A kaggle_environments-callable agent from a checkpoint: matchrunner's
    v2/pre-M3-aware model pilot plus the deck-return first call (the direct
    engine loop never asks for the deck; env.run does)."""
    from rl.matchrunner import make_pilot
    fn, deck_ids = make_pilot(("model", str(ckpt_path), deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in fn(obs_dict)]
    return agent


def _solver_opponent(deck: str, instance: str):
    """The FIXED eval opponent (M7.5 attempt 2): the live solver ship agent.
    Attempt 1 promoted on the mirror (vs the drifting champion) and the two
    metrics decorrelated — vs_best held ~47% while G3 collapsed 0.44->0.145.
    Promotion now tracks the actual goal: beat the rules agent we ship."""
    from rl.matchrunner import make_pilot
    fn, deck_ids = make_pilot(("solver", deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in fn(obs_dict)]
    return agent


def _save_train_state(path: Path, model, opt, next_it: int, best_wr: float,
                      meta: dict) -> None:
    """M44 3a: everything a killed leg needs to continue — model + optimizer
    state, the first iteration NOT yet run, the promotion bar, and the run
    identity (`meta`). Atomic: <path>.tmp then os.replace, so a SIGKILL
    mid-save leaves the previous state intact."""
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "opt": opt.state_dict(),
             "next_it": next_it, "best_wr": best_wr, "meta": meta}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_train_state(path: Path, meta: dict) -> dict:
    """M44 3b: load + refuse on identity mismatch. start/iterations/learn_deck
    must match the CLI exactly (a resumed leg is the SAME leg); `opponents` is
    stored for the record but not enforced — the pool legitimately updates as
    other league legs complete."""
    if not path.exists():
        raise FileNotFoundError(f"--resume: no state file {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("start", "iterations", "learn_deck"):
        if state["meta"].get(k) != meta.get(k):
            raise ValueError(
                f"--resume meta mismatch on {k!r}: state has "
                f"{state['meta'].get(k)!r}, CLI has {meta.get(k)!r}")
    return state


def train(iterations: int, games_per_iter: int = 400, workers: int = 4,
          lr: float = 1e-4, eval_every: int = 5, start: str = "bc_v1.pt",
          value_ckpt: str | None = None, decks_file: str | None = None,
          eval_deck: str = "kyogre", entropy_coef: float = ENTROPY_COEF,
          race_shaping: float = 0.0, shaping: str = "race",
          phi_ckpt: str | None = None,
          defect_penalty: float = 0.0, kl_coef: float = 0.0,
          learn_deck: str | None = None, tag: str = "", eval_games: int = 200,
          opponents: list[str] | None = None,
          opponent_schedule: str | None = None,
          plan_tau: float = 0.0, plan_dirichlet: float = 0.0,
          plan_coef: float = 0.0, gust_boost: float = 0.0,
          entropy_anneal_to: float | None = None,
          kl_anneal_to: float | None = None,
          device: str = "auto", resume: bool = False):
    """M21 additions (all default-off = the M20 recipe exactly):
    opponents        — 'spec=weight' mixture for the collector (rl/collector
                       parse_pool syntax incl. mirror=/past= tokens)
    opponent_schedule— JSON file [{"until_iter": N, "opponents": [...]}, ...];
                       first entry with it < until_iter wins over `opponents`
    plan_tau/plan_dirichlet — plan-level exploration at collection
    plan_coef        — plan-head clipped-surrogate weight in ppo_update
                       (turns on plan recording in the collector)
    entropy_anneal_to/kl_anneal_to — linear per-iteration anneal targets for
                       entropy_coef / kl_coef across the leg."""
    import json as _json

    from torch.utils.tensorboard import SummaryWriter
    from rl.collector import parse_pool
    from rl.eval import play_games
    from rl.teacher import load_teacher

    schedule = (_json.loads(Path(opponent_schedule).read_text())
                if opponent_schedule else None)

    def _mix_for(it: int) -> list[str] | None:
        if schedule:
            for entry in schedule:
                if it < int(entry["until_iter"]):
                    return entry["opponents"]
            return schedule[-1]["opponents"]
        return opponents

    def _annealed(base: float, target: float | None, it: int) -> float:
        if target is None or iterations <= 1:
            return base
        return base + (target - base) * it / (iterations - 1)

    suffix = f"_{tag}" if tag else ""       # M20: namespace runs so the M8-era
    writer = SummaryWriter(str(ROOT / "runs" / f"ppo{suffix}"))  # artifacts survive
    best_path = CKPT_DIR / f"ppo_best{suffix}.pt"

    # M44 3a-3c: pausable leg. State is saved at the end of every iteration;
    # a sentinel file pauses cleanly at the next iteration top.
    state_path = CKPT_DIR / f"ppo_state{suffix}.pt"
    pause_path = CKPT_DIR / f"ppo_pause{suffix}"
    meta = {"start": start, "iterations": iterations,
            "learn_deck": learn_deck, "opponents": opponents}
    saved = _load_train_state(state_path, meta) if resume else None
    # --resume consumes the sentinel; a FRESH run clears a stale one too, so a
    # leftover touch from a finished leg cannot instantly pause its successor.
    pause_path.unlink(missing_ok=True)

    if not resume and not best_path.exists():
        shutil.copy(CKPT_DIR / start, best_path)       # the start is the initial champion

    dev = resolve_device(device)
    if dev.type != "cpu":
        print(f"training on {dev} (updates only; collection/eval stay CPU)",
              flush=True)
    model = _load_model(CKPT_DIR / start)
    if resume:
        model.load_state_dict(saved["model"])
    elif value_ckpt:
        # not on resume: the resumed sd already contains the warmed critic
        loaded = warm_start_value(model, Path(value_ckpt))
        print(f"critic warm-start from {value_ckpt}: {loaded}", flush=True)
    model.to(dev)
    ref_model = None
    if kl_coef:
        # M20 leg B: the KL anchor is the FROZEN start policy.
        ref_model = _load_model(CKPT_DIR / start)
        ref_model.to(dev)
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if resume:
        # After AdamW construction (params already on dev): load_state_dict
        # re-homes exp_avg/exp_avg_sq to each param's device/dtype.
        opt.load_state_dict(saved["opt"])

    if resume:
        # M44 3b: the stored bar, NOT a re-measure — re-measuring would drift
        # the promotion bar by eval noise on every pause/resume cycle.
        next_it, best_wr = saved["next_it"], saved["best_wr"]
        print(f"resume: starting at iter {next_it}, "
              f"best vs_teacher {best_wr:.1%} (stored)", flush=True)
    else:
        next_it = 0
        # M23: promotion tracks the TEACHER (the sample agent — the strongest
        # observable opponent), not the solver the M22 audit demoted to an
        # endogenous in-loop readout. Bar = the current best's own measured rate,
        # so a fresh warm start must genuinely improve to promote. vs_solver is
        # still measured and logged every eval; it just no longer gates.
        best_wr, _ = play_games(_as_kaggle_agent(best_path, eval_deck, "ppo_bp_base"),
                                load_teacher("ppo_tc_base"), eval_games,
                                names=("best", "teacher"))
        print(f"baseline: current best vs_teacher {best_wr:.1%}", flush=True)

    for it in range(next_it, iterations):
        # M44 3c: pause sentinel — checked before the shard wipe so a paused
        # leg leaves nothing half-collected.
        if pause_path.exists():
            _save_train_state(state_path, model, opt, it, best_wr, meta)
            print(f"PAUSED before iter {it}", flush=True)
            return
        # 1. fresh self-play data with the CURRENT policy
        for old in PPO_DIR.glob("ppo_shard_*.npz"):
            old.unlink()                                # on-policy: stale data is poison
        work = CKPT_DIR / f"ppo_current{suffix}.pt"
        # CPU tensors: the spawn workers (and every other consumer) load on CPU
        cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_sd, work)
        mix = _mix_for(it)
        pool = (parse_pool(mix, str(work), learn_deck or "kyogre", tag=tag)
                if mix else None)
        if mix:
            print(f"iter {it}: opponent mix {mix}", flush=True)
        _, defect_rate, cause_counts = collect(
            games_per_iter, str(work), workers, decks_file=decks_file,
            pool=pool, race_shaping=race_shaping, shaping=shaping,
            # M43: the value potential defaults to the FROZEN START's head,
            # never the moving learner — same anchor the KL term uses.
            # --phi-ckpt overrides it for the Phase-0 branch where the critic
            # warm-starts from a different checkpoint (--value-ckpt) and Phi
            # must follow.
            phi_ckpt=(str(CKPT_DIR / (phi_ckpt or start))
                      if shaping == "value" else None),
            defect_penalty=defect_penalty,
            plan_tau=plan_tau, plan_dirichlet=plan_dirichlet,
            plan_ppo=plan_coef > 0, gust_boost=gust_boost,
            **({"learn_deck": learn_deck} if learn_deck else {}))
        writer.add_scalar("train/defect_rate", defect_rate, it)
        # M44 3g: per-cause game-end telemetry (engine RESULT reason). The
        # registered read: no loss cause may RISE over a leg while overall win
        # rate improves — outcome-only reward can hide that trade.
        from rl.collector import CAUSE_BUCKETS
        for c in CAUSE_BUCKETS:
            writer.add_scalar(f"loss_cause/{c}",
                              cause_counts.get(f"loss_{c}", 0) / games_per_iter, it)
            writer.add_scalar(f"win_cause/{c}",
                              cause_counts.get(f"win_{c}", 0) / games_per_iter, it)
        print(f"iter {it}: loss-cause "
              + " ".join(f"{c}:{cause_counts.get('loss_' + c, 0)}"
                         for c in CAUSE_BUCKETS)
              + "  win-cause "
              + " ".join(f"{c}:{cause_counts.get('win_' + c, 0)}"
                         for c in CAUSE_BUCKETS)
              + f"  /{games_per_iter}g", flush=True)
        data = load_shards()

        # 2. advantages per (game, seat) trajectory  [Piotr's compute_gae]
        advantages = np.zeros(len(data["actions"]), dtype=np.float32)
        returns = np.zeros_like(advantages)
        for rows in trajectory_slices(data):
            adv, ret = compute_gae(data["rewards"][rows], data["values"][rows])
            advantages[rows], returns[rows] = adv, ret

        # M23 audit S4: advantage mass by chosen-option type (logging only)
        adv_stats = advantage_by_type(data, advantages)
        for name, (mu, sd, n) in adv_stats.items():
            writer.add_scalar(f"adv/{name}_mean", mu, it)
            writer.add_scalar(f"adv/{name}_std", sd, it)
        print("iter %d: adv-by-type %s" % (it, "  ".join(
            f"{k} {mu:+.3f}±{sd:.3f}(n={n})"
            for k, (mu, sd, n) in sorted(adv_stats.items()))), flush=True)

        # 3. clipped update (M20 KL anchor; M21 plan surrogate + anneals)
        ec_it = _annealed(entropy_coef, entropy_anneal_to, it)
        kl_it = _annealed(kl_coef, kl_anneal_to, it) if kl_coef else 0.0
        stats = ppo_update(model, opt, data, advantages, returns,
                           entropy_coef=ec_it,
                           ref_model=ref_model, kl_coef=kl_it,
                           plan_coef=plan_coef, device=dev)
        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, it)
        writer.add_scalar("train/entropy_coef", ec_it, it)
        writer.add_scalar("train/kl_coef", kl_it, it)
        print(f"iter {it}: defect/g {defect_rate:.2f}  ec {ec_it:.4f}  "
              f"kl_coef {kl_it:.3f}  {stats}", flush=True)

        # 4. periodic evaluation + promotion vs the fixed goal opponent
        if (it + 1) % eval_every == 0:
            challenger = _as_kaggle_agent(work, eval_deck, f"ppo_ch{it}")
            wr, _ = play_games(challenger, _solver_opponent(eval_deck, f"ppo_sv{it}"),
                               eval_games, names=("challenger", "solver"))
            wr_teacher, _ = play_games(challenger, load_teacher(f"ev{it}"), eval_games,
                                       names=("challenger", "teacher"))
            writer.add_scalar("eval/vs_solver", wr, it)
            writer.add_scalar("eval/vs_teacher", wr_teacher, it)
            print(f"iter {it}: vs_solver {wr:.1%}  vs_teacher {wr_teacher:.1%}")
            if wr_teacher > best_wr:
                best_wr = wr_teacher
                cpu_sd = {k: v.detach().cpu()
                          for k, v in model.state_dict().items()}
                torch.save(cpu_sd, best_path)
                torch.save(cpu_sd, CKPT_DIR / f"ppo{suffix}_it{it:04d}.pt")
                print(f"iter {it}: PROMOTED (best vs_teacher {wr_teacher:.1%})")

        # M44 3a: iteration complete — persist. A SIGKILL now loses nothing;
        # mid-iteration it loses only this iteration.
        _save_train_state(state_path, model, opt, it + 1, best_wr, meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--start", type=str, default="bc_v1.pt")
    p.add_argument("--value-ckpt", type=str, default=None,
                   help="critic warm-start from a rl.value_train checkpoint")
    p.add_argument("--decks", type=str, default=None,
                   help="deck population for multi-deck self-play (v2 checkpoints)")
    p.add_argument("--eval-deck", type=str, default="kyogre")
    p.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF,
                   help="entropy bonus; cut it when entropy RISES over a run "
                        "(the M2 diffusion signature, seen again in attempt 1)")
    p.add_argument("--race-shaping", type=float, default=0.0,
                   help="potential-based setup-shaping coef for the collector")
    p.add_argument("--shaping-coef", dest="race_shaping", type=float,
                   default=argparse.SUPPRESS,
                   help="neutral alias for --race-shaping (M43: the coef "
                        "applies to whichever potential --shaping selects, "
                        "not just the race delta). SUPPRESS keeps the "
                        "original's 0.0 default when the alias is absent")
    p.add_argument("--shaping", choices=["race", "dev", "value"], default="race",
                   help="shaping potential: race delta (M7.4b), the M8.3 "
                        "dev potential (race+ready+evo+bench), or the frozen "
                        "start's own V(s) (M43 — BACKLOG #13(c) step 2)")
    p.add_argument("--phi-ckpt", type=str, default=None,
                   help="checkpoint whose value head serves as the Phi "
                        "potential under --shaping value (default: --start). "
                        "Set it alongside --value-ckpt so Phi follows the "
                        "critic's warm start instead of silently staying on "
                        "--start")
    p.add_argument("--eval-every", type=int, default=5,
                   help="iterations between evals (M8.3 decay probes use 2)")
    p.add_argument("--defect-penalty", type=float, default=0.0,
                   help="M20: per-step reward dock for over-attach actions "
                        "(0.1 = one prize-shaping unit); 0 = off")
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="M20 leg B: KL(pi_new || frozen start) anchor coef; "
                        "0 = clip only")
    p.add_argument("--learn-deck", type=str, default=None,
                   help="learner's fixed deck (default: collector's LEARN_DECK); "
                        "the M20 probe uses lucario (the measured pairing)")
    p.add_argument("--tag", type=str, default="",
                   help="namespaces ppo_best/ppo_it checkpoints + the TB run "
                        "dir (keeps M8-era artifacts intact)")
    p.add_argument("--eval-games", type=int, default=200,
                   help="games per baseline/promotion eval (sequential — the "
                        "wall-clock bottleneck; smoke runs use small values)")
    p.add_argument("--opponents", type=str, nargs="*", default=None,
                   help="M21: 'spec=weight' collector mixture "
                        "(mirror=/past= tokens; default = default_pool)")
    p.add_argument("--opponent-schedule", type=str, default=None,
                   help="M21: JSON file [{'until_iter': N, 'opponents': "
                        "[...]}, ...] — overrides --opponents per iteration")
    p.add_argument("--plan-tau", type=float, default=0.0,
                   help="M21: plan sampling temperature at collection "
                        "(0 = greedy argmax, the M20 behavior)")
    p.add_argument("--plan-dirichlet", type=float, default=0.0,
                   help="M21: Dirichlet(0.5) mix into plan sampling")
    p.add_argument("--plan-coef", type=float, default=0.0,
                   help="M21: plan-head clipped-surrogate weight (0 = plan "
                        "head untrained, the M20 behavior); also turns on "
                        "plan recording in the collector")
    p.add_argument("--entropy-anneal-to", type=float, default=None,
                   help="M21: linear entropy_coef target at the last iter")
    p.add_argument("--gust-boost", type=float, default=0.0,
                   help="M21 B3: mix this probability onto the Boss's Orders "
                        "PLAY option when the committed plan needs a gust "
                        "(guided exploration; 0 = off)")
    p.add_argument("--kl-anneal-to", type=float, default=None,
                   help="M21: linear kl_coef target at the last iter "
                        "(anneal to 0 = pure self-play by leg end)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="M43: device for the PPO update (auto = cuda when "
                        "available). Collection and eval stay CPU; "
                        "checkpoints are saved as CPU tensors either way; "
                        "cuda runs are NOT bit-reproducible vs cpu")
    p.add_argument("--resume", action="store_true",
                   help="M44: resume a paused/killed leg from checkpoints/"
                        "ppo_state_<tag>.pt (requires --tag; skips the "
                        "baseline re-measure — the stored promotion bar is "
                        "reused — and deletes the pause sentinel). The CLI "
                        "must repeat the original --start/--iterations/"
                        "--learn-deck exactly")
    args = p.parse_args()
    if args.resume and not args.tag:
        p.error("--resume requires --tag (state files are tag-namespaced)")
    if args.resume and not (CKPT_DIR / f"ppo_state_{args.tag}.pt").exists():
        p.error(f"--resume: no state file checkpoints/ppo_state_{args.tag}.pt")
    if args.shaping != "race" and args.race_shaping == 0:
        # A non-default potential with a zero coefficient is always a mistake:
        # the collector builds no phi net and the leg silently runs unshaped
        # (rl/collector.py gates on `race_shaping and shaping == ...`).
        p.error(f"--shaping {args.shaping} is a no-op without "
                "--race-shaping > 0")
    if args.phi_ckpt and args.shaping != "value":
        p.error("--phi-ckpt only applies under --shaping value")
    train(args.iterations, args.games_per_iter, args.workers, lr=args.lr,
          start=args.start, value_ckpt=args.value_ckpt, decks_file=args.decks,
          eval_deck=args.eval_deck, entropy_coef=args.entropy_coef,
          race_shaping=args.race_shaping, shaping=args.shaping,
          phi_ckpt=args.phi_ckpt,
          eval_every=args.eval_every, defect_penalty=args.defect_penalty,
          kl_coef=args.kl_coef, learn_deck=args.learn_deck, tag=args.tag,
          eval_games=args.eval_games, opponents=args.opponents,
          opponent_schedule=args.opponent_schedule, plan_tau=args.plan_tau,
          plan_dirichlet=args.plan_dirichlet, plan_coef=args.plan_coef,
          gust_boost=args.gust_boost,
          entropy_anneal_to=args.entropy_anneal_to,
          kl_anneal_to=args.kl_anneal_to,
          device=args.device, resume=args.resume)
