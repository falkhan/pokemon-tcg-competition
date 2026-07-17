"""Option-scoring policy network + value head (torch, training side only).

The network scores each presented option against the shared state encoding —
a pointer-network setup, so action masking is implicit: options that aren't
presented simply don't exist this step (ARCHITECTURE.md §3.1).

Export with `save_npz()` for the numpy-only Kaggle submission.
"""
import numpy as np
import torch
import torch.nn as nn

from .encoders import (EMBED_DIM, N_CARD_IDS, N_CONTEXTS, N_OPTION_IDS,
                       N_STATE_IDS, OPTION_DIM, OPTION_V2_DIM, STATE_DIM,
                       STATE_V2_DIM)


class OptionScorer(nn.Module):
    """score(state, context, option_i) for each option i; plus V(state)."""

    def __init__(self, hidden: int = 256, state_ctx_dim: int | None = None):
        # state_ctx_dim: override for checkpoints trained on an older encoder
        # (bc_v1 predates the M3 combat features); defaults to the current one.
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_ctx_dim or (STATE_DIM + N_CONTEXTS), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.option_enc = nn.Sequential(
            nn.Linear(OPTION_DIM, hidden), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),   # V in (-1, 1), matching the ±1 reward
        )

    def forward(self, state_ctx: torch.Tensor, options: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """state_ctx: (B, STATE_DIM+N_CONTEXTS); options: (B, N_opt, OPTION_DIM).

        Returns (logits (B, N_opt), value (B,)). Pad variable option counts to
        N_opt per batch and mask pad logits to -inf in the loss.
        """
        s = self.state_enc(state_ctx)                          # (B, H)
        o = self.option_enc(options)                           # (B, N, H)
        s_tiled = s.unsqueeze(1).expand(-1, o.shape[1], -1)    # (B, N, H)
        logits = self.score_head(torch.cat([s_tiled, o], dim=-1)).squeeze(-1)
        value = self.value_head(s).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state_ctx: np.ndarray, options: np.ndarray, k: int,
            greedy: bool = False) -> list[int]:
        """Pick k distinct option indices for one decision (numpy in, ints out)."""
        sc = torch.from_numpy(state_ctx).unsqueeze(0)
        op = torch.from_numpy(options).unsqueeze(0)
        logits, _ = self(sc, op)
        logits = logits.squeeze(0)
        if greedy:
            return torch.topk(logits, k).indices.tolist()
        # sample k without replacement via Gumbel top-k
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
        return torch.topk(logits + gumbel, k).indices.tolist()


class OptionScorerV2(nn.Module):
    """v2 pointer net (M7.3): the v1 architecture with card-id EMBEDDINGS.

    Takes the encoders-v2 (numeric, ids) pairs: ids index a learnable
    nn.Embedding whose rows concatenate onto the numeric vectors — the 36
    features are nearly blind for trainers; embeddings learn per-card behavior
    from data. Export stays numpy-trivial: the embedding is one more matrix in
    the npz, and inference is a row lookup (M7-plan §3.1b).
    """

    def __init__(self, hidden: int = 256, embed: int = EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(N_CARD_IDS, embed, padding_idx=0)
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_V2_DIM + N_CONTEXTS + N_STATE_IDS * embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.option_enc = nn.Sequential(
            nn.Linear(OPTION_V2_DIM + N_OPTION_IDS * embed, hidden), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, state_ctx: torch.Tensor, state_ids: torch.Tensor,
                options: torch.Tensor, option_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """state_ctx (B, STATE_V2_DIM+N_CONTEXTS); state_ids (B, N_STATE_IDS) long;
        options (B, N, OPTION_V2_DIM); option_ids (B, N, N_OPTION_IDS) long.
        Returns (logits (B, N), value (B,))."""
        se = self.embedding(state_ids).flatten(-2)             # (B, IDS*E)
        s = self.state_enc(torch.cat([state_ctx, se], dim=-1))  # (B, H)
        oe = self.embedding(option_ids).flatten(-2)            # (B, N, 2E)
        o = self.option_enc(torch.cat([options, oe], dim=-1))  # (B, N, H)
        s_tiled = s.unsqueeze(1).expand(-1, o.shape[1], -1)
        logits = self.score_head(torch.cat([s_tiled, o], dim=-1)).squeeze(-1)
        value = self.value_head(s).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state_ctx: np.ndarray, state_ids: np.ndarray,
            options: np.ndarray, option_ids: np.ndarray, k: int,
            greedy: bool = False) -> list[int]:
        """Pick k distinct option indices for one decision (numpy in, ints out)."""
        logits, _ = self(torch.from_numpy(state_ctx).unsqueeze(0),
                         torch.from_numpy(state_ids).long().unsqueeze(0),
                         torch.from_numpy(options).unsqueeze(0),
                         torch.from_numpy(option_ids).long().unsqueeze(0))
        logits = logits.squeeze(0)
        if greedy:
            return torch.topk(logits, k).indices.tolist()
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
        return torch.topk(logits + gumbel, k).indices.tolist()


class OptionScorerV3(nn.Module):
    """v3 (M11): OptionScorerV2 + turn-plan conditioning.

    `forward` takes an extra `plan` (B, PLAN_DIM) block concatenated into the
    state encoder input BETWEEN state_ctx and the id embeddings — that layout
    lets load_v2_into_v3 (rl/plan_iter.py) zero-init exactly the plan columns
    so V3(plan=0) == V2 (the warm-start invariant, unit-tested).
    `plan_logits` scores candidate plan encodings against the trunk — the
    plans-as-options pointer idiom: the chosen candidate's PLAN_DIM vector is
    byte-identical to what conditions the policy, so there is no train/serve
    mismatch in plan encoding. All-zeros plan == "no plan"."""

    def __init__(self, hidden: int = 256, embed: int = EMBED_DIM,
                 plan_dim: int | None = None, n_state_ids: int = N_STATE_IDS):
        super().__init__()
        if plan_dim is None:
            from rl.plan import PLAN_DIM
            plan_dim = PLAN_DIM
        self.plan_dim = plan_dim
        self.n_state_ids = n_state_ids     # M15: 12 legacy | 20 hand-aware
        self.embedding = nn.Embedding(N_CARD_IDS, embed, padding_idx=0)
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_V2_DIM + N_CONTEXTS + plan_dim + n_state_ids * embed,
                      hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.option_enc = nn.Sequential(
            nn.Linear(OPTION_V2_DIM + N_OPTION_IDS * embed, hidden), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )
        self.plan_enc = nn.Sequential(
            nn.Linear(plan_dim, hidden), nn.ReLU(),
        )
        self.plan_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _trunk(self, state_ctx, plan, state_ids):
        se = self.embedding(state_ids).flatten(-2)
        return self.state_enc(torch.cat([state_ctx, plan, se], dim=-1))

    def forward(self, state_ctx: torch.Tensor, plan: torch.Tensor,
                state_ids: torch.Tensor, options: torch.Tensor,
                option_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """state_ctx (B, STATE_V2_DIM+N_CONTEXTS); plan (B, PLAN_DIM);
        state_ids (B, N_STATE_IDS) long; options (B, N, OPTION_V2_DIM);
        option_ids (B, N, N_OPTION_IDS) long. Returns (logits (B,N), value (B,))."""
        s = self._trunk(state_ctx, plan, state_ids)
        oe = self.embedding(option_ids).flatten(-2)
        o = self.option_enc(torch.cat([options, oe], dim=-1))
        s_tiled = s.unsqueeze(1).expand(-1, o.shape[1], -1)
        logits = self.score_head(torch.cat([s_tiled, o], dim=-1)).squeeze(-1)
        value = self.value_head(s).squeeze(-1)
        return logits, value

    def plan_logits(self, state_ctx: torch.Tensor, state_ids: torch.Tensor,
                    plan_cands: torch.Tensor) -> torch.Tensor:
        """plan_cands (B, M, PLAN_DIM) -> (B, M). The trunk runs with
        plan=zeros ("no plan committed yet")."""
        zeros = state_ctx.new_zeros(state_ctx.shape[0], self.plan_dim)
        s = self._trunk(state_ctx, zeros, state_ids)
        p = self.plan_enc(plan_cands)
        s_tiled = s.unsqueeze(1).expand(-1, p.shape[1], -1)
        return self.plan_head(torch.cat([s_tiled, p], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, state_ctx: np.ndarray, plan: np.ndarray, state_ids: np.ndarray,
            options: np.ndarray, option_ids: np.ndarray, k: int,
            greedy: bool = False) -> list[int]:
        """Pick k distinct option indices for one decision (numpy in, ints out)."""
        logits, _ = self(torch.from_numpy(state_ctx).unsqueeze(0),
                         torch.from_numpy(plan).unsqueeze(0),
                         torch.from_numpy(state_ids).long().unsqueeze(0),
                         torch.from_numpy(options).unsqueeze(0),
                         torch.from_numpy(option_ids).long().unsqueeze(0))
        logits = logits.squeeze(0)
        if greedy:
            return torch.topk(logits, k).indices.tolist()
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
        return torch.topk(logits + gumbel, k).indices.tolist()

    @torch.no_grad()
    def act_plan(self, state_ctx: np.ndarray, state_ids: np.ndarray,
                 plan_cands: np.ndarray, tau: float = 0.0,
                 dirichlet_eps: float = 0.0, rng=None) -> int:
        """Pick a plan candidate index. tau<=0: argmax (eval). tau>0:
        softmax(logits/tau) sampling, optionally mixed with Dirichlet(0.5)
        noise (AZ-style, collection round 1 only)."""
        logits = self.plan_logits(
            torch.from_numpy(state_ctx).unsqueeze(0),
            torch.from_numpy(state_ids).long().unsqueeze(0),
            torch.from_numpy(plan_cands).unsqueeze(0)).squeeze(0)
        if tau <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / tau, dim=-1)
        if dirichlet_eps > 0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.dirichlet([0.5] * probs.shape[0]).astype(np.float32)
            probs = (1 - dirichlet_eps) * probs + dirichlet_eps * torch.from_numpy(noise)
        return int(torch.multinomial(probs, 1).item())


def save_npz(model: OptionScorer, path: str) -> None:
    """Export weights for the numpy-only submission agent."""
    arrays = {name: p.detach().cpu().numpy() for name, p in model.state_dict().items()}
    np.savez(path, **arrays)
