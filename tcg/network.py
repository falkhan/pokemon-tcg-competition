"""Option-scoring policy network + value head (torch, training side only).

Same network as the old ``rl/policy.py``. The submodule attribute names
(``state_enc`` / ``option_enc`` / ``score_head`` / ``value_head``) and the
default width are part of every saved checkpoint's ``state_dict`` keys and of
the exported ``.npz`` — do not rename them (test-pinned).

The network scores each presented option against the shared state encoding —
a pointer-network setup, so action masking is implicit: options that aren't
presented simply don't exist this step (ARCHITECTURE.md §3.1).

Export with ``save_npz()`` for the numpy-only Kaggle submission.
"""
import numpy as np
import torch
import torch.nn as nn

from tcg.encoders import (EMBED_DIM, N_CARD_IDS, N_CONTEXTS, N_OPTION_IDS,
                          N_STATE_IDS, OPTION_DIM, OPTION_V2_DIM, STATE_DIM,
                          STATE_V2_DIM)

MASKED_LOGIT = -1e9
"""Logit assigned to padding options so softmax ignores them (see forward())."""


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
        N_opt per batch and mask pad logits to MASKED_LOGIT in the loss.
        """
        state = self.state_enc(state_ctx)                          # (B, H)
        option = self.option_enc(options)                          # (B, N, H)
        state_tiled = state.unsqueeze(1).expand(-1, option.shape[1], -1)  # (B, N, H)
        logits = self.score_head(torch.cat([state_tiled, option], dim=-1)).squeeze(-1)
        value = self.value_head(state).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state_ctx: np.ndarray, options: np.ndarray, k: int,
            greedy: bool = False) -> list[int]:
        """Pick k distinct option indices for one decision (numpy in, ints out)."""
        state_ctx_batch = torch.from_numpy(state_ctx).unsqueeze(0)
        option_batch = torch.from_numpy(options).unsqueeze(0)
        logits, _ = self(state_ctx_batch, option_batch)
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


def save_npz(model: OptionScorer, path: str) -> None:
    """Export weights for the numpy-only submission agent."""
    arrays = {name: parameter.detach().cpu().numpy()
              for name, parameter in model.state_dict().items()}
    np.savez(path, **arrays)
