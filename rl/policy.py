"""Option-scoring policy network + value head (torch, training side only).

The network scores each presented option against the shared state encoding —
a pointer-network setup, so action masking is implicit: options that aren't
presented simply don't exist this step (ARCHITECTURE.md §3.1).

Export with `save_npz()` for the numpy-only Kaggle submission.
"""
import numpy as np
import torch
import torch.nn as nn

from .encoders import N_CONTEXTS, OPTION_DIM, STATE_DIM


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


def save_npz(model: OptionScorer, path: str) -> None:
    """Export weights for the numpy-only submission agent."""
    arrays = {name: p.detach().cpu().numpy() for name, p in model.state_dict().items()}
    np.savez(path, **arrays)
