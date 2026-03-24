"""Adaptive Loss Network (ALN).

The ALN learns per-denoising-step importance weights during imitation learning,
indicating which steps are informationally critical for each state-action pair.
During RL, these weights suppress ratio contributions from uninformative
denoising steps, reducing the variance of the path-conditioned ratio product.

Architecture: 3-layer MLP with sinusoidal timestep embeddings, optionally
conditioned on a state embedding. Outputs sigmoid-activated weights ∈ [0, 1].

ALN RL reward signal (used to update ALN during RL):
    r_k^RL = -(ℓ_k - μ_ℓ) / (σ_ℓ + ε) + β · |log r_k(θ)|
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from adagrpo.utils.diffusion_utils import SinusoidalEmbedding


class AdaptiveLossNetwork(nn.Module):
    """Lightweight network that predicts per-step importance weights.

    During IL: trained to minimise variance-weighted denoising loss, learning
    which timesteps carry the most gradient signal for a given state.

    During RL: weights are used to compute the importance-weighted ratio
    log r_w(θ) = Σ_k  w_k · log r_k(θ).
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        state_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.timestep_embed = SinusoidalEmbedding(embed_dim)

        input_dim = embed_dim
        if state_dim is not None:
            self.state_proj = nn.Linear(state_dim, embed_dim)
            input_dim = embed_dim * 2
        else:
            self.state_proj = None

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        state_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute unnormalized weight logits.

        Args:
            timesteps: [B] or [K] integer timestep indices.
            state_emb: optional [B, state_dim] state embedding for
                       state-conditioned weights.

        Returns:
            logits: [B] or [K] raw logits (apply sigmoid externally).
        """
        t_emb = self.timestep_embed(timesteps)  # [*, embed_dim]

        if self.state_proj is not None and state_emb is not None:
            s_emb = self.state_proj(state_emb)  # [B, embed_dim]
            if t_emb.shape[0] != s_emb.shape[0]:
                # Broadcasting: expand state embedding to match timesteps
                s_emb = s_emb.expand(t_emb.shape[0], -1)
            t_emb = torch.cat([t_emb, s_emb], dim=-1)

        return self.mlp(t_emb).squeeze(-1)

    @torch.no_grad()
    def get_weights(
        self,
        K: int,
        state_emb: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get normalized importance weights for K denoising steps.

        Args:
            K: number of denoising steps.
            state_emb: optional [B, state_dim] state embedding.
            device: device for the timestep tensor.

        Returns:
            weights: [K] or [B, K] weights in [0, 1].
        """
        if device is None:
            device = next(self.parameters()).device
        k = torch.arange(1, K + 1, device=device)
        logits = self.forward(k, state_emb)
        return torch.sigmoid(logits)


def compute_aln_il_loss(
    per_step_losses: torch.Tensor,
    aln: AdaptiveLossNetwork,
    timesteps: torch.Tensor,
    state_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ALN co-training loss during imitation learning.

    The ALN learns to upweight timesteps with high denoising loss (they are
    more informative) while maintaining coverage via an entropy bonus.

    Args:
        per_step_losses: [B, K] denoising loss at each timestep.
        aln: the Adaptive Loss Network.
        timesteps: [K] timestep indices.
        state_emb: optional [B, state_dim] state embedding.

    Returns:
        Scalar loss for ALN parameter update.
    """
    logits = aln(timesteps, state_emb)  # [K]
    weights = torch.sigmoid(logits)      # [K]

    # Weighted denoising loss (ALN should upweight high-loss steps)
    weighted_loss = (per_step_losses * weights.unsqueeze(0)).mean()

    # Entropy bonus to avoid degenerate collapse to a single step
    entropy = -weights * torch.log(weights + 1e-8) - (1 - weights) * torch.log(1 - weights + 1e-8)
    entropy_bonus = entropy.mean()

    return weighted_loss - 0.01 * entropy_bonus


def compute_aln_rl_reward(
    per_step_losses: torch.Tensor,
    per_step_log_ratios: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """ALN reward signal during RL (used to optionally fine-tune ALN).

    r_k^RL = -(ℓ_k - μ_ℓ) / (σ_ℓ + ε) + β · |log r_k(θ)|

    Args:
        per_step_losses: [B, K] denoising loss at each step.
        per_step_log_ratios: [B, K] per-step log importance ratios.
        beta: coefficient for ratio magnitude term.

    Returns:
        rewards: [B, K] per-step ALN rewards.
    """
    mu = per_step_losses.mean(dim=0, keepdim=True)
    sigma = per_step_losses.std(dim=0, keepdim=True)
    normalized = -(per_step_losses - mu) / (sigma + 1e-8)
    ratio_magnitude = per_step_log_ratios.abs()
    return normalized + beta * ratio_magnitude
