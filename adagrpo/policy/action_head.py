"""Unified action head interface for diffusion and flow-matching policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DenoisingPath:
    """Complete record of a denoising trajectory, stored during rollout.

    This is the critical data structure for path-conditioned ratio computation.
    Standard diffusion policy inference does NOT record this — we must capture it.
    """

    actions: torch.Tensor
    """[B, H, D] final clean action chunk a^(0).  H = action horizon."""

    noisy_actions: torch.Tensor
    """[B, K, H, D] intermediate samples along the denoising path.
    Index k=0 is a^(K) (pure noise end), k=K-1 is a^(1) (near-clean end).
    We store a^(k-1) — the *target* of each denoising step."""

    means: torch.Tensor
    """[B, K, H, D] predicted means μ_{θ_old}(a^(k), k, s) from the rollout policy."""

    sigmas: torch.Tensor
    """[K] or [B, K] per-step noise standard deviations σ_k."""

    noise_samples: Optional[torch.Tensor] = None
    """[B, K, H, D] noise samples ε_k used at each step (optional, for debugging)."""


class ActionHead(ABC, nn.Module):
    """Abstract interface for diffusion/flow-matching action heads."""

    @abstractmethod
    def predict_action(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Standard inference: return only the final clean action.

        Args:
            obs_features: [B, F] fused vision-language features.
            num_denoise_steps: override default denoising budget.

        Returns:
            actions: [B, H, D] clean action chunk.
        """
        ...

    @abstractmethod
    def predict_action_with_path(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> DenoisingPath:
        """Inference with full denoising path recording for RL.

        Same as predict_action but additionally captures the complete
        denoising trajectory for subsequent ratio computation.

        Args:
            obs_features: [B, F] fused vision-language features.
            num_denoise_steps: override default denoising budget.

        Returns:
            DenoisingPath with all intermediate quantities.
        """
        ...

    @abstractmethod
    def denoise_step(
        self,
        noisy_action: torch.Tensor,
        timestep: int,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step: predict μ_θ(a^(k), k, s).

        Used during ratio computation to get the updated policy's mean
        at a recorded denoising step without re-running the full chain.

        Args:
            noisy_action: [B, H, D] noisy action a^(k).
            timestep: denoising step index k.
            obs_features: [B, F] observation features.

        Returns:
            mean: [B, H, D] predicted mean.
        """
        ...

    @abstractmethod
    def compute_denoising_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Standard denoising loss (MSE on predicted noise) for auxiliary loss.

        Args:
            obs_features: [B, F] observation features.
            actions: [B, H, D] ground-truth action chunks.

        Returns:
            Scalar loss.
        """
        ...
