"""Diffusion Policy with denoising path recording for AdaGRPO.

Wraps a noise-prediction network (CNN or Transformer) with DDPM/DDIM
scheduling and provides the path-recording inference required for
path-conditioned ratio computation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adagrpo.policy.action_head import ActionHead, DenoisingPath
from adagrpo.utils.diffusion_utils import (
    DDIMScheduler,
    DDPMScheduler,
    SinusoidalEmbedding,
)


class ConditionalUNet1D(nn.Module):
    """Minimal 1-D U-Net noise prediction network for action sequences.

    This is a simplified version suitable for prototyping. For production,
    replace with the full architecture from the Diffusion Policy codebase.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int = 16,
        hidden_dim: int = 256,
        num_timesteps: int = 100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.time_embed = SinusoidalEmbedding(hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        input_dim = action_dim + hidden_dim * 2  # action + time_emb + obs_emb

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise (or mean, depending on parameterisation).

        Args:
            noisy_action: [B, H, D] noisy action chunk.
            timestep:     [B] integer timesteps.
            obs_features: [B, F] observation features.

        Returns:
            predicted: [B, H, D] predicted noise.
        """
        B, H, D = noisy_action.shape
        t_emb = self.time_embed(timestep)  # [B, hidden]
        o_emb = self.obs_proj(obs_features)  # [B, hidden]

        # Expand embeddings across horizon
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, hidden]
        o_emb_exp = o_emb.unsqueeze(1).expand(-1, H, -1)  # [B, H, hidden]

        x = torch.cat([noisy_action, t_emb_exp, o_emb_exp], dim=-1)  # [B, H, input_dim]
        return self.net(x)  # [B, H, D]


class DiffusionPolicy(ActionHead):
    """Diffusion Policy action head with path recording for AdaGRPO.

    Supports both DDPM (training) and DDIM (fast inference / RL rollouts).
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int = 16,
        hidden_dim: int = 256,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        schedule: str = "cosine",
        ddim_eta: float = 0.0,
        noise_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Noise prediction network
        if noise_net is not None:
            self.noise_net = noise_net
        else:
            self.noise_net = ConditionalUNet1D(
                action_dim=action_dim,
                obs_dim=obs_dim,
                action_horizon=action_horizon,
                hidden_dim=hidden_dim,
                num_timesteps=num_train_timesteps,
            )

        # Schedulers
        self.ddpm = DDPMScheduler(num_train_timesteps, schedule)
        self.ddim = DDIMScheduler(num_train_timesteps, num_inference_steps, schedule, ddim_eta)

    # ------------------------------------------------------------------
    # Standard inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> torch.Tensor:
        B = obs_features.shape[0]
        device = obs_features.device

        scheduler = self._get_inference_scheduler(num_denoise_steps)
        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        for t, t_prev in scheduler.get_step_pairs():
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_net(x, t_batch, obs_features)
            x, _, _ = scheduler.step(noise_pred, x, t, t_prev)

        return x

    # ------------------------------------------------------------------
    # Path-recording inference (critical for AdaGRPO)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_with_path(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> DenoisingPath:
        B = obs_features.shape[0]
        device = obs_features.device

        scheduler = self._get_inference_scheduler(num_denoise_steps)
        step_pairs = scheduler.get_step_pairs()
        K = len(step_pairs)

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        all_inputs = []  # a^(k): input to each denoising step
        all_noisy = []   # a^{k-1}: output (target) of each denoising step
        all_means = []   # μ_{θ_old}
        all_sigmas = []  # σ_k (effective posterior std for ratio computation)

        for t, t_prev in step_pairs:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            all_inputs.append(x.clone())  # Store input a^(k)
            noise_pred = self.noise_net(x, t_batch, obs_features)
            x_prev, mean, sigma = scheduler.step(noise_pred, x, t, t_prev)

            all_noisy.append(x_prev)  # a^{k-1}
            all_means.append(mean)
            all_sigmas.append(sigma)

            x = x_prev

        return DenoisingPath(
            actions=x,  # final clean a^(0)
            noisy_actions=torch.stack(all_noisy, dim=1),  # [B, K, H, D]
            means=torch.stack(all_means, dim=1),  # [B, K, H, D]
            sigmas=torch.tensor(all_sigmas, device=device),  # [K]
            noise_samples=torch.stack(all_inputs, dim=1),  # [B, K, H, D] — a^(k) inputs
        )

    # ------------------------------------------------------------------
    # Single-step denoising (for ratio computation with updated policy)
    # ------------------------------------------------------------------

    def denoise_step(
        self,
        noisy_action: torch.Tensor,
        timestep: int,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute μ_θ(a^k, k, s) — the current policy's predicted mean."""
        B = noisy_action.shape[0]
        device = noisy_action.device
        t_batch = torch.full((B,), timestep, device=device, dtype=torch.long)

        noise_pred = self.noise_net(noisy_action, t_batch, obs_features)

        # For DDIM: mean = α_{t_prev}^{1/2} * pred_x0 + dir_xt
        # We return the mean (deterministic component) of the reverse step.
        scheduler = self.ddim
        alpha_t = scheduler.state.alphas_cumprod[timestep]

        # pred_x0
        pred_x0 = (noisy_action - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # Find t_prev from step_pairs
        step_pairs = scheduler.get_step_pairs()
        t_prev = 0
        for t_cur, tp in step_pairs:
            if t_cur == timestep:
                t_prev = tp
                break

        alpha_prev = scheduler.state.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
        eta = scheduler.eta
        sigma_t = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp(min=0).sqrt()
        dir_xt = (1 - alpha_prev - sigma_t**2).clamp(min=0).sqrt() * noise_pred
        mean = alpha_prev.sqrt() * pred_x0 + dir_xt

        return mean

    # ------------------------------------------------------------------
    # Recompute means for full path (batch operation for ratio computation)
    # ------------------------------------------------------------------

    def recompute_path_means(
        self,
        path: DenoisingPath,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute μ_θ for all K steps using the current policy.

        Uses the stored a^(k) inputs (noise_samples field) so that we feed
        the exact same noisy actions to the new policy as the old policy saw.

        Args:
            path: recorded denoising path from rollout (θ_old).
                  path.noise_samples must contain [B, K, H, D] of a^(k) inputs.
            obs_features: [B, F] observation features.

        Returns:
            means_new: [B, K, H, D] predicted means from current policy.
        """
        B, K, H, D = path.noisy_actions.shape
        device = obs_features.device
        step_pairs = self.ddim.get_step_pairs()

        # noise_samples stores a^(k) — the input to each denoising step
        assert path.noise_samples is not None, (
            "DenoisingPath.noise_samples must be populated with a^(k) inputs "
            "for recompute_path_means(). Use predict_action_with_path()."
        )

        means_new = []
        for k_idx in range(K):
            t = step_pairs[k_idx][0]
            a_k = path.noise_samples[:, k_idx]  # [B, H, D] exact input to step k
            mean = self.denoise_step(a_k, t, obs_features)
            means_new.append(mean)

        return torch.stack(means_new, dim=1)  # [B, K, H, D]

    # ------------------------------------------------------------------
    # Denoising loss (auxiliary / IL training)
    # ------------------------------------------------------------------

    def compute_denoising_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Standard MSE denoising loss for IL training or auxiliary loss.

        Args:
            obs_features: [B, F] observation features.
            actions: [B, H, D] ground-truth action chunks.
            reduction: 'mean', 'none', or 'per_step'.

        Returns:
            loss: scalar if reduction='mean', [B, T] if 'per_step',
                  [B] if 'none'.
        """
        B = actions.shape[0]
        device = actions.device

        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(actions)
        noisy_actions = self.ddpm.q_sample(
            actions.view(B, -1), t, noise.view(B, -1)
        ).view(B, self.action_horizon, self.action_dim)

        # Predict noise
        noise_pred = self.noise_net(noisy_actions, t, obs_features)

        if reduction == "mean":
            return F.mse_loss(noise_pred, noise)
        elif reduction == "none":
            return F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2))
        elif reduction == "per_step":
            # Return loss grouped by timestep (for ALN training)
            return F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_inference_scheduler(self, num_steps: Optional[int] = None) -> DDIMScheduler:
        if num_steps is not None and num_steps != self.ddim.num_inference_steps:
            return DDIMScheduler(
                self.num_train_timesteps,
                num_steps,
                schedule="cosine",
                eta=self.ddim.eta,
            )
        return self.ddim
