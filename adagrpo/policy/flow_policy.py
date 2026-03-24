"""Flow-matching policy variant.

Flow matching is an alternative to DDPM/DDIM that learns a velocity field
v_θ(x_t, t) such that integrating the ODE dx/dt = v_θ(x_t, t) from t=1
(noise) to t=0 (data) produces clean samples.

The path-conditioned ratio decomposition still applies: each discretised
Euler step has a tractable Gaussian form when we add stochastic noise.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adagrpo.policy.action_head import ActionHead, DenoisingPath
from adagrpo.policy.diffusion_policy import ConditionalUNet1D


class FlowMatchingPolicy(ActionHead):
    """Flow-matching action head with path recording."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int = 16,
        hidden_dim: int = 256,
        num_inference_steps: int = 10,
        sigma_min: float = 1e-4,
        noise_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min

        if noise_net is not None:
            self.velocity_net = noise_net
        else:
            self.velocity_net = ConditionalUNet1D(
                action_dim=action_dim,
                obs_dim=obs_dim,
                action_horizon=action_horizon,
                hidden_dim=hidden_dim,
            )

    def _get_timesteps(self, K: int, device: torch.device) -> torch.Tensor:
        """Linearly spaced timesteps from 1 (noise) to 0 (data)."""
        return torch.linspace(1.0, 0.0, K + 1, device=device)

    @torch.no_grad()
    def predict_action(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> torch.Tensor:
        K = num_denoise_steps or self.num_inference_steps
        B = obs_features.shape[0]
        device = obs_features.device
        ts = self._get_timesteps(K, device)

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        for i in range(K):
            t = ts[i]
            dt = ts[i + 1] - ts[i]  # Negative (going from 1 to 0)
            t_batch = torch.full((B,), t.item(), device=device).long()
            # Scale to integer timestep for embedding
            t_int = (t * 99).long().expand(B)
            v = self.velocity_net(x, t_int, obs_features)
            x = x + v * dt

        return x

    @torch.no_grad()
    def predict_action_with_path(
        self,
        obs_features: torch.Tensor,
        num_denoise_steps: Optional[int] = None,
    ) -> DenoisingPath:
        K = num_denoise_steps or self.num_inference_steps
        B = obs_features.shape[0]
        device = obs_features.device
        ts = self._get_timesteps(K, device)

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        all_noisy = []
        all_means = []
        all_sigmas = []

        for i in range(K):
            t = ts[i]
            dt = ts[i + 1] - ts[i]
            t_int = (t * 99).long().expand(B)
            v = self.velocity_net(x, t_int, obs_features)

            # Euler step mean
            mean = x + v * dt
            # Add small stochastic noise for non-final steps (stochastic Euler)
            sigma = self.sigma_min if i == K - 1 else abs(dt.item()) * 0.1
            if i < K - 1:
                noise = torch.randn_like(x) * sigma
                x_next = mean + noise
            else:
                x_next = mean

            all_noisy.append(x_next)
            all_means.append(mean)
            all_sigmas.append(sigma)

            x = x_next

        return DenoisingPath(
            actions=x,
            noisy_actions=torch.stack(all_noisy, dim=1),
            means=torch.stack(all_means, dim=1),
            sigmas=torch.tensor(all_sigmas, device=device),
        )

    def denoise_step(
        self,
        noisy_action: torch.Tensor,
        timestep: int,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        B = noisy_action.shape[0]
        device = noisy_action.device
        K = self.num_inference_steps
        ts = self._get_timesteps(K, device)

        # Map integer timestep to continuous time
        t = ts[timestep]
        dt = ts[timestep + 1] - ts[timestep] if timestep + 1 <= K else torch.tensor(0.0)
        t_int = (t * 99).long().expand(B)
        v = self.velocity_net(noisy_action, t_int, obs_features)
        return noisy_action + v * dt

    def compute_denoising_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional flow matching loss."""
        B = actions.shape[0]
        device = actions.device

        t = torch.rand(B, device=device)
        noise = torch.randn_like(actions)

        # Interpolate: x_t = (1-t) * x_0 + t * noise
        t_exp = t[:, None, None]
        x_t = (1 - t_exp) * actions + t_exp * noise

        # Target velocity: noise - x_0
        target_v = noise - actions

        t_int = (t * 99).long()
        pred_v = self.velocity_net(x_t, t_int, obs_features)

        return F.mse_loss(pred_v, target_v)
