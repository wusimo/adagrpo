"""Diffusion scheduling utilities: DDPM, DDIM, sinusoidal embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps into *dim*-dimensional vectors.

        Args:
            timesteps: [B] integer or float timestep indices.

        Returns:
            [B, dim] sinusoidal embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine variance schedule (Nichol & Dhariwal 2021).

    Returns:
        betas: [T] tensor of beta values.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0, 0.999)


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


@dataclass
class DiffusionScheduleState:
    """Pre-computed diffusion schedule quantities."""

    betas: torch.Tensor          # [T]
    alphas: torch.Tensor         # [T]
    alphas_cumprod: torch.Tensor # [T]
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor  # [T]


def _build_schedule(betas: torch.Tensor) -> DiffusionScheduleState:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return DiffusionScheduleState(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=alphas_cumprod.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt(),
        sqrt_recip_alphas=(1.0 / alphas).sqrt(),
        posterior_variance=posterior_variance,
    )


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler."""

    def __init__(self, num_timesteps: int = 100, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.num_timesteps = num_timesteps
        self.state = _build_schedule(betas)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to x_start at timestep t.

        Args:
            x_start: [B, D] clean samples.
            t: [B] integer timesteps in [0, T-1].
            noise: optional pre-sampled noise.

        Returns:
            [B, D] noisy samples.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.state.sqrt_alphas_cumprod.to(x_start.device)[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.state.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t].unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_sample_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Single DDPM reverse step.

        Args:
            model_output: predicted noise ε_θ(x_t, t).
            x_t: [B, D] current noisy sample.
            t: scalar timestep.
            clip_denoised: whether to clip the predicted x_0.

        Returns:
            (x_{t-1}, predicted_mean, sigma_t)
        """
        beta_t = self.state.betas[t]
        sqrt_recip_alpha = self.state.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod = self.state.sqrt_one_minus_alphas_cumprod[t]

        # Predicted mean
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_alpha_cumprod * model_output)

        if clip_denoised:
            mean = mean.clamp(-1.0, 1.0)

        sigma_t = self.state.posterior_variance[t].sqrt()

        if t > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev, mean, sigma_t.item()


class DDIMScheduler:
    """Denoising Diffusion Implicit Model scheduler (deterministic or stochastic)."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        schedule: str = "cosine",
        eta: float = 0.0,
    ):
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.state = _build_schedule(betas)

        # Sub-sampled timestep sequence for inference
        step_ratio = num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long().flip(0)

    def step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Single DDIM reverse step.

        Args:
            model_output: predicted noise ε_θ(x_t, t).
            x_t: [B, D] current noisy sample.
            t: current timestep index (in training schedule).
            t_prev: previous timestep index (or 0 for final step).

        Returns:
            (x_{t_prev}, predicted_mean, sigma_t) where sigma_t is the stochastic
            component magnitude.
        """
        alpha_t = self.state.alphas_cumprod[t]
        alpha_prev = self.state.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        pred_x0 = (x_t - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()

        # Compute sigma for stochastic DDIM
        # The "natural" posterior variance (used even for deterministic DDIM
        # as the effective scale for importance ratio computation)
        posterior_var = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        posterior_var = posterior_var.clamp(min=1e-8)
        sigma_t = self.eta * posterior_var.sqrt()

        # Direction pointing to x_t
        dir_xt = (1 - alpha_prev - sigma_t**2).clamp(min=0).sqrt() * model_output

        # Mean prediction
        mean = alpha_prev.sqrt() * pred_x0 + dir_xt

        if sigma_t > 0 and t_prev > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        # Return the effective posterior std for ratio computation.
        # Even when eta=0 (deterministic DDIM), we report the natural posterior
        # std so that importance ratios remain well-defined.
        effective_sigma = posterior_var.sqrt().item()

        return x_prev, mean, effective_sigma

    def get_step_pairs(self) -> list[tuple[int, int]]:
        """Return (t, t_prev) pairs for the full denoising trajectory."""
        pairs = []
        for i in range(len(self.timesteps)):
            t = self.timesteps[i].item()
            t_prev = self.timesteps[i + 1].item() if i + 1 < len(self.timesteps) else 0
            pairs.append((t, t_prev))
        return pairs
