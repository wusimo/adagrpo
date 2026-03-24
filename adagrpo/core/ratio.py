"""Path-conditioned importance ratio computation for diffusion policies.

The marginal action likelihood π_θ(a⁰|s) is intractable for diffusion models.
We decompose the importance ratio along the denoising path into per-step
Gaussian ratios, following the two-layer MDP formulation of DPPO, then
optionally apply learned importance weights (ALN) to suppress uninformative
steps and reduce variance.

Key equations
-------------
Per-step log-ratio:
    log r_k(θ) = (1 / 2σ_k²) * (||a^{k-1} - μ_{θ_old}||² - ||a^{k-1} - μ_θ||²)

Uniform path-conditioned log-ratio:
    log r(θ) = Σ_k  log r_k(θ)

Importance-weighted path-conditioned log-ratio:
    log r_w(θ) = Σ_k  w_k · log r_k(θ)
"""

from __future__ import annotations

import torch

# Numerical stability bound for log-ratios before exponentiation.
LOG_RATIO_CLAMP = 10.0


def compute_per_step_log_ratio(
    noisy_actions: torch.Tensor,
    means_old: torch.Tensor,
    means_new: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Compute per-denoising-step Gaussian log-ratios.

    All inputs are assumed to have been recorded during the rollout (old policy)
    and recomputed with the current policy (new means).

    Args:
        noisy_actions: [B, K, D]  a^{k-1} samples along the denoising path.
                       Index 0 corresponds to the step closest to data (k=1),
                       index K-1 to the step closest to noise (k=K).
        means_old:     [B, K, D]  μ_{θ_old}(a^k, k, s) recorded during rollout.
        means_new:     [B, K, D]  μ_θ(a^k, k, s) computed with updated policy.
        sigmas:        [K] or [B, K]  per-step standard deviations σ_k.

    Returns:
        log_ratios: [B, K]  per-step log r_k(θ).
    """
    if sigmas.dim() == 1:
        # Broadcast [K] -> [1, K, 1]
        sigmas = sigmas.unsqueeze(0).unsqueeze(-1)
    elif sigmas.dim() == 2:
        sigmas = sigmas.unsqueeze(-1)

    # ||a^{k-1} - μ||²  summed over action dimension D
    diff_old_sq = (noisy_actions - means_old).pow(2).sum(dim=-1)  # [B, K]
    diff_new_sq = (noisy_actions - means_new).pow(2).sum(dim=-1)  # [B, K]

    inv_two_sigma_sq = 0.5 / (sigmas.squeeze(-1).pow(2) + 1e-8)  # [1, K] or [B, K]

    log_ratios = inv_two_sigma_sq * (diff_old_sq - diff_new_sq)  # [B, K]
    return log_ratios


def compute_uniform_log_ratio(
    per_step_log_ratios: torch.Tensor,
) -> torch.Tensor:
    """Sum per-step log-ratios uniformly (standard path-conditioned ratio).

    Args:
        per_step_log_ratios: [B, K]

    Returns:
        log_ratio: [B]
    """
    return per_step_log_ratios.sum(dim=-1)


def compute_weighted_log_ratio(
    per_step_log_ratios: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Importance-weighted path-conditioned log-ratio.

    Args:
        per_step_log_ratios: [B, K]
        weights:             [K] or [B, K]  ALN importance weights in [0, 1].

    Returns:
        log_ratio: [B]  =  Σ_k  w_k · log r_k
    """
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)  # [1, K]
    return (per_step_log_ratios * weights).sum(dim=-1)


def safe_exp_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """Exponentiate log-ratio with clamping for numerical stability.

    Args:
        log_ratio: [B]

    Returns:
        ratio: [B]  exp(clamp(log_ratio))
    """
    return torch.exp(log_ratio.clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP))
