"""AdaGRPO loss: importance-weighted path-conditioned GRPO with auxiliary denoising loss.

Loss = E[ min( r_w(θ) · Â_i,  clip(r_w(θ), 1-ε, 1+ε) · Â_i ) ] + λ · L_aux

where r_w(θ) = exp( Σ_k  w_k · log r_k(θ) )  is the ALN-weighted ratio
and Â_i is the group-relative advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from adagrpo.core.ratio import (
    compute_per_step_log_ratio,
    compute_uniform_log_ratio,
    compute_weighted_log_ratio,
    safe_exp_ratio,
)
from adagrpo.core.advantages import (
    compute_batched_group_advantages,
    filter_uninformative_groups,
)


@dataclass
class GRPOLossOutput:
    """Container for AdaGRPO loss components and diagnostics."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    aux_loss: torch.Tensor
    ratio_mean: float
    ratio_std: float
    ratio_max: float
    clipped_frac: float
    advantage_mean: float
    log_ratio_var: float


class AdaGRPOLoss(nn.Module):
    """Compute the full AdaGRPO surrogate objective."""

    def __init__(
        self,
        clip_eps: float = 0.2,
        aux_weight: float = 0.1,
        group_size: int = 8,
        success_threshold: float = 0.5,
        use_aln_weights: bool = True,
        filter_uniform_groups: bool = True,
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.aux_weight = aux_weight
        self.group_size = group_size
        self.success_threshold = success_threshold
        self.use_aln_weights = use_aln_weights
        self.filter_uniform_groups = filter_uniform_groups

    def forward(
        self,
        noisy_actions: torch.Tensor,
        means_old: torch.Tensor,
        means_new: torch.Tensor,
        sigmas: torch.Tensor,
        rewards: torch.Tensor,
        aln_weights: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> GRPOLossOutput:
        """Compute the AdaGRPO loss.

        Args:
            noisy_actions: [B, K, D] recorded denoising path samples.
            means_old:     [B, K, D] old policy predicted means.
            means_new:     [B, K, D] current policy predicted means.
            sigmas:        [K] or [B, K] per-step noise scales.
            rewards:       [B] episode rewards (B = num_groups * group_size).
            aln_weights:   [K] or [B, K] ALN importance weights, or None for
                           uniform weighting.
            aux_loss:      optional scalar auxiliary denoising loss.

        Returns:
            GRPOLossOutput with loss and diagnostics.
        """
        # 1. Per-step log-ratios
        per_step_lr = compute_per_step_log_ratio(
            noisy_actions, means_old, means_new, sigmas
        )  # [B, K]

        # 2. Aggregate log-ratio (weighted or uniform)
        if self.use_aln_weights and aln_weights is not None:
            log_ratio = compute_weighted_log_ratio(per_step_lr, aln_weights)
        else:
            log_ratio = compute_uniform_log_ratio(per_step_lr)
        # [B]

        ratio = safe_exp_ratio(log_ratio)  # [B]

        # 3. Group-relative advantages
        advantages = compute_batched_group_advantages(
            rewards, self.group_size
        )  # [B]

        # 4. Optionally filter uninformative groups
        if self.filter_uniform_groups:
            group_mask = filter_uninformative_groups(
                rewards, self.group_size, self.success_threshold
            )
            # Expand group mask to per-trajectory mask
            traj_mask = group_mask.repeat_interleave(self.group_size)
            if traj_mask.sum() == 0:
                # All groups are uninformative; skip update
                zero = torch.tensor(0.0, device=rewards.device, requires_grad=True)
                return GRPOLossOutput(
                    loss=zero, policy_loss=zero, aux_loss=zero,
                    ratio_mean=1.0, ratio_std=0.0, ratio_max=1.0,
                    clipped_frac=0.0, advantage_mean=0.0, log_ratio_var=0.0,
                )
        else:
            traj_mask = torch.ones(rewards.shape[0], dtype=torch.bool, device=rewards.device)

        # 5. Clipped surrogate objective
        ratio_m = ratio[traj_mask]
        adv_m = advantages[traj_mask]

        surr1 = ratio_m * adv_m
        surr2 = torch.clamp(ratio_m, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_m
        policy_loss = -torch.min(surr1, surr2).mean()

        # 6. Total loss
        if aux_loss is None:
            aux_loss_val = torch.tensor(0.0, device=rewards.device)
        else:
            aux_loss_val = aux_loss
        total_loss = policy_loss + self.aux_weight * aux_loss_val

        # Diagnostics
        clipped = ((ratio_m - 1.0).abs() > self.clip_eps).float().mean().item()
        log_ratio_m = log_ratio[traj_mask]

        return GRPOLossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            aux_loss=aux_loss_val,
            ratio_mean=ratio_m.mean().item(),
            ratio_std=ratio_m.std().item(),
            ratio_max=ratio_m.max().item(),
            clipped_frac=clipped,
            advantage_mean=adv_m.mean().item(),
            log_ratio_var=log_ratio_m.var().item(),
        )
