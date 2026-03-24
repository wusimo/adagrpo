"""Group-relative advantage computation for GRPO.

GRPO computes advantages by normalizing returns within a group of G
trajectories sampled from the same initial state, eliminating the need
for a learned value function.

    Â_i = (R_i - mean(R_{1:G})) / (std(R_{1:G}) + δ)
"""

from __future__ import annotations

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    delta: float = 1e-8,
) -> torch.Tensor:
    """Group-relative advantage normalisation.

    Args:
        rewards: [G] cumulative episode rewards for G trajectories
                 sampled from the same initial state.
        delta:   small constant for numerical stability.

    Returns:
        advantages: [G] zero-mean, unit-variance advantages.
    """
    mean_r = rewards.mean()
    std_r = rewards.std()
    return (rewards - mean_r) / (std_r + delta)


def compute_batched_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    delta: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for a batch of groups.

    Args:
        rewards:    [B] flat tensor where B = num_groups * group_size.
        group_size: G, number of trajectories per group.
        delta:      numerical stability constant.

    Returns:
        advantages: [B] advantages (each group independently normalised).
    """
    num_groups = rewards.shape[0] // group_size
    grouped = rewards.view(num_groups, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - means) / (stds + delta)
    return advantages.view(-1)


def filter_uninformative_groups(
    rewards: torch.Tensor,
    group_size: int,
    success_threshold: float = 0.5,
) -> torch.Tensor:
    """Return mask indicating which groups have mixed outcomes.

    Groups where all trajectories succeed or all fail provide no
    contrastive signal and should be discarded from the GRPO update
    (per SimpleVLA-RL / RLinf-VLA).

    Args:
        rewards:           [B] flat rewards, B = num_groups * group_size.
        group_size:        G.
        success_threshold: reward above this is counted as success.

    Returns:
        group_mask: [num_groups] boolean mask, True = informative group.
    """
    num_groups = rewards.shape[0] // group_size
    grouped = rewards.view(num_groups, group_size)
    successes = (grouped > success_threshold).float().sum(dim=1)
    # Keep groups that are neither all-success nor all-fail
    return (successes > 0) & (successes < group_size)
