"""Rollout collection with optional stage-aware scheduling.

Collects trajectories (episodes) from the environment using the current
policy. When HVTS is available, dynamically adjusts the denoising budget
(N_d) and action horizon (N_a) per task stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from adagrpo.policy.action_head import ActionHead, DenoisingPath
from adagrpo.scheduling.budget_allocator import BudgetAllocator, RolloutBudget
from adagrpo.scheduling.hvts import TaskDecomposition

logger = logging.getLogger("adagrpo")


@dataclass
class Trajectory:
    """A single episode trajectory with recorded denoising paths."""

    observations: list[dict]
    """Per-step observations."""

    actions: list[torch.Tensor]
    """Per-step action chunks [H, D]."""

    denoising_paths: list[DenoisingPath]
    """Recorded denoising paths for each action chunk."""

    rewards: list[float]
    """Per-step rewards."""

    obs_features: list[torch.Tensor]
    """[F] observation feature vectors (input to action head)."""

    total_reward: float = 0.0
    success: bool = False
    length: int = 0

    budgets_used: list[RolloutBudget] = field(default_factory=list)
    """Per-step budgets (if stage-aware scheduling is used)."""


@dataclass
class RolloutBatch:
    """A batch of trajectories collected for a GRPO group."""

    trajectories: list[Trajectory]
    state_id: int = 0  # Initial state identifier for hard-trajectory mining

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor([t.total_reward for t in self.trajectories])

    @property
    def successes(self) -> list[bool]:
        return [t.success for t in self.trajectories]

    @property
    def mean_length(self) -> float:
        lengths = [t.length for t in self.trajectories]
        return sum(lengths) / len(lengths) if lengths else 0.0


class RolloutCollector:
    """Collect rollout trajectories with optional stage-aware scheduling."""

    def __init__(
        self,
        obs_encoder: Optional[torch.nn.Module] = None,
        budget_allocator: Optional[BudgetAllocator] = None,
        default_denoise_steps: int = 10,
        default_action_horizon: int = 16,
        device: str = "cuda",
    ):
        """
        Args:
            obs_encoder: maps raw observation dict to feature tensor [B, F].
                If None, concatenates available state vectors.
            budget_allocator: stage-aware budget allocator. If None, uses fixed budget.
            default_denoise_steps: N_d when no stage scheduling.
            default_action_horizon: N_a when no stage scheduling.
            device: torch device.
        """
        self.obs_encoder = obs_encoder
        self.budget_allocator = budget_allocator
        self.default_budget = RolloutBudget(
            num_denoise_steps=default_denoise_steps,
            action_horizon=default_action_horizon,
        )
        self.device = device

    def encode_obs(self, obs: dict) -> torch.Tensor:
        """Convert raw observation to feature tensor."""
        if self.obs_encoder is not None:
            # Assume obs_encoder handles dict input
            return self.obs_encoder(obs)

        # Default: concatenate state vectors
        parts = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, np.ndarray) and val.ndim == 1:
                parts.append(torch.tensor(val, dtype=torch.float32))
            elif isinstance(val, torch.Tensor) and val.dim() == 1:
                parts.append(val.float())
        if not parts:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        return torch.cat(parts).to(self.device)

    @torch.no_grad()
    def collect_trajectory(
        self,
        env,
        policy: ActionHead,
        task_decomposition: Optional[TaskDecomposition] = None,
        max_steps: int = 300,
    ) -> Trajectory:
        """Collect a single episode trajectory.

        Args:
            env: gymnasium-compatible environment.
            policy: action head with path recording.
            task_decomposition: optional HVTS decomposition for stage-aware budgets.
            max_steps: maximum episode length.

        Returns:
            Trajectory with recorded denoising paths.
        """
        obs, info = env.reset()
        traj = Trajectory(
            observations=[], actions=[], denoising_paths=[],
            rewards=[], obs_features=[],
        )

        step = 0
        done = False

        while not done and step < max_steps:
            # Determine budget for this step
            if task_decomposition is not None and self.budget_allocator is not None:
                stage = task_decomposition.get_stage_at_step(step)
                budget = self.budget_allocator.get_budget(stage)
            else:
                budget = self.default_budget

            # Encode observation
            obs_feat = self.encode_obs(obs).unsqueeze(0)  # [1, F]
            traj.obs_features.append(obs_feat.squeeze(0))

            # Generate action with path recording
            path = policy.predict_action_with_path(
                obs_feat,
                num_denoise_steps=budget.num_denoise_steps,
            )

            action_chunk = path.actions[0]  # [H, D]
            traj.observations.append(obs)
            traj.actions.append(action_chunk)
            traj.denoising_paths.append(path)
            traj.budgets_used.append(budget)

            # Execute action chunk in environment
            chunk_reward = 0.0
            for h in range(min(budget.action_horizon, action_chunk.shape[0])):
                if done:
                    break
                action = action_chunk[h].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                chunk_reward += reward
                step += 1
                done = terminated or truncated

            traj.rewards.append(chunk_reward)

        traj.total_reward = sum(traj.rewards)
        traj.success = traj.total_reward > 0.5  # Binary success
        traj.length = step

        return traj

    def collect_group(
        self,
        env,
        policy: ActionHead,
        group_size: int,
        task_decomposition: Optional[TaskDecomposition] = None,
        max_steps: int = 300,
        state_id: int = 0,
    ) -> RolloutBatch:
        """Collect a GRPO group of G trajectories from the same initial state.

        Args:
            env: gymnasium environment (will be reset for each trajectory).
            policy: action head.
            group_size: G, number of trajectories per group.
            task_decomposition: optional stage decomposition.
            max_steps: max episode length.
            state_id: identifier for the initial state (for hard-trajectory mining).

        Returns:
            RolloutBatch containing G trajectories.
        """
        trajectories = []
        for _ in range(group_size):
            traj = self.collect_trajectory(
                env, policy, task_decomposition, max_steps
            )
            trajectories.append(traj)

        return RolloutBatch(trajectories=trajectories, state_id=state_id)
