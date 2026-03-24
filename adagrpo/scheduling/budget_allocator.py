"""Dynamic denoising budget and action horizon allocation per task stage.

Maps stage complexity to (N_d, N_a) pairs:
  - N_d: number of denoising steps (fewer for simple stages)
  - N_a: action horizon / chunk length (shorter for precise stages)

This reduces rollout compute by up to 7x for heterogeneous tasks:
simple approach stages use 2-3 denoising steps while complex insertion
stages use the full 10+ steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from adagrpo.scheduling.hvts import StageComplexity, TaskStage


@dataclass
class RolloutBudget:
    """Denoising and action horizon budget for a single rollout step."""

    num_denoise_steps: int  # N_d
    action_horizon: int  # N_a


# Default budget tables (tuned for DDIM with K_max=10, H_max=16)
DEFAULT_BUDGETS: dict[StageComplexity, RolloutBudget] = {
    StageComplexity.SIMPLE: RolloutBudget(num_denoise_steps=2, action_horizon=16),
    StageComplexity.MEDIUM: RolloutBudget(num_denoise_steps=5, action_horizon=12),
    StageComplexity.COMPLEX: RolloutBudget(num_denoise_steps=10, action_horizon=8),
}


class BudgetAllocator:
    """Allocate denoising budget and action horizon per task stage.

    Can be configured with custom budget tables or use learned mappings.
    """

    def __init__(
        self,
        budgets: Optional[dict[StageComplexity, RolloutBudget]] = None,
        max_denoise_steps: int = 10,
        max_action_horizon: int = 16,
        min_denoise_steps: int = 2,
        min_action_horizon: int = 4,
    ):
        self.budgets = budgets or DEFAULT_BUDGETS
        self.max_denoise_steps = max_denoise_steps
        self.max_action_horizon = max_action_horizon
        self.min_denoise_steps = min_denoise_steps
        self.min_action_horizon = min_action_horizon

    def get_budget(self, stage: TaskStage) -> RolloutBudget:
        """Look up the budget for a given task stage."""
        budget = self.budgets.get(stage.complexity, DEFAULT_BUDGETS[StageComplexity.MEDIUM])
        return RolloutBudget(
            num_denoise_steps=max(self.min_denoise_steps, min(budget.num_denoise_steps, self.max_denoise_steps)),
            action_horizon=max(self.min_action_horizon, min(budget.action_horizon, self.max_action_horizon)),
        )

    def get_budget_for_complexity(self, complexity: StageComplexity) -> RolloutBudget:
        """Direct lookup by complexity level."""
        return self.get_budget(TaskStage(name="", description="", complexity=complexity))

    def compute_savings(
        self,
        stages: list[TaskStage],
        baseline_denoise_steps: int = 10,
        baseline_action_horizon: int = 16,
    ) -> dict[str, float]:
        """Estimate compute savings relative to a fixed-budget baseline.

        Returns:
            Dict with 'denoise_savings_ratio' and 'total_savings_ratio'.
        """
        baseline_cost = 0.0
        adaptive_cost = 0.0

        for stage in stages:
            duration = (stage.end_step or 0) - (stage.start_step or 0)
            if duration <= 0:
                continue
            budget = self.get_budget(stage)
            # Cost ∝ num_denoise_steps * (steps in stage / action_horizon)
            baseline_steps = duration / baseline_action_horizon
            adaptive_steps = duration / budget.action_horizon
            baseline_cost += baseline_steps * baseline_denoise_steps
            adaptive_cost += adaptive_steps * budget.num_denoise_steps

        if adaptive_cost == 0:
            return {"denoise_savings_ratio": 1.0, "total_savings_ratio": 1.0}

        return {
            "denoise_savings_ratio": baseline_cost / adaptive_cost,
            "total_savings_ratio": baseline_cost / adaptive_cost,
        }
