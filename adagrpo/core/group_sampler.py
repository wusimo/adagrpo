"""Hard-Trajectory Mining for GRPO group construction.

Standard GRPO samples groups uniformly from the state distribution. Many groups
end up with all-success or all-fail outcomes, providing zero contrastive signal.
Hard-trajectory mining biases sampling toward states with historically mixed
outcomes, maximising information per group.

Implementation: maintain a difficulty buffer that tracks per-state success rates
from recent rollouts. States with success rate near 0.5 are the most informative
and receive the highest sampling weight.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class StateRecord:
    """Track success history for a given initial state."""
    successes: int = 0
    failures: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # Unknown → maximally uncertain
        return self.successes / self.total

    @property
    def difficulty_weight(self) -> float:
        """Weight based on proximity to 0.5 success rate.

        States with ~50% success rate are maximally informative for GRPO
        because they produce groups with mixed outcomes.
        """
        sr = self.success_rate
        # Peaked at 0.5, zero at 0 and 1. Using 4*p*(1-p) which is 1 at 0.5.
        return 4.0 * sr * (1.0 - sr)


class HardTrajectoryMiner:
    """Difficulty-weighted state sampler for GRPO group construction.

    Maintains a rolling buffer of per-state success/failure counts and
    provides weighted sampling that biases toward states with mixed outcomes.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        min_history: int = 2,
        temperature: float = 1.0,
        uniform_mix: float = 0.2,
        decay: float = 0.99,
    ):
        """
        Args:
            buffer_size:  maximum number of distinct states tracked.
            min_history:  minimum rollouts per state before it gets a
                          difficulty-based weight (otherwise uniform).
            temperature:  sharpness of the difficulty-based distribution.
            uniform_mix:  fraction of sampling budget reserved for uniform
                          exploration (prevents starvation of easy/hard states).
            decay:        exponential decay applied to old counts each epoch.
        """
        self.buffer_size = buffer_size
        self.min_history = min_history
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.decay = decay

        self._records: dict[int, StateRecord] = defaultdict(StateRecord)
        self._state_ids: list[int] = []

    def update(self, state_id: int, success: bool) -> None:
        """Record an episode outcome for a given initial state."""
        if state_id not in self._records:
            if len(self._state_ids) >= self.buffer_size:
                # Evict oldest
                oldest = self._state_ids.pop(0)
                del self._records[oldest]
            self._state_ids.append(state_id)

        rec = self._records[state_id]
        if success:
            rec.successes += 1
        else:
            rec.failures += 1

    def sample_states(
        self,
        n: int,
        available_state_ids: Optional[list[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list[int]:
        """Sample n initial states weighted by difficulty.

        Args:
            n: number of states to sample.
            available_state_ids: pool of state IDs to sample from.
                If None, samples from all tracked states.
            rng: numpy random generator.

        Returns:
            List of n state IDs.
        """
        if rng is None:
            rng = np.random.default_rng()

        if available_state_ids is None:
            available_state_ids = list(self._records.keys())

        if len(available_state_ids) == 0:
            raise ValueError("No states available for sampling.")

        # Compute weights
        weights = np.array([
            self._records[sid].difficulty_weight
            if (sid in self._records and self._records[sid].total >= self.min_history)
            else 1.0  # uniform for unknown states
            for sid in available_state_ids
        ])

        # Apply temperature
        weights = weights ** (1.0 / self.temperature)

        # Mix with uniform
        uniform = np.ones_like(weights) / len(weights)
        weights = weights / (weights.sum() + 1e-8)
        mixed = (1 - self.uniform_mix) * weights + self.uniform_mix * uniform
        mixed = mixed / mixed.sum()

        indices = rng.choice(len(available_state_ids), size=n, p=mixed, replace=True)
        return [available_state_ids[i] for i in indices]

    def apply_decay(self) -> None:
        """Apply exponential decay to all counts (call once per epoch)."""
        for rec in self._records.values():
            rec.successes = int(rec.successes * self.decay)
            rec.failures = int(rec.failures * self.decay)

    def get_stats(self) -> dict[str, float]:
        """Return summary statistics about the difficulty buffer."""
        if not self._records:
            return {"num_states": 0, "mean_difficulty": 0.0, "mean_success_rate": 0.5}
        weights = [r.difficulty_weight for r in self._records.values()]
        srs = [r.success_rate for r in self._records.values()]
        return {
            "num_states": len(self._records),
            "mean_difficulty": float(np.mean(weights)),
            "mean_success_rate": float(np.mean(srs)),
        }
