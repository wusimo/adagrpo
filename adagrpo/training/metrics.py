"""Metrics tracking for AdaGRPO training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class MetricsTracker:
    """Accumulate and summarise training metrics per epoch."""

    def __init__(self):
        self._buffers: dict[str, list[float]] = defaultdict(list)

    def update(self, key: str, value: float) -> None:
        self._buffers[key].append(value)

    def update_dict(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._buffers[k].append(v)

    def summarise(self, prefix: str = "") -> dict[str, float]:
        """Return mean of each metric and reset buffers."""
        summary = {}
        for k, vals in self._buffers.items():
            key = f"{prefix}{k}" if prefix else k
            summary[f"{key}/mean"] = float(np.mean(vals))
            if len(vals) > 1:
                summary[f"{key}/std"] = float(np.std(vals))
                summary[f"{key}/min"] = float(np.min(vals))
                summary[f"{key}/max"] = float(np.max(vals))
        self._buffers.clear()
        return summary

    def get(self, key: str) -> list[float]:
        return self._buffers.get(key, [])


def compute_success_rate(rewards: list[float], threshold: float = 0.5) -> float:
    """Fraction of episodes with reward above threshold."""
    if not rewards:
        return 0.0
    return sum(1.0 for r in rewards if r > threshold) / len(rewards)


def compute_ratio_diagnostics(
    log_ratios: Any,  # torch.Tensor [B] or [B, K]
) -> dict[str, float]:
    """Compute diagnostic statistics for importance ratios."""
    import torch

    if not isinstance(log_ratios, torch.Tensor):
        log_ratios = torch.tensor(log_ratios)

    if log_ratios.dim() == 2:
        # Per-step: report both per-step and aggregated
        per_step_var = log_ratios.var(dim=0).mean().item()
        agg = log_ratios.sum(dim=-1)
    else:
        per_step_var = 0.0
        agg = log_ratios

    ratios = torch.exp(agg.clamp(-10, 10))
    return {
        "log_ratio/mean": agg.mean().item(),
        "log_ratio/var": agg.var().item(),
        "log_ratio/per_step_var": per_step_var,
        "ratio/mean": ratios.mean().item(),
        "ratio/std": ratios.std().item(),
        "ratio/max": ratios.max().item(),
        "ratio/min": ratios.min().item(),
    }
