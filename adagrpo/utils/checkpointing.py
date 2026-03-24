"""Checkpoint save / load helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("adagrpo")


def save_checkpoint(
    path: str | Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    logger.info("Saved checkpoint to %s (iter %d)", path, iteration)


def load_checkpoint(
    path: str | Path,
    policy: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    policy.load_state_dict(state["policy"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    logger.info("Loaded checkpoint from %s (iter %d)", path, state.get("iteration", -1))
    return state
