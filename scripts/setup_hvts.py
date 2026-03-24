#!/usr/bin/env python3
"""Phase 2: Set up HVTS task decomposition and distil stage classifier.

Usage:
    python scripts/setup_hvts.py env=libero algo.hvts.use_vlm=false
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from adagrpo.scheduling.budget_allocator import BudgetAllocator
from adagrpo.scheduling.hvts import HierarchicalVisionTaskSegmenter
from adagrpo.utils.logging import setup_logger

logger = setup_logger("adagrpo.setup_hvts")


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Setting up HVTS for %s / %s", cfg.env.name, cfg.env.task_name)

    # Build HVTS
    hvts = HierarchicalVisionTaskSegmenter(
        use_vlm=cfg.algo.hvts.use_vlm,
        vlm_model_name=cfg.algo.hvts.vlm_model_name,
        device=cfg.device,
    )

    # Decompose task
    task_instruction = cfg.env.task_name.replace("_", " ")
    decomposition = hvts.decompose(
        task_instruction=task_instruction,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    # Print decomposition
    logger.info("Task: %s", decomposition.task_instruction)
    logger.info("Stages:")
    for s in decomposition.stages:
        logger.info(
            "  [%s] %s (%s) steps %d-%d",
            s.complexity.name, s.name, s.description,
            s.start_step or 0, s.end_step or 0,
        )

    # Compute savings
    allocator = BudgetAllocator()
    savings = allocator.compute_savings(decomposition.stages)
    logger.info("Estimated compute savings: %.1fx", savings["total_savings_ratio"])

    # Save decomposition
    save_path = Path(cfg.checkpoint.save_dir) / "hvts_decomposition.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    decomp_data = {
        "task_instruction": decomposition.task_instruction,
        "stages": [
            {
                "name": s.name,
                "description": s.description,
                "complexity": s.complexity.name,
                "start_step": s.start_step,
                "end_step": s.end_step,
            }
            for s in decomposition.stages
        ],
        "savings": savings,
    }
    with open(save_path, "w") as f:
        json.dump(decomp_data, f, indent=2)
    logger.info("Saved decomposition to %s", save_path)


if __name__ == "__main__":
    main()
