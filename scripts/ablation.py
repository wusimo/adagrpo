#!/usr/bin/env python3
"""Run ablation experiments.

Runs the following ablation variants:
1. Full AdaGRPO (all components)
2. No ALN (uniform path-conditioned ratio)
3. No hard-trajectory mining (uniform group sampling)
4. No HVTS (fixed denoising budget)
5. No auxiliary loss (λ=0)

Usage:
    python scripts/ablation.py env=libero
"""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from adagrpo.utils.logging import setup_logger

logger = setup_logger("adagrpo.ablation")

ABLATION_CONFIGS = {
    "full_adagrpo": {},
    "no_aln": {"algo.use_aln": False},
    "no_mining": {"algo.use_hard_mining": False},
    "no_hvts": {"algo.use_hvts": False},
    "no_aux_loss": {"algo.aux_weight": 0.0},
    "no_aln_no_mining": {"algo.use_aln": False, "algo.use_hard_mining": False},
    "vanilla_grpo": {
        "algo.use_aln": False,
        "algo.use_hard_mining": False,
        "algo.use_hvts": False,
    },
}


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Running ablation experiments for %s / %s", cfg.env.name, cfg.env.task_name)

    base_save_dir = Path(cfg.checkpoint.save_dir)

    for ablation_name, overrides in ABLATION_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Starting ablation: %s", ablation_name)
        logger.info("  Overrides: %s", overrides)

        # Build command
        cmd = [
            sys.executable, "scripts/train_rl.py",
            f"env={cfg.env.name}",
            f"wandb.name={ablation_name}",
            f"checkpoint.save_dir={base_save_dir / ablation_name}",
        ]
        for key, val in overrides.items():
            cmd.append(f"{key}={val}")

        logger.info("  Command: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("  Completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error("  Failed with code %d: %s", e.returncode, e.stderr[-500:] if e.stderr else "")

    logger.info("All ablation experiments complete.")


if __name__ == "__main__":
    main()
