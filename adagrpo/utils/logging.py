"""Logging utilities: console logger and Weights & Biases integration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

LOG_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"


def setup_logger(
    name: str = "adagrpo",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class WandbLogger:
    """Thin wrapper around wandb for optional dependency."""

    def __init__(
        self,
        project: str = "adagrpo",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None
        if enabled:
            try:
                import wandb

                self._run = wandb.init(project=project, name=name, config=config)
            except ImportError:
                logging.getLogger("adagrpo").warning(
                    "wandb not installed; logging disabled."
                )
                self.enabled = False

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.log(data, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.finish()
