"""Imitation Learning pre-training with optional ALN co-training.

Phase 1 of AdaGRPO: train the diffusion policy on demonstration data
using standard denoising loss, optionally co-training the ALN to learn
which denoising timesteps are most informative.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adagrpo.core.aln import AdaptiveLossNetwork, compute_aln_il_loss
from adagrpo.policy.action_head import ActionHead
from adagrpo.training.metrics import MetricsTracker
from adagrpo.utils.checkpointing import save_checkpoint
from adagrpo.utils.logging import WandbLogger

logger = logging.getLogger("adagrpo")


class ILTrainer:
    """Imitation learning trainer with ALN co-training.

    Trains a diffusion policy on demonstration data (state, action) pairs
    using the standard denoising objective. Optionally co-trains an ALN
    that learns per-timestep importance weights.
    """

    def __init__(
        self,
        policy: ActionHead,
        aln: Optional[AdaptiveLossNetwork] = None,
        lr: float = 1e-4,
        aln_lr: float = 3e-4,
        weight_decay: float = 1e-6,
        aln_warmup_steps: int = 500,
        num_epochs: int = 100,
        save_dir: str = "checkpoints/il",
        save_every: int = 10,
        wandb_project: str = "adagrpo",
        wandb_enabled: bool = True,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.aln = aln.to(device) if aln is not None else None
        self.device = device
        self.num_epochs = num_epochs
        self.aln_warmup_steps = aln_warmup_steps
        self.save_dir = Path(save_dir)
        self.save_every = save_every

        self.policy_optimizer = torch.optim.AdamW(
            policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.aln_optimizer = (
            torch.optim.Adam(aln.parameters(), lr=aln_lr) if aln is not None else None
        )

        self.metrics = MetricsTracker()
        self.wandb = WandbLogger(
            project=wandb_project, name="il_pretrain", enabled=wandb_enabled
        )
        self._global_step = 0

    def train(self, dataloader: DataLoader) -> None:
        """Run IL pre-training.

        Expected batch format from dataloader:
            {"obs_features": [B, F], "actions": [B, H, D]}
        """
        logger.info("Starting IL pre-training for %d epochs", self.num_epochs)
        self.policy.train()
        if self.aln is not None:
            self.aln.train()

        for epoch in range(self.num_epochs):
            for batch in dataloader:
                obs = batch["obs_features"].to(self.device)
                actions = batch["actions"].to(self.device)

                # Policy denoising loss
                loss = self.policy.compute_denoising_loss(obs, actions)

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

                self.metrics.update("il/loss", loss.item())

                # ALN co-training (after warmup)
                if self.aln is not None and self._global_step > self.aln_warmup_steps:
                    self._train_aln_step(obs, actions)

                self._global_step += 1

            # Epoch summary
            summary = self.metrics.summarise(prefix="train/")
            summary["epoch"] = epoch
            self.wandb.log(summary, step=self._global_step)
            logger.info("Epoch %d | loss=%.4f", epoch, summary.get("train/il/loss/mean", 0))

            if (epoch + 1) % self.save_every == 0:
                extra = {}
                if self.aln is not None:
                    extra["aln"] = self.aln.state_dict()
                save_checkpoint(
                    self.save_dir / f"epoch_{epoch+1}.pt",
                    self.policy,
                    self.policy_optimizer,
                    epoch + 1,
                    extra=extra,
                )

        self.wandb.finish()
        logger.info("IL pre-training complete.")

    def _train_aln_step(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        """Single ALN co-training step.

        Compute per-timestep denoising losses and train the ALN to upweight
        informative timesteps.
        """
        B = actions.shape[0]
        K = self.aln.num_timesteps

        # Sample a set of timesteps and compute per-step losses
        num_sample_steps = min(K, 20)  # Sample a subset for efficiency
        timesteps = torch.randint(0, K, (num_sample_steps,), device=self.device)

        per_step_losses = []
        for t_val in timesteps:
            t = t_val.expand(B)
            noise = torch.randn_like(actions)
            noisy = self.policy.ddpm.q_sample(
                actions.view(B, -1), t, noise.view(B, -1)
            ).view_as(actions)
            noise_pred = self.policy.noise_net(noisy, t, obs)
            step_loss = (noise_pred - noise).pow(2).mean(dim=(1, 2))  # [B]
            per_step_losses.append(step_loss)

        per_step_losses = torch.stack(per_step_losses, dim=1)  # [B, num_sample_steps]

        aln_loss = compute_aln_il_loss(per_step_losses, self.aln, timesteps)

        self.aln_optimizer.zero_grad()
        aln_loss.backward()
        self.aln_optimizer.step()

        self.metrics.update("il/aln_loss", aln_loss.item())
