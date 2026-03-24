#!/usr/bin/env python3
"""Phase 1: Imitation Learning pre-training with optional ALN co-training.

Usage:
    python scripts/train_il.py env=libero policy=diffusion_cnn
    python scripts/train_il.py env=robomimic policy=diffusion_transformer
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

from adagrpo.core.aln import AdaptiveLossNetwork
from adagrpo.policy.diffusion_policy import DiffusionPolicy
from adagrpo.policy.flow_policy import FlowMatchingPolicy
from adagrpo.training.il_trainer import ILTrainer
from adagrpo.utils.logging import setup_logger

logger = setup_logger("adagrpo.train_il")


def build_policy(cfg: DictConfig):
    """Instantiate policy based on config."""
    if cfg.policy.type == "diffusion":
        return DiffusionPolicy(
            action_dim=cfg.env.action_dim,
            obs_dim=cfg.env.obs_dim,
            action_horizon=cfg.env.action_horizon,
            hidden_dim=cfg.policy.hidden_dim,
            num_train_timesteps=cfg.policy.num_train_timesteps,
            num_inference_steps=cfg.policy.num_inference_steps,
            schedule=cfg.policy.schedule,
            ddim_eta=cfg.policy.ddim_eta,
        )
    elif cfg.policy.type == "flow":
        return FlowMatchingPolicy(
            action_dim=cfg.env.action_dim,
            obs_dim=cfg.env.obs_dim,
            action_horizon=cfg.env.action_horizon,
            hidden_dim=cfg.policy.hidden_dim,
            num_inference_steps=cfg.policy.num_inference_steps,
            sigma_min=cfg.policy.sigma_min,
        )
    else:
        raise ValueError(f"Unknown policy type: {cfg.policy.type}")


def build_demo_dataloader(cfg: DictConfig) -> DataLoader:
    """Build a dataloader from demonstration data.

    For prototyping, generates random data. Replace with actual demo loading.
    """
    logger.info("Building demonstration dataloader (using random data for prototyping)")
    num_demos = 1000
    obs_features = torch.randn(num_demos, cfg.env.obs_dim)
    actions = torch.randn(num_demos, cfg.env.action_horizon, cfg.env.action_dim)
    dataset = TensorDataset(obs_features, actions)

    # Wrap to return dict
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, tensor_dataset):
            self.dataset = tensor_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            obs, act = self.dataset[idx]
            return {"obs_features": obs, "actions": act}

    return DataLoader(
        DictDataset(dataset),
        batch_size=cfg.policy.il.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device = cfg.device

    # Build policy
    policy = build_policy(cfg)

    # Build ALN (optional)
    aln = None
    if cfg.algo.get("use_aln", False):
        aln = AdaptiveLossNetwork(
            num_timesteps=cfg.policy.get("num_train_timesteps", 100),
            embed_dim=cfg.algo.aln.embed_dim,
            hidden_dim=cfg.algo.aln.hidden_dim,
        )
        logger.info("ALN co-training enabled.")

    # Build dataloader
    dataloader = build_demo_dataloader(cfg)

    # Train
    trainer = ILTrainer(
        policy=policy,
        aln=aln,
        lr=cfg.policy.il.lr,
        aln_lr=cfg.policy.il.aln_lr,
        weight_decay=cfg.policy.il.weight_decay,
        aln_warmup_steps=cfg.policy.il.aln_warmup_steps,
        num_epochs=cfg.policy.il.num_epochs,
        save_dir=cfg.checkpoint.save_dir + "/il",
        wandb_project=cfg.wandb.project,
        wandb_enabled=cfg.wandb.enabled,
        device=device,
    )
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
