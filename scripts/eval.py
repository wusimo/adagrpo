#!/usr/bin/env python3
"""Evaluate a trained policy.

Usage:
    python scripts/eval.py env=libero checkpoint.resume_from=checkpoints/rl/iter_2000.pt
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from adagrpo.policy.diffusion_policy import DiffusionPolicy
from adagrpo.policy.flow_policy import FlowMatchingPolicy
from adagrpo.training.rollout import RolloutCollector
from adagrpo.training.metrics import compute_success_rate
from adagrpo.utils.checkpointing import load_checkpoint
from adagrpo.utils.logging import setup_logger

logger = setup_logger("adagrpo.eval")


def build_env(cfg):
    if cfg.env.name == "libero":
        from adagrpo.envs.libero_wrapper import make_libero_env
        return make_libero_env(cfg.env.task_name, cfg.env.suite, cfg.env.image_size, cfg.env.max_episode_steps, cfg.seed)
    elif cfg.env.name == "robomimic":
        from adagrpo.envs.robomimic_wrapper import make_robomimic_env
        return make_robomimic_env(cfg.env.task_name, cfg.env.image_size, cfg.env.max_episode_steps, cfg.seed)
    elif cfg.env.name == "maniskill":
        from adagrpo.envs.maniskill_wrapper import make_maniskill_env
        return make_maniskill_env(cfg.env.task_name, cfg.env.obs_mode, cfg.env.control_mode, cfg.env.image_size, cfg.env.max_episode_steps, cfg.seed)
    else:
        raise ValueError(f"Unknown env: {cfg.env.name}")


def build_policy(cfg):
    if cfg.policy.type == "diffusion":
        return DiffusionPolicy(
            action_dim=cfg.env.action_dim, obs_dim=cfg.env.obs_dim,
            action_horizon=cfg.env.action_horizon, hidden_dim=cfg.policy.hidden_dim,
            num_train_timesteps=cfg.policy.num_train_timesteps,
            num_inference_steps=cfg.policy.num_inference_steps,
            schedule=cfg.policy.schedule, ddim_eta=cfg.policy.ddim_eta,
        )
    elif cfg.policy.type == "flow":
        return FlowMatchingPolicy(
            action_dim=cfg.env.action_dim, obs_dim=cfg.env.obs_dim,
            action_horizon=cfg.env.action_horizon, hidden_dim=cfg.policy.hidden_dim,
            num_inference_steps=cfg.policy.num_inference_steps,
            sigma_min=cfg.policy.sigma_min,
        )
    else:
        raise ValueError(f"Unknown policy type: {cfg.policy.type}")


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Evaluation config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    num_episodes = cfg.algo.get("num_eval_episodes", 50)
    device = cfg.device

    env = build_env(cfg)
    policy = build_policy(cfg)

    if cfg.checkpoint.resume_from:
        load_checkpoint(cfg.checkpoint.resume_from, policy)

    policy = policy.to(device)
    policy.eval()

    collector = RolloutCollector(
        default_denoise_steps=cfg.env.default_denoise_steps,
        default_action_horizon=cfg.env.action_horizon,
        device=device,
    )

    rewards = []
    lengths = []

    for ep in range(num_episodes):
        traj = collector.collect_trajectory(
            env, policy, max_steps=cfg.env.max_episode_steps
        )
        rewards.append(traj.total_reward)
        lengths.append(traj.length)
        if (ep + 1) % 10 == 0:
            logger.info("Episode %d/%d: reward=%.3f len=%d", ep + 1, num_episodes, traj.total_reward, traj.length)

    success_rate = compute_success_rate(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)

    logger.info("=" * 50)
    logger.info("Results over %d episodes:", num_episodes)
    logger.info("  Success rate: %.2f%%", success_rate * 100)
    logger.info("  Mean reward:  %.4f ± %.4f", mean_reward, std_reward)
    logger.info("  Mean length:  %.1f", mean_length)
    logger.info("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
