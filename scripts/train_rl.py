#!/usr/bin/env python3
"""Phase 3: AdaGRPO RL post-training.

Usage:
    python scripts/train_rl.py algo=adagrpo env=libero policy=diffusion_cnn
    python scripts/train_rl.py algo=vanilla_grpo env=libero policy=diffusion_cnn
    python scripts/train_rl.py algo=adagrpo env=libero checkpoint.resume_from=checkpoints/il/epoch_100.pt
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from adagrpo.core.aln import AdaptiveLossNetwork
from adagrpo.envs.libero_wrapper import make_libero_env
from adagrpo.envs.robomimic_wrapper import make_robomimic_env
from adagrpo.envs.maniskill_wrapper import make_maniskill_env
from adagrpo.policy.diffusion_policy import DiffusionPolicy
from adagrpo.policy.flow_policy import FlowMatchingPolicy
from adagrpo.scheduling.budget_allocator import BudgetAllocator
from adagrpo.scheduling.hvts import HierarchicalVisionTaskSegmenter
from adagrpo.training.rl_trainer import RLTrainer
from adagrpo.utils.checkpointing import load_checkpoint
from adagrpo.utils.logging import setup_logger

logger = setup_logger("adagrpo.train_rl")


def build_env(cfg: DictConfig):
    """Instantiate environment based on config."""
    if cfg.env.name == "libero":
        return make_libero_env(
            task_name=cfg.env.task_name,
            suite=cfg.env.suite,
            image_size=cfg.env.image_size,
            max_episode_steps=cfg.env.max_episode_steps,
            seed=cfg.seed,
        )
    elif cfg.env.name == "robomimic":
        return make_robomimic_env(
            task_name=cfg.env.task_name,
            image_size=cfg.env.image_size,
            max_episode_steps=cfg.env.max_episode_steps,
            seed=cfg.seed,
        )
    elif cfg.env.name == "maniskill":
        return make_maniskill_env(
            task_name=cfg.env.task_name,
            obs_mode=cfg.env.obs_mode,
            control_mode=cfg.env.control_mode,
            image_size=cfg.env.image_size,
            max_episode_steps=cfg.env.max_episode_steps,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown env: {cfg.env.name}")


def build_policy(cfg: DictConfig):
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


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    # Build components
    env = build_env(cfg)
    policy = build_policy(cfg)

    # Load IL checkpoint if available
    if cfg.checkpoint.resume_from is not None:
        logger.info("Loading IL checkpoint: %s", cfg.checkpoint.resume_from)
        load_checkpoint(cfg.checkpoint.resume_from, policy)

    # ALN
    aln = None
    if cfg.algo.get("use_aln", False):
        aln = AdaptiveLossNetwork(
            num_timesteps=cfg.policy.get("num_train_timesteps", 100),
            embed_dim=cfg.algo.aln.embed_dim,
            hidden_dim=cfg.algo.aln.hidden_dim,
        )
        # Load ALN weights from IL checkpoint if available
        if cfg.checkpoint.resume_from is not None:
            state = torch.load(cfg.checkpoint.resume_from, map_location="cpu", weights_only=False)
            if "aln" in state:
                aln.load_state_dict(state["aln"])
                logger.info("Loaded ALN weights from checkpoint.")

    # HVTS
    hvts = None
    budget_allocator = None
    if cfg.algo.get("use_hvts", False):
        hvts = HierarchicalVisionTaskSegmenter(
            use_vlm=cfg.algo.hvts.use_vlm,
            vlm_model_name=cfg.algo.hvts.vlm_model_name,
            device=cfg.device,
        )
        budget_allocator = BudgetAllocator()

    # Task instruction
    task_instruction = cfg.env.task_name.replace("_", " ")

    # Build trainer
    trainer = RLTrainer(
        policy=policy,
        env=env,
        aln=aln,
        hvts=hvts,
        budget_allocator=budget_allocator,
        lr=cfg.algo.lr,
        clip_eps=cfg.algo.clip_eps,
        aux_weight=cfg.algo.aux_weight,
        group_size=cfg.algo.group_size,
        num_groups_per_iter=cfg.algo.num_groups_per_iter,
        num_update_epochs=cfg.algo.num_update_epochs,
        max_grad_norm=cfg.algo.max_grad_norm,
        max_episode_steps=cfg.env.max_episode_steps,
        default_denoise_steps=cfg.env.default_denoise_steps,
        default_action_horizon=cfg.env.action_horizon,
        ref_update_interval=cfg.algo.ref_update_interval,
        use_hard_mining=cfg.algo.get("use_hard_mining", False),
        num_iterations=cfg.algo.num_iterations,
        save_dir=cfg.checkpoint.save_dir + "/rl",
        save_every=cfg.checkpoint.save_every,
        eval_every=cfg.algo.eval_every,
        num_eval_episodes=cfg.algo.num_eval_episodes,
        wandb_project=cfg.wandb.project,
        wandb_enabled=cfg.wandb.enabled,
        device=cfg.device,
        task_instruction=task_instruction,
    )

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
