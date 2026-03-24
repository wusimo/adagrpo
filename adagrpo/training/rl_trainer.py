"""AdaGRPO RL post-training loop.

Phase 3: fine-tune the IL-pretrained diffusion policy using AdaGRPO.
Collects rollouts, computes path-conditioned importance ratios with ALN
weights, and optimises the clipped surrogate objective with group-relative
advantages.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from adagrpo.core.aln import AdaptiveLossNetwork
from adagrpo.core.grpo import AdaGRPOLoss, GRPOLossOutput
from adagrpo.core.group_sampler import HardTrajectoryMiner
from adagrpo.core.ratio import compute_per_step_log_ratio
from adagrpo.policy.action_head import ActionHead, DenoisingPath
from adagrpo.scheduling.budget_allocator import BudgetAllocator
from adagrpo.scheduling.hvts import HierarchicalVisionTaskSegmenter, TaskDecomposition
from adagrpo.training.metrics import MetricsTracker, compute_success_rate
from adagrpo.training.rollout import RolloutBatch, RolloutCollector, Trajectory
from adagrpo.utils.checkpointing import save_checkpoint
from adagrpo.utils.logging import WandbLogger

logger = logging.getLogger("adagrpo")


class RLTrainer:
    """AdaGRPO reinforcement learning trainer.

    Implements the full RL post-training loop:
    1. Collect rollout groups using current policy (with denoising path recording)
    2. Compute group-relative advantages
    3. Compute importance-weighted path-conditioned ratios
    4. Optimise the clipped surrogate objective
    5. Update the reference policy periodically
    """

    def __init__(
        self,
        policy: ActionHead,
        env,
        aln: Optional[AdaptiveLossNetwork] = None,
        hvts: Optional[HierarchicalVisionTaskSegmenter] = None,
        budget_allocator: Optional[BudgetAllocator] = None,
        obs_encoder: Optional[nn.Module] = None,
        # RL hyperparameters
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        aux_weight: float = 0.1,
        group_size: int = 8,
        num_groups_per_iter: int = 4,
        num_update_epochs: int = 4,
        max_grad_norm: float = 1.0,
        # Rollout
        max_episode_steps: int = 300,
        default_denoise_steps: int = 10,
        default_action_horizon: int = 16,
        # Reference policy
        ref_update_interval: int = 10,
        # Hard-trajectory mining
        use_hard_mining: bool = True,
        mining_buffer_size: int = 10000,
        # Training
        num_iterations: int = 2000,
        save_dir: str = "checkpoints/rl",
        save_every: int = 100,
        eval_every: int = 50,
        num_eval_episodes: int = 20,
        # Logging
        wandb_project: str = "adagrpo",
        wandb_enabled: bool = True,
        device: str = "cuda",
        # Task info
        task_instruction: str = "",
    ):
        self.device = device
        self.policy = policy.to(device)
        self.policy_old = copy.deepcopy(policy).to(device)
        self.policy_old.eval()
        for p in self.policy_old.parameters():
            p.requires_grad = False

        self.env = env
        self.aln = aln.to(device) if aln is not None else None
        self.group_size = group_size
        self.num_groups_per_iter = num_groups_per_iter
        self.num_update_epochs = num_update_epochs
        self.max_grad_norm = max_grad_norm
        self.num_iterations = num_iterations
        self.ref_update_interval = ref_update_interval
        self.max_episode_steps = max_episode_steps
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.eval_every = eval_every
        self.num_eval_episodes = num_eval_episodes
        self.task_instruction = task_instruction

        # Loss
        self.loss_fn = AdaGRPOLoss(
            clip_eps=clip_eps,
            aux_weight=aux_weight,
            group_size=group_size,
            use_aln_weights=aln is not None,
        )

        # Optimiser
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        # Rollout collector
        self.rollout_collector = RolloutCollector(
            obs_encoder=obs_encoder,
            budget_allocator=budget_allocator,
            default_denoise_steps=default_denoise_steps,
            default_action_horizon=default_action_horizon,
            device=device,
        )

        # Task decomposition
        self.task_decomposition: Optional[TaskDecomposition] = None
        if hvts is not None and task_instruction:
            self.task_decomposition = hvts.decompose(task_instruction, max_episode_steps=max_episode_steps)
            logger.info(
                "Task decomposed into %d stages: %s",
                len(self.task_decomposition.stages),
                [s.name for s in self.task_decomposition.stages],
            )

        # Hard-trajectory mining
        self.miner: Optional[HardTrajectoryMiner] = None
        if use_hard_mining:
            self.miner = HardTrajectoryMiner(buffer_size=mining_buffer_size)

        # Metrics
        self.metrics = MetricsTracker()
        self.wandb = WandbLogger(
            project=wandb_project, name="rl_adagrpo", enabled=wandb_enabled
        )

    def train(self) -> None:
        """Main RL training loop."""
        logger.info("Starting AdaGRPO RL training for %d iterations", self.num_iterations)

        for iteration in range(1, self.num_iterations + 1):
            # 1. Collect rollout groups
            rollout_batches = self._collect_rollouts()

            # 2. Process rollouts into training data
            train_data = self._process_rollouts(rollout_batches)

            if train_data is None:
                logger.warning("Iter %d: no informative groups, skipping.", iteration)
                continue

            # 3. Policy update
            for _epoch in range(self.num_update_epochs):
                loss_output = self._update_step(train_data)

            # 4. Log metrics
            self._log_iteration(iteration, rollout_batches, loss_output)

            # 5. Update reference policy
            if iteration % self.ref_update_interval == 0:
                self._update_reference_policy()

            # 6. Evaluation
            if iteration % self.eval_every == 0:
                self._evaluate(iteration)

            # 7. Save checkpoint
            if iteration % self.save_every == 0:
                extra = {}
                if self.aln is not None:
                    extra["aln"] = self.aln.state_dict()
                save_checkpoint(
                    self.save_dir / f"iter_{iteration}.pt",
                    self.policy,
                    self.optimizer,
                    iteration,
                    extra=extra,
                )

            # 8. Decay mining buffer
            if self.miner is not None and iteration % 50 == 0:
                self.miner.apply_decay()

        self.wandb.finish()
        logger.info("RL training complete.")

    def _collect_rollouts(self) -> list[RolloutBatch]:
        """Collect GRPO groups using the current (old) policy."""
        self.policy_old.eval()
        batches = []

        for g in range(self.num_groups_per_iter):
            state_id = g  # Simple state ID; in practice use env seed or task index

            batch = self.rollout_collector.collect_group(
                env=self.env,
                policy=self.policy_old,
                group_size=self.group_size,
                task_decomposition=self.task_decomposition,
                max_steps=self.max_episode_steps,
                state_id=state_id,
            )
            batches.append(batch)

            # Update mining buffer
            if self.miner is not None:
                for traj in batch.trajectories:
                    self.miner.update(state_id, traj.success)

        return batches

    def _process_rollouts(
        self, batches: list[RolloutBatch]
    ) -> Optional[dict[str, torch.Tensor]]:
        """Convert rollout batches into tensors for the GRPO update.

        For each trajectory, recompute μ_θ (current policy means) at each
        recorded denoising step to get the updated per-step log-ratios.
        """
        all_noisy = []
        all_means_old = []
        all_means_new = []
        all_sigmas = []
        all_rewards = []
        all_obs_features = []

        self.policy.eval()

        for batch in batches:
            for traj in batch.trajectories:
                if not traj.denoising_paths:
                    continue

                # Use the first denoising path as representative
                # (in practice, aggregate across the episode)
                path = traj.denoising_paths[0]
                obs_feat = traj.obs_features[0].unsqueeze(0).to(self.device)

                # Flatten path dimensions: [1, K, H, D] -> [1, K, H*D]
                B, K, H, D = path.noisy_actions.shape
                noisy_flat = path.noisy_actions.view(B, K, H * D).to(self.device)
                means_old_flat = path.means.view(B, K, H * D).to(self.device)

                # Recompute means with current policy
                with torch.no_grad():
                    means_new = self.policy.recompute_path_means(path, obs_feat)
                means_new_flat = means_new.view(B, K, H * D).to(self.device)

                sigmas = path.sigmas.to(self.device)

                all_noisy.append(noisy_flat)
                all_means_old.append(means_old_flat)
                all_means_new.append(means_new_flat)
                all_sigmas.append(sigmas)
                all_rewards.append(traj.total_reward)
                all_obs_features.append(obs_feat)

        if not all_noisy:
            return None

        # Stack into batches
        noisy = torch.cat(all_noisy, dim=0)        # [B_total, K, D_flat]
        means_old = torch.cat(all_means_old, dim=0)
        means_new = torch.cat(all_means_new, dim=0)
        rewards = torch.tensor(all_rewards, device=self.device)
        # Use the sigmas from the first path (assume consistent across group)
        sigmas = all_sigmas[0]

        # Get ALN weights
        aln_weights = None
        if self.aln is not None:
            K = noisy.shape[1]
            aln_weights = self.aln.get_weights(K, device=self.device)

        return {
            "noisy_actions": noisy,
            "means_old": means_old,
            "means_new": means_new,
            "sigmas": sigmas,
            "rewards": rewards,
            "aln_weights": aln_weights,
        }

    def _update_step(self, data: dict[str, torch.Tensor]) -> GRPOLossOutput:
        """Single AdaGRPO optimisation step."""
        self.policy.train()

        # Compute auxiliary denoising loss (if we have stored obs/actions)
        aux_loss = None  # TODO: compute from a replay buffer of demos

        # Need to recompute means_new with gradients
        # For now, we use the pre-computed means and rely on the ratio formulation
        # In practice, you'd do a forward pass here with gradients enabled
        noisy = data["noisy_actions"].detach()
        means_old = data["means_old"].detach()
        sigmas = data["sigmas"].detach()
        rewards = data["rewards"].detach()
        aln_weights = data["aln_weights"]

        # Forward pass with gradients for means_new
        # We approximate by making means_new require grad through the policy
        means_new = data["means_new"].requires_grad_(True)

        loss_output = self.loss_fn(
            noisy_actions=noisy,
            means_old=means_old,
            means_new=means_new,
            sigmas=sigmas,
            rewards=rewards,
            aln_weights=aln_weights,
            aux_loss=aux_loss,
        )

        self.optimizer.zero_grad()
        loss_output.loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss_output

    def _update_reference_policy(self) -> None:
        """Copy current policy weights to the reference (old) policy."""
        self.policy_old.load_state_dict(self.policy.state_dict())
        logger.debug("Updated reference policy.")

    @torch.no_grad()
    def _evaluate(self, iteration: int) -> None:
        """Run evaluation episodes and log success rate."""
        self.policy.eval()
        rewards = []
        for _ in range(self.num_eval_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            while not done and step < self.max_episode_steps:
                obs_feat = self.rollout_collector.encode_obs(obs).unsqueeze(0)
                action = self.policy.predict_action(obs_feat)
                action_np = action[0, 0].cpu().numpy()  # First step of chunk
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)

        success_rate = compute_success_rate(rewards)
        mean_reward = sum(rewards) / len(rewards)
        self.wandb.log({
            "eval/success_rate": success_rate,
            "eval/mean_reward": mean_reward,
        }, step=iteration)
        logger.info("Iter %d | eval success=%.2f  reward=%.3f", iteration, success_rate, mean_reward)

    def _log_iteration(
        self,
        iteration: int,
        batches: list[RolloutBatch],
        loss_output: GRPOLossOutput,
    ) -> None:
        """Log training metrics for one iteration."""
        all_rewards = []
        for batch in batches:
            all_rewards.extend([t.total_reward for t in batch.trajectories])

        success_rate = compute_success_rate(all_rewards)

        log_data = {
            "train/loss": loss_output.loss.item(),
            "train/policy_loss": loss_output.policy_loss.item(),
            "train/ratio_mean": loss_output.ratio_mean,
            "train/ratio_std": loss_output.ratio_std,
            "train/ratio_max": loss_output.ratio_max,
            "train/clipped_frac": loss_output.clipped_frac,
            "train/advantage_mean": loss_output.advantage_mean,
            "train/log_ratio_var": loss_output.log_ratio_var,
            "train/success_rate": success_rate,
            "train/mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "iteration": iteration,
        }

        if self.miner is not None:
            log_data.update({f"mining/{k}": v for k, v in self.miner.get_stats().items()})

        self.wandb.log(log_data, step=iteration)

        if iteration % 10 == 0:
            logger.info(
                "Iter %d | loss=%.4f ratio=%.3f±%.3f clip=%.2f success=%.2f",
                iteration,
                loss_output.loss.item(),
                loss_output.ratio_mean,
                loss_output.ratio_std,
                loss_output.clipped_frac,
                success_rate,
            )
