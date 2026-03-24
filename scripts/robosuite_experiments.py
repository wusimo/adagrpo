#!/usr/bin/env python3
"""Full experiment suite on robosuite tasks.

Runs AdaGRPO, DPPO baseline, and SFT-only on Lift, Door, NutAssemblySquare.
Collects learning curves, ratio diagnostics, and success rates.

Usage: /home/simo/miniconda3/envs/libero/bin/python scripts/robosuite_experiments.py
"""

import copy, json, os, time, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/robosuite")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")

# ============================================================
# Diffusion Policy (reusable)
# ============================================================

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class NoiseNet(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden=256):
        super().__init__()
        self.t_emb = SinusoidalEmbed(hidden)
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden * 2, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x_t, t, obs):
        return self.net(torch.cat([x_t, self.t_emb(t), self.obs_proj(obs)], dim=-1))

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    ac = torch.cos((t + s) / (1 + s) * np.pi / 2).pow(2)
    ac = ac / ac[0]
    return (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)

class DiffPolicy:
    def __init__(self, action_dim, obs_dim, T=50, K=5, hidden=256, eta=1.0, sigma_min=0.05):
        self.action_dim, self.obs_dim, self.T, self.K = action_dim, obs_dim, T, K
        self.eta, self.sigma_min, self.hidden = eta, sigma_min, hidden
        self.noise_net = NoiseNet(action_dim, obs_dim, hidden).to(DEVICE)
        betas = cosine_beta_schedule(T).to(DEVICE)
        self.ac = torch.cumprod(1 - betas, 0)
        step_ratio = T // K
        self.timesteps = (torch.arange(0, K) * step_ratio).long().flip(0).to(DEVICE)

    def parameters(self): return self.noise_net.parameters()
    def state_dict(self): return self.noise_net.state_dict()
    def load_state_dict(self, sd): self.noise_net.load_state_dict(sd)
    def train(self): self.noise_net.train()
    def eval(self): self.noise_net.eval()

    def _step_params(self, idx):
        t = self.timesteps[idx]
        t_prev = self.timesteps[idx+1] if idx+1 < self.K else torch.tensor(0, device=DEVICE)
        ac_t = self.ac[t]
        ac_prev = self.ac[t_prev] if t_prev > 0 else torch.tensor(1.0, device=DEVICE)
        pvar = ((1-ac_prev)/(1-ac_t+1e-8)*(1-ac_t/(ac_prev+1e-8))).clamp(min=1e-8)
        return ac_t, ac_prev, max(self.eta * pvar.sqrt().item(), self.sigma_min), t

    def _mean(self, x_t, noise_pred, ac_t, ac_prev, sigma):
        pred_x0 = (x_t - (1-ac_t).sqrt() * noise_pred) / ac_t.sqrt()
        dir_xt = (1-ac_prev-sigma**2).clamp(min=0).sqrt() * noise_pred
        return ac_prev.sqrt() * pred_x0 + dir_xt

    def denoising_loss(self, obs, actions):
        B = actions.shape[0]
        t = torch.randint(0, self.T, (B,), device=DEVICE)
        noise = torch.randn_like(actions)
        ac_ = self.ac[t].unsqueeze(-1)
        return F.mse_loss(self.noise_net(ac_.sqrt()*actions+(1-ac_).sqrt()*noise, t, obs), noise)

    @torch.no_grad()
    def sample(self, obs):
        B = obs.shape[0]; x = torch.randn(B, self.action_dim, device=DEVICE)
        for i in range(self.K):
            ac_t, ac_prev, sigma, t = self._step_params(i)
            x = self._mean(x, self.noise_net(x, t.expand(B), obs), ac_t, ac_prev, sigma)
        return x

    @torch.no_grad()
    def sample_with_path(self, obs):
        B = obs.shape[0]; x = torch.randn(B, self.action_dim, device=DEVICE)
        inp, out, means, sigs = [], [], [], []
        for i in range(self.K):
            ac_t, ac_prev, sigma, t = self._step_params(i)
            inp.append(x.clone())
            mean = self._mean(x, self.noise_net(x, t.expand(B), obs), ac_t, ac_prev, sigma)
            x_next = (mean + sigma * torch.randn_like(x)) if i < self.K-1 else mean
            out.append(x_next.clone()); means.append(mean); sigs.append(sigma); x = x_next
        return x, torch.stack(inp,1), torch.stack(out,1), torch.stack(means,1), torch.tensor(sigs, device=DEVICE)

    def compute_mean_at_step(self, a_k, idx, obs):
        ac_t, ac_prev, sigma, t = self._step_params(idx)
        return self._mean(a_k, self.noise_net(a_k, t.expand(a_k.shape[0]), obs), ac_t, ac_prev, sigma)

    def clone(self):
        p = DiffPolicy(self.action_dim, self.obs_dim, self.T, self.K, self.hidden, self.eta, self.sigma_min)
        p.load_state_dict(copy.deepcopy(self.state_dict()))
        return p


# ============================================================
# Robosuite wrapper
# ============================================================

class RobosuiteEnv:
    """Lightweight robosuite wrapper for state-based RL."""
    def __init__(self, task_name="Lift", max_steps=200):
        import robosuite as suite
        self.env = suite.make(
            task_name, robots="Panda",
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, reward_shaping=True,
            control_freq=20, horizon=max_steps,
        )
        self.max_steps = max_steps
        self.action_dim = self.env.action_dim
        # Get obs dim from a reset
        obs = self.env.reset()
        self._obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object-state']
        self.obs_dim = sum(obs[k].shape[0] for k in self._obs_keys if k in obs)
        self.task_name = task_name

    def _flatten_obs(self, obs):
        parts = [obs[k] for k in self._obs_keys if k in obs]
        return np.concatenate(parts).astype(np.float32)

    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), float(reward), done, info

    def collect_demos(self, n_episodes=100, noise_std=0.3):
        """Collect demonstrations using shaped reward as guidance (noisy random)."""
        demos = []
        for ep in range(n_episodes):
            obs = self.reset()
            ep_obs, ep_acts = [], []
            for step in range(self.max_steps):
                # Simple heuristic: move toward object then up
                action = np.random.randn(self.action_dim).astype(np.float32) * noise_std
                ep_obs.append(obs)
                ep_acts.append(action)
                obs, reward, done, info = self.step(action)
                if done:
                    break
            demos.append({
                'obs': np.array(ep_obs),
                'actions': np.array(ep_acts),
                'length': len(ep_obs),
            })
        return demos

    def rollout(self, policy, n_episodes=20):
        """Evaluate policy, return per-episode rewards and success."""
        results = []
        policy.eval()
        for ep in range(n_episodes):
            obs = self.reset()
            total_reward = 0.0
            success = False
            for step in range(self.max_steps):
                obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action = policy.sample(obs_t)[0].cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                obs, reward, done, info = self.step(action)
                total_reward += reward
                if info.get('is_success', False) or (hasattr(self.env, 'check_success') and self.env._check_success()):
                    success = True
                if done:
                    break
            results.append({'reward': total_reward, 'success': success})
        return results

    def rollout_group(self, policy, G=8):
        """Rollout G episodes (group) for GRPO, recording denoising paths."""
        policy.eval()
        group_data = []
        for i in range(G):
            obs = self.reset()
            total_reward = 0.0
            ep_data = {'obs': [], 'paths': [], 'rewards': []}
            for step in range(self.max_steps):
                obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action, inp, out, means, sigmas = policy.sample_with_path(obs_t)
                ep_data['obs'].append(obs_t)
                ep_data['paths'].append((inp, out, means, sigmas))
                a_np = action[0].cpu().numpy()
                a_np = np.clip(a_np, -1.0, 1.0)
                obs, reward, done, info = self.step(a_np)
                total_reward += reward
                ep_data['rewards'].append(reward)
                if done:
                    break
            ep_data['total_reward'] = total_reward
            group_data.append(ep_data)
        return group_data


# ============================================================
# IL Pretraining (from random exploration demos)
# ============================================================

def il_pretrain(policy, env, num_steps=5000, batch_size=64, lr=1e-3):
    """Pretrain on demos collected from random exploration."""
    print(f"  Collecting demos...")
    demos = env.collect_demos(n_episodes=200, noise_std=0.5)

    # Flatten into dataset
    all_obs, all_acts = [], []
    for d in demos:
        all_obs.append(d['obs'])
        all_acts.append(d['actions'])
    all_obs = np.concatenate(all_obs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    # Clip actions to valid range
    all_acts = np.clip(all_acts, -1.0, 1.0)

    obs_t = torch.tensor(all_obs, device=DEVICE, dtype=torch.float32)
    acts_t = torch.tensor(all_acts, device=DEVICE, dtype=torch.float32)
    N = obs_t.shape[0]
    print(f"  Dataset: {N} transitions")

    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    losses = []
    for step in range(num_steps):
        idx = torch.randint(0, N, (batch_size,))
        loss = policy.denoising_loss(obs_t[idx], acts_t[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"    IL step {step+1}: loss={np.mean(losses[-100:]):.4f}")

    return losses


# ============================================================
# GRPO Training (AdaGRPO)
# ============================================================

def grpo_train(policy, env, num_iters=200, G=8, num_groups=2, lr=3e-5,
               clip_eps=0.2, ref_freq=5, per_step_clamp=2.0, eval_freq=20,
               n_eval=10):
    """AdaGRPO training loop on robosuite."""
    policy_old = policy.clone()
    policy_old.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    log = defaultdict(list)

    for iteration in range(num_iters):
        policy.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_used = 0
        iter_rewards = []
        iter_ratios = []

        for _ in range(num_groups):
            group = env.rollout_group(policy_old, G=G)
            rewards = torch.tensor([g['total_reward'] for g in group], device=DEVICE)
            iter_rewards.extend(rewards.tolist())

            if rewards.std() < 1e-8:
                continue
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            n_used += 1

            # Compute ratio over all episode steps
            group_log_ratios = torch.zeros(G, device=DEVICE)
            n_steps_used = 0

            for g_idx in range(G):
                paths = group[g_idx]['paths']
                obs_list = group[g_idx]['obs']
                log_ratio = torch.zeros(1, device=DEVICE)

                for t_step in range(min(len(paths), 50)):  # Cap at 50 steps for ratio
                    inp, out, means_old, sigmas = paths[t_step]
                    obs_t = obs_list[t_step]
                    K = inp.shape[1]

                    for k in range(K):
                        mu_new = policy.compute_mean_at_step(inp[0:1, k].detach(), k, obs_t)
                        inv2s2 = 0.5 / (sigmas[k].item()**2)
                        d_old = (out[0:1, k].detach() - means_old[0:1, k].detach()).pow(2).sum(-1)
                        d_new = (out[0:1, k].detach() - mu_new).pow(2).sum(-1)
                        lr_k = (inv2s2 * (d_old - d_new)).clamp(-per_step_clamp, per_step_clamp)
                        log_ratio = log_ratio + lr_k
                    n_steps_used += 1

                group_log_ratios[g_idx] = log_ratio.squeeze()

            ratio = torch.exp(group_log_ratios.clamp(-5, 5))
            iter_ratios.extend(ratio.detach().cpu().tolist())

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            loss = -torch.min(surr1, surr2).mean()
            total_loss = total_loss + loss

        if n_used == 0:
            continue
        total_loss = total_loss / n_used

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum(p.grad.norm().item()**2 for p in policy.parameters()
                        if p.grad is not None)**0.5
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (iteration + 1) % ref_freq == 0:
            policy_old.load_state_dict(policy.state_dict())

        # Logging
        log['reward'].append(np.mean(iter_rewards))
        log['loss'].append(total_loss.item())
        log['grad_norm'].append(grad_norm)
        log['ratio_mean'].append(np.mean(iter_ratios) if iter_ratios else 1.0)
        log['ratio_std'].append(np.std(iter_ratios) if iter_ratios else 0.0)

        if (iteration + 1) % eval_freq == 0:
            policy.eval()
            eval_results = env.rollout(policy, n_episodes=n_eval)
            eval_r = np.mean([r['reward'] for r in eval_results])
            eval_sr = np.mean([r['success'] for r in eval_results])
            log['eval_reward'].append(eval_r)
            log['eval_success'].append(eval_sr)
            log['eval_iter'].append(iteration + 1)
            print(f"    GRPO iter {iteration+1}: eval_r={eval_r:.3f}  sr={eval_sr:.2f}  "
                  f"loss={total_loss.item():.4f}  grad={grad_norm:.3f}  "
                  f"ratio={np.mean(iter_ratios):.3f}±{np.std(iter_ratios):.3f}")

    return log


# ============================================================
# DPPO Baseline (per-step PPO with value function)
# ============================================================

class ValueNet(nn.Module):
    """Simple value function for DPPO baseline."""
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def dppo_train(policy, env, num_iters=200, G=8, num_groups=2, lr=3e-5,
               value_lr=3e-4, clip_eps=0.2, ref_freq=5, gamma=0.99,
               eval_freq=20, n_eval=10):
    """DPPO baseline: per-step PPO with value function."""
    policy_old = policy.clone()
    policy_old.eval()
    value_fn = ValueNet(policy.obs_dim).to(DEVICE)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=value_lr)

    log = defaultdict(list)

    for iteration in range(num_iters):
        policy.train()
        total_policy_loss = torch.tensor(0.0, device=DEVICE)
        total_value_loss = torch.tensor(0.0, device=DEVICE)
        n_used = 0
        iter_rewards = []

        for _ in range(num_groups):
            group = env.rollout_group(policy_old, G=G)
            rewards_list = [g['total_reward'] for g in group]
            iter_rewards.extend(rewards_list)

            for g_idx in range(G):
                paths = group[g_idx]['paths']
                obs_list = group[g_idx]['obs']
                step_rewards = group[g_idx]['rewards']

                # Compute per-step returns
                T_ep = len(step_rewards)
                returns = torch.zeros(T_ep, device=DEVICE)
                G_t = 0.0
                for t in reversed(range(T_ep)):
                    G_t = step_rewards[t] + gamma * G_t
                    returns[t] = G_t

                # Per-step PPO
                for t_step in range(min(T_ep, 50)):
                    obs_t = obs_list[t_step]
                    inp, out, means_old, sigmas = paths[t_step]
                    K = inp.shape[1]

                    # Value baseline
                    v = value_fn(obs_t.squeeze(0))
                    advantage = (returns[t_step] - v).detach()

                    # Per-step ratio and PPO loss
                    for k in range(K):
                        mu_new = policy.compute_mean_at_step(inp[0:1, k].detach(), k, obs_t)
                        inv2s2 = 0.5 / (sigmas[k].item()**2)
                        d_old = (out[0:1, k].detach() - means_old[0:1, k].detach()).pow(2).sum(-1)
                        d_new = (out[0:1, k].detach() - mu_new).pow(2).sum(-1)
                        lr_k = inv2s2 * (d_old - d_new)
                        ratio_k = torch.exp(lr_k.clamp(-5, 5))

                        surr1 = ratio_k * advantage
                        surr2 = torch.clamp(ratio_k, 1-clip_eps, 1+clip_eps) * advantage
                        total_policy_loss = total_policy_loss - torch.min(surr1, surr2).mean()

                    # Value loss
                    total_value_loss = total_value_loss + F.mse_loss(v, returns[t_step].detach())

                n_used += 1

        if n_used == 0:
            continue

        total_policy_loss = total_policy_loss / n_used
        total_value_loss = total_value_loss / n_used

        optimizer.zero_grad()
        total_policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        value_optimizer.zero_grad()
        total_value_loss.backward()
        value_optimizer.step()

        if (iteration + 1) % ref_freq == 0:
            policy_old.load_state_dict(policy.state_dict())

        log['reward'].append(np.mean(iter_rewards))
        log['loss'].append(total_policy_loss.item())

        if (iteration + 1) % eval_freq == 0:
            policy.eval()
            eval_results = env.rollout(policy, n_episodes=n_eval)
            eval_r = np.mean([r['reward'] for r in eval_results])
            eval_sr = np.mean([r['success'] for r in eval_results])
            log['eval_reward'].append(eval_r)
            log['eval_success'].append(eval_sr)
            log['eval_iter'].append(iteration + 1)
            print(f"    DPPO iter {iteration+1}: eval_r={eval_r:.3f}  sr={eval_sr:.2f}")

    return log


# ============================================================
# Main experiment runner
# ============================================================

def run_task(task_name, max_steps=200, il_steps=5000, rl_iters=150,
             G=6, num_groups=2, eval_freq=25, n_eval=10):
    """Run full experiment for one task: IL → AdaGRPO, IL → DPPO, IL-only."""
    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print(f"{'='*70}")

    env = RobosuiteEnv(task_name, max_steps=max_steps)
    print(f"  obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    # IL Pretrain
    print(f"\n  --- IL Pretraining ---")
    policy_il = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=256,
                            eta=1.0, sigma_min=0.05)
    il_losses = il_pretrain(policy_il, env, num_steps=il_steps)
    il_state = copy.deepcopy(policy_il.state_dict())

    # Evaluate IL baseline
    policy_il.eval()
    il_results = env.rollout(policy_il, n_episodes=n_eval)
    il_reward = np.mean([r['reward'] for r in il_results])
    il_sr = np.mean([r['success'] for r in il_results])
    print(f"  IL baseline: reward={il_reward:.3f}, success={il_sr:.2f}")

    # AdaGRPO
    print(f"\n  --- AdaGRPO RL Fine-Tuning ---")
    policy_grpo = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=256,
                              eta=1.0, sigma_min=0.05)
    policy_grpo.load_state_dict(copy.deepcopy(il_state))
    grpo_log = grpo_train(policy_grpo, env, num_iters=rl_iters, G=G,
                          num_groups=num_groups, eval_freq=eval_freq, n_eval=n_eval)

    # DPPO baseline
    print(f"\n  --- DPPO Baseline ---")
    policy_dppo = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=256,
                              eta=1.0, sigma_min=0.05)
    policy_dppo.load_state_dict(copy.deepcopy(il_state))
    dppo_log = dppo_train(policy_dppo, env, num_iters=rl_iters, G=G,
                          num_groups=num_groups, eval_freq=eval_freq, n_eval=n_eval)

    # Final evaluation
    print(f"\n  --- Final Evaluation (20 episodes) ---")
    final_grpo = env.rollout(policy_grpo, n_episodes=20)
    final_dppo = env.rollout(policy_dppo, n_episodes=20)

    grpo_r = np.mean([r['reward'] for r in final_grpo])
    grpo_sr = np.mean([r['success'] for r in final_grpo])
    dppo_r = np.mean([r['reward'] for r in final_dppo])
    dppo_sr = np.mean([r['success'] for r in final_dppo])

    print(f"\n  Results for {task_name}:")
    print(f"    {'Method':<15} {'Reward':>10} {'Success':>10}")
    print(f"    {'IL (SFT)':<15} {il_reward:>10.3f} {il_sr:>10.2f}")
    print(f"    {'AdaGRPO':<15} {grpo_r:>10.3f} {grpo_sr:>10.2f}")
    print(f"    {'DPPO':<15} {dppo_r:>10.3f} {dppo_sr:>10.2f}")

    # Save results
    result = {
        'task': task_name,
        'il': {'reward': il_reward, 'success': il_sr},
        'adagrpo': {'reward': grpo_r, 'success': grpo_sr, 'log': {k: [float(x) for x in v] for k, v in grpo_log.items()}},
        'dppo': {'reward': dppo_r, 'success': dppo_sr, 'log': {k: [float(x) for x in v] for k, v in dppo_log.items()}},
        'il_losses': [float(x) for x in il_losses[::100]],
    }

    with open(RESULTS_DIR / f"{task_name}.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Scaling study: vary K (denoising steps) and episode length
# ============================================================

def scaling_study(task_name="Lift", max_steps=100):
    """Test how K affects ratio stability and performance."""
    print(f"\n{'='*70}")
    print(f"SCALING STUDY: K sweep on {task_name}")
    print(f"{'='*70}")

    env = RobosuiteEnv(task_name, max_steps=max_steps)
    results = {}

    for K in [3, 5, 8, 10]:
        print(f"\n  K={K}:")
        policy = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=K, hidden=256,
                             eta=1.0, sigma_min=0.05)
        il_pretrain(policy, env, num_steps=3000)
        il_state = copy.deepcopy(policy.state_dict())

        policy.eval()
        il_r = np.mean([r['reward'] for r in env.rollout(policy, n_episodes=10)])

        policy.load_state_dict(il_state)
        grpo_log = grpo_train(policy, env, num_iters=80, G=6, num_groups=2,
                              eval_freq=20, n_eval=10)

        grpo_r = np.mean([r['reward'] for r in env.rollout(policy, n_episodes=10)])

        results[K] = {
            'il_reward': float(il_r),
            'grpo_reward': float(grpo_r),
            'ratio_mean': grpo_log['ratio_mean'],
            'ratio_std': grpo_log['ratio_std'],
        }
        print(f"    IL={il_r:.3f} → GRPO={grpo_r:.3f}  "
              f"ratio_mean={np.mean(grpo_log['ratio_mean']):.3f}  "
              f"ratio_std={np.mean(grpo_log['ratio_std']):.3f}")

    with open(RESULTS_DIR / "scaling_K.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Eta study
# ============================================================

def eta_study(task_name="Lift", max_steps=100):
    """Test how eta affects performance."""
    print(f"\n{'='*70}")
    print(f"ETA STUDY on {task_name}")
    print(f"{'='*70}")

    env = RobosuiteEnv(task_name, max_steps=max_steps)
    results = {}

    for eta in [0.0, 0.3, 0.5, 1.0]:
        print(f"\n  eta={eta}:")
        policy = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=256,
                             eta=eta, sigma_min=0.05)
        il_pretrain(policy, env, num_steps=3000)
        il_state = copy.deepcopy(policy.state_dict())

        policy.eval()
        il_r = np.mean([r['reward'] for r in env.rollout(policy, n_episodes=10)])

        policy.load_state_dict(il_state)
        grpo_log = grpo_train(policy, env, num_iters=80, G=6, num_groups=2,
                              eval_freq=20, n_eval=10)

        grpo_r = np.mean([r['reward'] for r in env.rollout(policy, n_episodes=10)])
        results[eta] = {
            'il_reward': float(il_r),
            'grpo_reward': float(grpo_r),
            'grad_norms': grpo_log['grad_norm'],
            'ratio_mean': grpo_log['ratio_mean'],
        }
        print(f"    IL={il_r:.3f} → GRPO={grpo_r:.3f}  "
              f"avg_grad={np.mean(grpo_log['grad_norm']):.4f}")

    with open(RESULTS_DIR / "eta_study.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()

    all_results = {}

    # Main experiments: 3 robosuite tasks
    for task in ["Lift", "Door", "NutAssemblySquare"]:
        try:
            all_results[task] = run_task(
                task, max_steps=150, il_steps=5000, rl_iters=150,
                G=6, num_groups=2, eval_freq=25, n_eval=10
            )
        except Exception as e:
            print(f"  ERROR on {task}: {e}")
            import traceback; traceback.print_exc()

    # Ablation studies
    try:
        all_results['scaling'] = scaling_study("Lift", max_steps=100)
    except Exception as e:
        print(f"  ERROR on scaling study: {e}")

    try:
        all_results['eta'] = eta_study("Lift", max_steps=100)
    except Exception as e:
        print(f"  ERROR on eta study: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for task in ["Lift", "Door", "NutAssemblySquare"]:
        if task in all_results:
            r = all_results[task]
            print(f"\n  {task}:")
            print(f"    {'Method':<15} {'Reward':>10} {'Success':>10}")
            print(f"    {'IL (SFT)':<15} {r['il']['reward']:>10.3f} {r['il']['success']:>10.2f}")
            print(f"    {'AdaGRPO':<15} {r['adagrpo']['reward']:>10.3f} {r['adagrpo']['success']:>10.2f}")
            print(f"    {'DPPO':<15} {r['dppo']['reward']:>10.3f} {r['dppo']['success']:>10.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save all results
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump({k: v for k, v in all_results.items() if isinstance(v, dict)}, f, indent=2, default=str)

    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
