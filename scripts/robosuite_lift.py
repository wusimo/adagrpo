#!/usr/bin/env python3
"""AdaGRPO on robosuite Lift using robomimic demo data.

Uses human demos from robomimic for IL pretraining, then GRPO fine-tuning.
"""

import copy, json, time, sys, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from grpo_diffusion_test import DiffPolicy, DEVICE

RESULTS_DIR = Path("results/robosuite")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load robomimic demos
# ============================================================

def load_robomimic_demos(path, max_demos=200, obs_keys=None):
    """Load (obs, action) pairs from robomimic hdf5."""
    f = h5py.File(path, 'r')
    demos = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))[:max_demos]

    if obs_keys is None:
        # Use all obs keys from first demo, sorted for consistency
        obs_keys = sorted(list(f[f'data/{demos[0]}/obs'].keys()))

    all_obs, all_acts = [], []
    for demo_key in demos:
        demo = f[f'data/{demo_key}']
        obs_parts = [demo['obs'][key][:] for key in obs_keys if key in demo['obs']]
        obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)
        acts = demo['actions'][:].astype(np.float32)
        all_obs.append(obs)
        all_acts.append(acts)

    all_obs = np.concatenate(all_obs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    f.close()
    return all_obs, all_acts, obs_keys


# ============================================================
# Robosuite Lift env
# ============================================================

class LiftEnv:
    def __init__(self, max_steps=200, obs_keys=None):
        import robosuite as suite
        self.env = suite.make(
            'Lift', robots='Panda',
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, reward_shaping=True,
            control_freq=20, horizon=max_steps,
        )
        self.max_steps = max_steps
        self.action_dim = self.env.action_dim

        obs = self.env.reset()
        # Use specified obs keys (matching demo data)
        if obs_keys is not None:
            self.obs_keys = [k for k in obs_keys if k in obs]
        else:
            self.obs_keys = sorted(obs.keys())
        self.obs_dim = sum(obs[k].shape[0] for k in self.obs_keys)
        print(f"  Lift: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"  Obs keys ({len(self.obs_keys)}): {self.obs_keys[:5]}...")

    def _flat_obs(self, obs):
        return np.concatenate([obs[k] for k in self.obs_keys]).astype(np.float32)

    def rollout(self, policy, n_episodes=10):
        """Evaluate policy."""
        results = []
        policy.eval()
        for ep in range(n_episodes):
            obs = self.env.reset()
            obs_flat = self._flat_obs(obs)
            total_r = 0.0
            for step in range(self.max_steps):
                obs_t = torch.tensor(obs_flat, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action = policy.sample(obs_t)[0].cpu().numpy()
                if action.shape[0] < self.action_dim:
                    action = np.concatenate([action, np.zeros(self.action_dim - action.shape[0])])
                action = np.clip(action, -1, 1)
                obs, r, done, info = self.env.step(action)
                obs_flat = self._flat_obs(obs)
                total_r += r
                if done: break
            results.append(total_r)
        return results

    def rollout_with_path(self, policy, n_episodes=1):
        """Rollout recording denoising paths for GRPO."""
        policy.eval()
        episodes = []
        for ep in range(n_episodes):
            obs = self.env.reset()
            obs_flat = self._flat_obs(obs)
            ep_data = {'obs': [], 'paths': [], 'rewards': [], 'total_r': 0.0}
            for step in range(self.max_steps):
                obs_t = torch.tensor(obs_flat, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action, inp, out, means, sigmas = policy.sample_with_path(obs_t)
                ep_data['obs'].append(obs_t)
                ep_data['paths'].append((inp, out, means, sigmas))
                a_np = action[0].cpu().numpy()
                if a_np.shape[0] < self.action_dim:
                    a_np = np.concatenate([a_np, np.zeros(self.action_dim - a_np.shape[0])])
                a_np = np.clip(a_np, -1, 1)
                obs, r, done, info = self.env.step(a_np)
                obs_flat = self._flat_obs(obs)
                ep_data['rewards'].append(float(r))
                ep_data['total_r'] += r
                if done: break
            episodes.append(ep_data)
        return episodes


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    torch.manual_seed(42)

    # Load demos
    demo_path = "/home/simo/miniconda3/envs/libero/lib/python3.10/site-packages/datasets/lift/ph/low_dim.hdf5"
    print("Loading robomimic demos...")
    # Use only the obs keys that exist in both demos and live env
    SHARED_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                   'robot0_gripper_qvel', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin',
                   'robot0_joint_vel']
    # Also include object state (named 'object' in demos, 'object-state' in env)
    DEMO_KEYS = SHARED_KEYS + ['object']
    ENV_KEYS = SHARED_KEYS + ['object-state']

    obs_data, act_data, _ = load_robomimic_demos(demo_path, max_demos=200, obs_keys=DEMO_KEYS)
    print(f"  Demos: {obs_data.shape[0]} transitions, obs_dim={obs_data.shape[1]}, act_dim={act_data.shape[1]}")

    obs_t = torch.tensor(obs_data, device=DEVICE)
    act_t = torch.tensor(act_data, device=DEVICE)

    # Setup env
    print("Setting up Lift env...")
    env = LiftEnv(max_steps=200, obs_keys=ENV_KEYS)

    # Use demo dimensions for policy (they must match)
    obs_dim = obs_data.shape[1]
    action_dim = act_data.shape[1]
    assert obs_dim == env.obs_dim, f"Obs dim mismatch: demos={obs_dim}, env={env.obs_dim}"
    print(f"  Policy: obs_dim={obs_dim}, action_dim={action_dim} (env action_dim={env.action_dim})")
    # If env has more action dims (e.g. 8 vs 7), pad with zeros during rollout
    action_pad = env.action_dim - action_dim

    # IL Pretrain
    print("\n--- IL Pretraining ---")
    policy = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=256, eta=1.0, sigma_min=0.05)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    N = obs_t.shape[0]
    for step in range(5000):
        idx = torch.randint(0, N, (128,))
        loss = policy.denoising_loss(obs_t[idx], act_t[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}")

    # Evaluate IL
    print("\nEvaluating IL...")
    il_results = env.rollout(policy, n_episodes=10)
    il_reward = np.mean(il_results)
    print(f"  IL reward: {il_reward:.3f} (per-episode mean)")

    il_state = copy.deepcopy(policy.state_dict())

    # GRPO Fine-Tuning
    print("\n--- AdaGRPO Fine-Tuning ---")
    policy.load_state_dict(il_state)
    policy_old = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=256, eta=1.0, sigma_min=0.05)
    policy_old.load_state_dict(il_state); policy_old.eval()
    rl_opt = torch.optim.Adam(policy.parameters(), lr=1e-5)

    G = 4  # Group size (expensive in sim)
    log = defaultdict(list)

    for iteration in range(100):
        policy.train()
        # Collect G episodes
        episodes = env.rollout_with_path(policy_old, n_episodes=G)
        rewards = torch.tensor([ep['total_r'] for ep in episodes], device=DEVICE)

        if rewards.std() < 1e-8:
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute ratio for each episode (cap at 30 steps for ratio)
        losses = []
        for g_idx in range(G):
            ep = episodes[g_idx]
            n_steps = min(len(ep['paths']), 30)
            log_ratio = torch.tensor(0.0, device=DEVICE)
            for t_step in range(n_steps):
                obs_t_s = ep['obs'][t_step]
                inp, out, mo, sig = ep['paths'][t_step]
                K = inp.shape[1]
                for k in range(K):
                    mn = policy.compute_mean_at_step(inp[0:1, k].detach(), k, obs_t_s)
                    i2 = 0.5 / (sig[k].item()**2)
                    do = (out[0:1, k].detach() - mo[0:1, k].detach()).pow(2).sum(-1)
                    dn = (out[0:1, k].detach() - mn).pow(2).sum(-1)
                    log_ratio = log_ratio + (i2 * (do - dn)).clamp(-2.0, 2.0)

            ratio = torch.exp(log_ratio.clamp(-5, 5))
            s1 = ratio * adv[g_idx]
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv[g_idx]
            losses.append(-torch.min(s1, s2))

        loss = torch.stack(losses).mean()

        # Add auxiliary denoising loss
        idx = torch.randint(0, N, (64,))
        aux_loss = policy.denoising_loss(obs_t[idx], act_t[idx])
        total_loss = loss + 0.1 * aux_loss

        rl_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        rl_opt.step()

        if (iteration + 1) % 5 == 0:
            policy_old.load_state_dict(policy.state_dict())

        log['reward'].append(float(rewards.mean().item()))
        log['loss'].append(float(total_loss.item()))

        if (iteration + 1) % 20 == 0:
            policy.eval()
            eval_r = env.rollout(policy, n_episodes=5)
            mean_r = np.mean(eval_r)
            log['eval_r'].append(float(mean_r))
            log['eval_it'].append(iteration + 1)
            print(f"  Iter {iteration+1}: rollout_r={rewards.mean():.3f}  eval_r={mean_r:.3f}")

    # Final evaluation
    policy.eval()
    final_results = env.rollout(policy, n_episodes=20)
    final_r = np.mean(final_results)
    delta = final_r - il_reward
    pct = delta / abs(il_reward) * 100 if il_reward != 0 else 0

    print(f"\n{'='*60}")
    print(f"LIFT RESULTS")
    print(f"{'='*60}")
    print(f"  IL reward:    {il_reward:.3f}")
    print(f"  GRPO reward:  {final_r:.3f}")
    print(f"  Improvement:  {delta:+.3f} ({pct:+.1f}%)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    result = {
        'il_reward': float(il_reward),
        'grpo_reward': float(final_r),
        'delta': float(delta),
        'pct': float(pct),
        'log': {k: [float(x) for x in v] for k, v in log.items()},
    }
    with open(RESULTS_DIR / "lift_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/lift_results.json")


if __name__ == "__main__":
    main()
