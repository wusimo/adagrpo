#!/usr/bin/env python3
"""Harder validation: GRPO for diffusion on problems where IL is clearly suboptimal.

Tests:
1. Biased expert (demos systematically miss the target) — can GRPO correct?
2. Multimodal target (two valid targets) — does diffusion + GRPO handle it?
3. Higher-dimensional action space (closer to real robotics)
4. Sequential decision making (multi-step episode, not single-shot bandit)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# Reuse DiffPolicy from the previous test
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
    def __init__(self, action_dim, obs_dim, hidden=128):
        super().__init__()
        self.t_emb = SinusoidalEmbed(hidden)
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden * 2, hidden), nn.Mish(),
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
    def __init__(self, action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05):
        self.action_dim, self.obs_dim = action_dim, obs_dim
        self.T, self.K, self.eta, self.sigma_min = T, K, eta, sigma_min
        self.noise_net = NoiseNet(action_dim, obs_dim, hidden).to(DEVICE)
        betas = cosine_beta_schedule(T).to(DEVICE)
        self.ac = torch.cumprod(1 - betas, 0)
        step_ratio = T // K
        self.timesteps = (torch.arange(0, K) * step_ratio).long().flip(0).to(DEVICE)

    def parameters(self):
        return self.noise_net.parameters()
    def state_dict(self):
        return self.noise_net.state_dict()
    def load_state_dict(self, sd):
        self.noise_net.load_state_dict(sd)
    def train(self):
        self.noise_net.train()
    def eval(self):
        self.noise_net.eval()

    def _step_params(self, idx):
        t = self.timesteps[idx]
        t_prev = self.timesteps[idx+1] if idx+1 < self.K else torch.tensor(0, device=DEVICE)
        ac_t, ac_prev = self.ac[t], (self.ac[t_prev] if t_prev > 0 else torch.tensor(1.0, device=DEVICE))
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


# ============================================================
# GRPO training loop (Strategy A: product ratio with sigma_min)
# ============================================================

def grpo_train(policy, env_reward_fn, obs_sampler, expert_fn, il_steps=3000,
               rl_iters=400, G=8, num_groups=4, lr_il=1e-3, lr_rl=3e-5,
               clip_eps=0.2, ref_freq=5, per_step_clamp=2.0, desc=""):

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")

    # IL pretrain
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr_il)
    for step in range(il_steps):
        obs = obs_sampler(64)
        expert = expert_fn(obs)
        loss = policy.denoising_loss(obs, expert)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 1000 == 0:
            policy.eval()
            eo = obs_sampler(500)
            with torch.no_grad():
                er = env_reward_fn(eo, policy.sample(eo)).mean().item()
            policy.train()
            print(f"  IL step {step+1}: loss={loss.item():.4f}, eval_r={er:.4f}")

    policy.eval()
    eo = obs_sampler(1000)
    with torch.no_grad():
        il_reward = env_reward_fn(eo, policy.sample(eo)).mean().item()
    print(f"  IL baseline: {il_reward:.4f}")
    il_state = copy.deepcopy(policy.state_dict())

    # GRPO
    policy.load_state_dict(il_state)
    hidden = policy.noise_net.obs_proj.out_features
    policy_old = DiffPolicy(policy.action_dim, policy.obs_dim, T=policy.T, K=policy.K,
                             hidden=hidden, eta=policy.eta, sigma_min=policy.sigma_min)
    policy_old.load_state_dict(il_state)
    policy_old.eval()

    rl_opt = torch.optim.Adam(policy.parameters(), lr=lr_rl)

    for iteration in range(rl_iters):
        policy.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_used = 0

        for _ in range(num_groups):
            obs = obs_sampler(1).expand(G, -1).contiguous()
            with torch.no_grad():
                actions, inp, out, means_old, sigmas = policy_old.sample_with_path(obs)
            rewards = env_reward_fn(obs, actions)
            if rewards.std() < 1e-8:
                continue
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            K = inp.shape[1]
            log_ratio = torch.zeros(G, device=DEVICE)
            for k in range(K):
                mu_new = policy.compute_mean_at_step(inp[:, k].detach(), k, obs)
                inv2s2 = 0.5 / (sigmas[k].item()**2)
                d_old = (out[:, k].detach() - means_old[:, k].detach()).pow(2).sum(-1)
                d_new = (out[:, k].detach() - mu_new).pow(2).sum(-1)
                log_ratio = log_ratio + (inv2s2 * (d_old - d_new)).clamp(-per_step_clamp, per_step_clamp)

            ratio = torch.exp(log_ratio.clamp(-5, 5))
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            total_loss = total_loss - torch.min(surr1, surr2).mean()
            n_used += 1

        if n_used == 0:
            continue
        total_loss = total_loss / n_used
        rl_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        rl_opt.step()

        if (iteration+1) % ref_freq == 0:
            policy_old.load_state_dict(policy.state_dict())

        if (iteration+1) % 100 == 0:
            policy.eval()
            eo = obs_sampler(500)
            with torch.no_grad():
                er = env_reward_fn(eo, policy.sample(eo)).mean().item()
            print(f"  RL iter {iteration+1}: eval_r={er:.4f}")

    policy.eval()
    eo = obs_sampler(1000)
    with torch.no_grad():
        final_r = env_reward_fn(eo, policy.sample(eo)).mean().item()
    delta = final_r - il_reward
    pct = delta / abs(il_reward) * 100 if il_reward != 0 else 0
    ok = "✓" if delta > 0 else "✗"
    print(f"  Result: IL={il_reward:.4f} → GRPO={final_r:.4f}  Δ={delta:+.4f} ({pct:+.1f}%) {ok}")
    return il_reward, final_r


# ============================================================
# Test 1: Biased expert (systematic offset — can GRPO correct?)
# ============================================================

def test_biased_expert():
    torch.manual_seed(42)
    action_dim, obs_dim = 2, 4
    target_fn = lambda obs: obs[:, :2] * 0.5

    def reward_fn(obs, action):
        return -(action - target_fn(obs)).pow(2).sum(-1)

    # Expert has systematic bias: always shifted by +0.3
    def biased_expert(obs):
        return target_fn(obs) + 0.3 + torch.randn(obs.shape[0], action_dim, device=DEVICE) * 0.05

    policy = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)
    return grpo_train(
        policy, reward_fn,
        obs_sampler=lambda n: torch.randn(n, obs_dim, device=DEVICE),
        expert_fn=biased_expert,
        il_steps=2000, rl_iters=400,
        desc="Test 1: Biased expert (expert has +0.3 offset)",
    )


# ============================================================
# Test 2: Higher-dimensional actions (7D like real robotics)
# ============================================================

def test_high_dim():
    torch.manual_seed(42)
    action_dim, obs_dim = 7, 10

    W_true = torch.randn(obs_dim, action_dim, device=DEVICE) * 0.3

    def reward_fn(obs, action):
        target = obs @ W_true
        return -(action - target).pow(2).sum(-1)

    def expert_fn(obs):
        # Noisy expert with some bias
        return obs @ W_true + 0.1 + torch.randn(obs.shape[0], action_dim, device=DEVICE) * 0.15

    policy = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=256, eta=1.0, sigma_min=0.05)
    return grpo_train(
        policy, reward_fn,
        obs_sampler=lambda n: torch.randn(n, obs_dim, device=DEVICE),
        expert_fn=expert_fn,
        il_steps=3000, rl_iters=400,
        desc="Test 2: 7D action space (robotics-scale)",
    )


# ============================================================
# Test 3: Sparse reward (only get reward=1 if within threshold)
# ============================================================

def test_sparse_reward():
    torch.manual_seed(42)
    action_dim, obs_dim = 2, 4
    target_fn = lambda obs: obs[:, :2] * 0.3

    def reward_fn(obs, action):
        dist = (action - target_fn(obs)).pow(2).sum(-1).sqrt()
        return (dist < 0.3).float()  # Binary: 1 if close, 0 otherwise

    def expert_fn(obs):
        return target_fn(obs) + torch.randn(obs.shape[0], action_dim, device=DEVICE) * 0.2

    policy = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)
    return grpo_train(
        policy, reward_fn,
        obs_sampler=lambda n: torch.randn(n, obs_dim, device=DEVICE),
        expert_fn=expert_fn,
        il_steps=2000, rl_iters=400, G=16,
        desc="Test 3: Sparse binary reward (success/fail)",
    )


# ============================================================
# Test 4: Multi-step episode (not just a bandit)
# ============================================================

def test_multistep():
    """Simple 5-step episode: at each step, choose action to move toward target.
    State evolves: s_{t+1} = s_t + a_t. Reward = -||s_final - target||^2."""
    torch.manual_seed(42)
    action_dim, obs_dim = 2, 4  # obs = [pos(2), target(2)]
    H = 5  # episode length

    def run_episode(policy, obs_batch):
        """Run H-step episode. obs_batch: [B, 4] = [pos, target]."""
        B = obs_batch.shape[0]
        pos = obs_batch[:, :2].clone()
        target = obs_batch[:, 2:4]
        total_reward = torch.zeros(B, device=DEVICE)

        all_obs = []
        all_actions = []
        all_paths = []

        for step in range(H):
            obs = torch.cat([pos, target], dim=-1)
            all_obs.append(obs)
            with torch.no_grad():
                action, inp, out, means, sigmas = policy.sample_with_path(obs)
            all_actions.append(action)
            all_paths.append((inp, out, means, sigmas))
            pos = pos + action * 0.2  # Move
            dist = (pos - target).pow(2).sum(-1).sqrt()
            total_reward += -dist  # Penalize distance at each step

        return total_reward, all_obs, all_actions, all_paths

    def run_episode_eval(policy, obs_batch):
        B = obs_batch.shape[0]
        pos = obs_batch[:, :2].clone()
        target = obs_batch[:, 2:4]
        total_reward = torch.zeros(B, device=DEVICE)
        for step in range(H):
            obs = torch.cat([pos, target], dim=-1)
            with torch.no_grad():
                action = policy.sample(obs)
            pos = pos + action * 0.2
            total_reward += -(pos - target).pow(2).sum(-1).sqrt()
        return total_reward

    policy = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)

    print(f"\n{'='*60}")
    print(f"  Test 4: Multi-step episode (H={H} steps)")
    print(f"{'='*60}")

    # IL pretrain
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for step in range(3000):
        obs = torch.randn(64, obs_dim, device=DEVICE)
        # Expert: move toward target
        pos, target = obs[:, :2], obs[:, 2:4]
        expert_action = (target - pos) * 0.5 + torch.randn(64, action_dim, device=DEVICE) * 0.1
        loss = policy.denoising_loss(obs, expert_action)
        opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    eo = torch.randn(500, obs_dim, device=DEVICE)
    il_reward = run_episode_eval(policy, eo).mean().item()
    print(f"  IL baseline: {il_reward:.4f}")
    il_state = copy.deepcopy(policy.state_dict())

    # GRPO with trajectory-level reward
    G, clip_eps, lr_rl, num_iters = 8, 0.2, 3e-5, 400
    policy.load_state_dict(il_state)
    policy_old = DiffPolicy(action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)
    policy_old.load_state_dict(il_state); policy_old.eval()
    rl_opt = torch.optim.Adam(policy.parameters(), lr=lr_rl)

    for iteration in range(num_iters):
        policy.train()
        # Sample initial state
        init_obs = torch.randn(1, obs_dim, device=DEVICE).expand(G, -1).contiguous()

        # Collect G episodes with old policy
        with torch.no_grad():
            rewards, all_obs, all_actions, all_paths = run_episode(policy_old, init_obs)

        if rewards.std() < 1e-8:
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute trajectory-level ratio = product over all timesteps and all denoising steps
        log_ratio = torch.zeros(G, device=DEVICE)
        for t_step in range(H):
            obs_t = all_obs[t_step]
            inp, out, means_old, sigmas = all_paths[t_step]
            K = inp.shape[1]
            for k in range(K):
                mu_new = policy.compute_mean_at_step(inp[:, k].detach(), k, obs_t)
                inv2s2 = 0.5 / (sigmas[k].item()**2)
                d_old = (out[:, k].detach() - means_old[:, k].detach()).pow(2).sum(-1)
                d_new = (out[:, k].detach() - mu_new).pow(2).sum(-1)
                log_ratio += (inv2s2 * (d_old - d_new)).clamp(-2.0, 2.0)

        ratio = torch.exp(log_ratio.clamp(-5, 5))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
        loss = -torch.min(surr1, surr2).mean()

        rl_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        rl_opt.step()

        if (iteration+1) % 5 == 0:
            policy_old.load_state_dict(policy.state_dict())

        if (iteration+1) % 100 == 0:
            policy.eval()
            eo = torch.randn(500, obs_dim, device=DEVICE)
            er = run_episode_eval(policy, eo).mean().item()
            print(f"  RL iter {iteration+1}: eval_r={er:.4f}  ratio={ratio.mean():.3f}±{ratio.std():.3f}")

    policy.eval()
    eo = torch.randn(1000, obs_dim, device=DEVICE)
    final_r = run_episode_eval(policy, eo).mean().item()
    d = final_r - il_reward
    pct = d/abs(il_reward)*100 if il_reward != 0 else 0
    ok = "✓" if d > 0 else "✗"
    print(f"  Result: IL={il_reward:.4f} → GRPO={final_r:.4f}  Δ={d:+.4f} ({pct:+.1f}%) {ok}")
    return il_reward, final_r


# ============================================================
# Main
# ============================================================

def main():
    results = {}
    results["biased_expert"] = test_biased_expert()
    results["high_dim_7d"] = test_high_dim()
    results["sparse_reward"] = test_sparse_reward()
    results["multi_step"] = test_multistep()

    print(f"\n{'='*70}")
    print("SUMMARY OF HARDER TESTS")
    print(f"{'='*70}")
    print(f"{'Test':<25} {'IL':>10} {'GRPO':>10} {'Delta':>10} {'%':>10}")
    print("-" * 65)
    for name, (il, grpo) in results.items():
        d = grpo - il
        p = d / abs(il) * 100 if il != 0 else 0
        ok = "✓" if d > 0 else "✗"
        print(f"{name:<25} {il:>10.4f} {grpo:>10.4f} {d:>+10.4f} {p:>+9.1f}% {ok}")


if __name__ == "__main__":
    main()
