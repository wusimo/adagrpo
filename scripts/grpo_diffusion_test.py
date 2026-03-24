#!/usr/bin/env python3
"""Focused test: Does path-conditioned GRPO improve a diffusion policy?

Discovered issues and fixes applied:
  1. Deterministic DDIM (eta=0): a^{k-1} = μ_old, so ratio = 1 always → DEAD
  2. Stochastic DDIM (eta>0): last steps have σ≈0, so 1/(2σ²) explodes → ratio underflows
  3. Fix: enforce sigma_min, AND try per-step PPO (DPPO-style) as alternative

This script tests three strategies:
  A. Product ratio + sigma floor (AdaGRPO paper's approach)
  B. Per-step PPO (DPPO-style: each denoising step is a separate "action")
  C. Average log-ratio (geometric mean instead of product)
"""

import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# Diffusion Policy (same as before)
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
    betas = 1 - ac[1:] / ac[:-1]
    return betas.clamp(0, 0.999)

class DiffPolicy:
    def __init__(self, action_dim, obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.01):
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
        ac_t = self.ac[t]
        ac_prev = self.ac[t_prev] if t_prev > 0 else torch.tensor(1.0, device=DEVICE)
        pvar = ((1-ac_prev)/(1-ac_t+1e-8)*(1-ac_t/(ac_prev+1e-8))).clamp(min=1e-8)
        sigma = max(self.eta * pvar.sqrt().item(), self.sigma_min)
        return ac_t, ac_prev, sigma, t

    def _mean(self, x_t, noise_pred, ac_t, ac_prev, sigma):
        pred_x0 = (x_t - (1-ac_t).sqrt() * noise_pred) / ac_t.sqrt()
        dir_xt = (1-ac_prev-sigma**2).clamp(min=0).sqrt() * noise_pred
        return ac_prev.sqrt() * pred_x0 + dir_xt

    def denoising_loss(self, obs, actions):
        B = actions.shape[0]
        t = torch.randint(0, self.T, (B,), device=DEVICE)
        noise = torch.randn_like(actions)
        ac_ = self.ac[t].unsqueeze(-1)
        x_t = ac_.sqrt() * actions + (1-ac_).sqrt() * noise
        return F.mse_loss(self.noise_net(x_t, t, obs), noise)

    @torch.no_grad()
    def sample(self, obs):
        B = obs.shape[0]
        x = torch.randn(B, self.action_dim, device=DEVICE)
        for i in range(self.K):
            ac_t, ac_prev, sigma, t = self._step_params(i)
            pred = self.noise_net(x, t.expand(B), obs)
            x = self._mean(x, pred, ac_t, ac_prev, sigma)
        return x

    @torch.no_grad()
    def sample_with_path(self, obs):
        B = obs.shape[0]
        x = torch.randn(B, self.action_dim, device=DEVICE)
        inputs, outputs, means, sigs = [], [], [], []
        for i in range(self.K):
            ac_t, ac_prev, sigma, t = self._step_params(i)
            inputs.append(x.clone())
            pred = self.noise_net(x, t.expand(B), obs)
            mean = self._mean(x, pred, ac_t, ac_prev, sigma)
            if i < self.K - 1:
                x_next = mean + sigma * torch.randn_like(x)
            else:
                x_next = mean
            outputs.append(x_next.clone())
            means.append(mean)
            sigs.append(sigma)
            x = x_next
        return (x, torch.stack(inputs, 1), torch.stack(outputs, 1),
                torch.stack(means, 1), torch.tensor(sigs, device=DEVICE))

    def compute_mean_at_step(self, a_k, idx, obs):
        ac_t, ac_prev, sigma, t = self._step_params(idx)
        pred = self.noise_net(a_k, t.expand(a_k.shape[0]), obs)
        return self._mean(a_k, pred, ac_t, ac_prev, sigma)


# ============================================================
# Environment
# ============================================================

class ReachingEnv:
    obs_dim, action_dim = 4, 2
    def get_target(self, obs):
        return obs[:, :2] * 0.5
    def reward(self, obs, action):
        return -(action - self.get_target(obs)).pow(2).sum(-1)


# ============================================================
# Three GRPO strategies
# ============================================================

def strategy_A_product_ratio(policy, policy_old, obs, actions, a_k_inputs,
                              a_km1_outputs, means_old, sigmas, advantages, clip_eps):
    """Standard path-conditioned GRPO: product of per-step ratios."""
    G, K, D = a_k_inputs.shape
    log_ratio = torch.zeros(G, device=DEVICE)

    for k in range(K):
        mu_new = policy.compute_mean_at_step(a_k_inputs[:, k].detach(), k, obs)
        inv2s2 = 0.5 / (sigmas[k].item()**2)
        d_old = (a_km1_outputs[:, k].detach() - means_old[:, k].detach()).pow(2).sum(-1)
        d_new = (a_km1_outputs[:, k].detach() - mu_new).pow(2).sum(-1)
        lr_k = (inv2s2 * (d_old - d_new)).clamp(-2.0, 2.0)  # Tight clamp
        log_ratio = log_ratio + lr_k

    ratio = torch.exp(log_ratio.clamp(-5, 5))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    return -torch.min(surr1, surr2).mean(), ratio


def strategy_B_per_step_ppo(policy, policy_old, obs, actions, a_k_inputs,
                             a_km1_outputs, means_old, sigmas, advantages, clip_eps):
    """DPPO-style: apply PPO clip to each denoising step independently."""
    G, K, D = a_k_inputs.shape
    total_loss = torch.tensor(0.0, device=DEVICE)
    all_ratios = []

    for k in range(K):
        mu_new = policy.compute_mean_at_step(a_k_inputs[:, k].detach(), k, obs)
        inv2s2 = 0.5 / (sigmas[k].item()**2)
        d_old = (a_km1_outputs[:, k].detach() - means_old[:, k].detach()).pow(2).sum(-1)
        d_new = (a_km1_outputs[:, k].detach() - mu_new).pow(2).sum(-1)
        lr_k = inv2s2 * (d_old - d_new)
        ratio_k = torch.exp(lr_k.clamp(-5, 5))

        surr1 = ratio_k * advantages
        surr2 = torch.clamp(ratio_k, 1-clip_eps, 1+clip_eps) * advantages
        total_loss = total_loss + (-torch.min(surr1, surr2).mean())
        all_ratios.append(ratio_k.detach())

    total_loss = total_loss / K
    ratio_combined = torch.stack(all_ratios).mean(dim=0)
    return total_loss, ratio_combined


def strategy_C_mean_log_ratio(policy, policy_old, obs, actions, a_k_inputs,
                               a_km1_outputs, means_old, sigmas, advantages, clip_eps):
    """Average (not sum) log-ratio — geometric mean of per-step ratios."""
    G, K, D = a_k_inputs.shape
    log_ratio = torch.zeros(G, device=DEVICE)

    for k in range(K):
        mu_new = policy.compute_mean_at_step(a_k_inputs[:, k].detach(), k, obs)
        inv2s2 = 0.5 / (sigmas[k].item()**2)
        d_old = (a_km1_outputs[:, k].detach() - means_old[:, k].detach()).pow(2).sum(-1)
        d_new = (a_km1_outputs[:, k].detach() - mu_new).pow(2).sum(-1)
        lr_k = inv2s2 * (d_old - d_new)
        log_ratio = log_ratio + lr_k

    # Average instead of sum
    log_ratio = log_ratio / K
    ratio = torch.exp(log_ratio.clamp(-5, 5))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    return -torch.min(surr1, surr2).mean(), ratio


# ============================================================
# Run experiment
# ============================================================

def run_one(strategy_name, strategy_fn, sigma_min=0.01):
    torch.manual_seed(42)
    env = ReachingEnv()

    policy = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=128,
                         eta=1.0, sigma_min=sigma_min)

    # IL pretrain
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for step in range(2000):
        obs = torch.randn(64, env.obs_dim, device=DEVICE)
        expert = env.get_target(obs) + torch.randn(64, env.action_dim, device=DEVICE) * 0.1
        loss = policy.denoising_loss(obs, expert)
        opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    eval_obs = torch.randn(500, env.obs_dim, device=DEVICE)
    with torch.no_grad():
        il_reward = env.reward(eval_obs, policy.sample(eval_obs)).mean().item()
    print(f"  IL baseline: {il_reward:.4f}")

    il_state = copy.deepcopy(policy.state_dict())

    # GRPO
    G, clip_eps, lr = 8, 0.2, 3e-5
    num_iters, ref_freq, num_groups = 300, 5, 4

    policy.load_state_dict(il_state)
    policy_old = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=5, hidden=128,
                              eta=1.0, sigma_min=sigma_min)
    policy_old.load_state_dict(il_state)
    policy_old.eval()

    rl_opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for iteration in range(num_iters):
        policy.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_used = 0
        iter_r = []

        for _ in range(num_groups):
            obs = torch.randn(1, env.obs_dim, device=DEVICE).expand(G, -1).contiguous()
            with torch.no_grad():
                actions, inp, out, means_old, sigmas = policy_old.sample_with_path(obs)
            rewards = env.reward(obs, actions)
            iter_r.extend(rewards.tolist())
            if rewards.std() < 1e-8:
                continue
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            loss, ratio = strategy_fn(policy, policy_old, obs, actions,
                                       inp, out, means_old, sigmas, adv, clip_eps)
            total_loss = total_loss + loss
            n_used += 1

        if n_used == 0:
            continue
        total_loss = total_loss / n_used

        rl_opt.zero_grad()
        total_loss.backward()
        grad_norm = sum(p.grad.norm().item()**2 for p in policy.parameters()
                        if p.grad is not None)**0.5
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        rl_opt.step()

        if (iteration + 1) % ref_freq == 0:
            policy_old.load_state_dict(policy.state_dict())

        if (iteration + 1) % 50 == 0:
            policy.eval()
            eo = torch.randn(500, env.obs_dim, device=DEVICE)
            with torch.no_grad():
                er = env.reward(eo, policy.sample(eo)).mean().item()
            print(f"  Iter {iteration+1:3d}: eval={er:.4f}  loss={total_loss.item():.6f}  "
                  f"grad={grad_norm:.4f}")

    policy.eval()
    eo = torch.randn(1000, env.obs_dim, device=DEVICE)
    with torch.no_grad():
        final_r = env.reward(eo, policy.sample(eo)).mean().item()
    delta = final_r - il_reward
    pct = delta / abs(il_reward) * 100 if il_reward != 0 else 0
    ok = "✓" if delta > 0 else "✗"
    print(f"  Result: IL={il_reward:.4f} → GRPO={final_r:.4f}  "
          f"Δ={delta:+.4f} ({pct:+.1f}%)  {ok}")
    return il_reward, final_r


def main():
    results = {}

    for sigma_min in [0.01, 0.05, 0.1, 0.2]:
        print(f"\n{'#'*70}")
        print(f"# Strategy A: Product ratio, sigma_min={sigma_min}")
        print(f"{'#'*70}")
        results[f"A_sigmin{sigma_min}"] = run_one(
            "product", strategy_A_product_ratio, sigma_min=sigma_min
        )

    print(f"\n{'#'*70}")
    print(f"# Strategy B: Per-step PPO (DPPO-style), sigma_min=0.1")
    print(f"{'#'*70}")
    results["B_perstep"] = run_one("per_step_ppo", strategy_B_per_step_ppo, sigma_min=0.1)

    print(f"\n{'#'*70}")
    print(f"# Strategy C: Mean log-ratio (geometric mean), sigma_min=0.1")
    print(f"{'#'*70}")
    results["C_mean"] = run_one("mean_log_ratio", strategy_C_mean_log_ratio, sigma_min=0.1)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'IL':>10} {'GRPO':>10} {'Delta':>10} {'%':>10}")
    print("-" * 65)
    for name, (il, grpo) in results.items():
        d = grpo - il
        p = d / abs(il) * 100 if il != 0 else 0
        ok = "✓" if d > 0 else "✗"
        print(f"{name:<25} {il:>10.4f} {grpo:>10.4f} {d:>+10.4f} {p:>+9.1f}% {ok}")


if __name__ == "__main__":
    main()
