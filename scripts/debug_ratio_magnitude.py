#!/usr/bin/env python3
"""Debug: Why do path-conditioned ratios collapse to 0?

Hypothesis: even after the first ref update (iter 5), the per-step log-ratios
are large and negative, so their product underflows.
"""
import copy, torch, torch.nn as nn, numpy as np
DEVICE = torch.device("cuda")

# Reuse the policy from grpo_diffusion_test
from grpo_diffusion_test import StochasticDiffusionPolicy, ReachingEnv, per_step_log_ratio

torch.manual_seed(42)
env = ReachingEnv()

policy = StochasticDiffusionPolicy(env.action_dim, env.obs_dim, T_train=50, K_infer=5, hidden=128, eta=1.0)

# IL pretrain
policy.train()
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
for step in range(1000):
    obs = torch.randn(64, env.obs_dim, device=DEVICE)
    target = env.get_target(obs)
    expert = target + torch.randn_like(target) * 0.1
    loss = policy.denoising_loss(obs, expert)
    opt.zero_grad()
    loss.backward()
    opt.step()

il_state = copy.deepcopy(policy.state_dict())

# Create old and new (slightly different) policies
policy_old = StochasticDiffusionPolicy(env.action_dim, env.obs_dim, T_train=50, K_infer=5, hidden=128, eta=1.0)
policy_old.load_state_dict(il_state)
policy_old.eval()

policy_new = StochasticDiffusionPolicy(env.action_dim, env.obs_dim, T_train=50, K_infer=5, hidden=128, eta=1.0)
policy_new.load_state_dict(il_state)

# Small perturbation to new policy (simulating a few gradient steps)
with torch.no_grad():
    for p in policy_new.parameters():
        p.add_(torch.randn_like(p) * 0.001)

print("=" * 60)
print("Per-step ratio analysis (eta=1.0, K=5)")
print("=" * 60)

G = 16
obs = torch.randn(1, env.obs_dim, device=DEVICE).expand(G, -1).contiguous()

with torch.no_grad():
    actions, a_k_inputs, a_km1_outputs, means_old, sigmas = policy_old.sample_with_path(obs)

print(f"\nSigmas: {sigmas.tolist()}")
print(f"Sigma range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")

K = a_k_inputs.shape[1]
total_log_r = torch.zeros(G, device=DEVICE)

for k in range(K):
    a_k = a_k_inputs[:, k]
    a_km1 = a_km1_outputs[:, k]
    mu_old = means_old[:, k]
    sigma_k = sigmas[k].item()

    mu_new = policy_new.compute_mean_at_step(a_k, k, obs)

    lr_k = per_step_log_ratio(a_km1, mu_old, mu_new, sigma_k)

    # Detailed diagnostics
    diff_old_sq = (a_km1 - mu_old).pow(2).sum(dim=-1)
    diff_new_sq = (a_km1 - mu_new).pow(2).sum(dim=-1)
    mu_diff = (mu_old - mu_new).pow(2).sum(dim=-1).sqrt()

    print(f"\nStep {k} (sigma={sigma_k:.6f}):")
    print(f"  ||a_km1 - mu_old||²: mean={diff_old_sq.mean():.6f}")
    print(f"  ||a_km1 - mu_new||²: mean={diff_new_sq.mean():.6f}")
    print(f"  ||mu_old - mu_new||:  mean={mu_diff.mean():.6f}")
    print(f"  1/(2σ²) = {0.5 / (sigma_k**2 + 1e-8):.4f}")
    print(f"  log_r_k: mean={lr_k.mean():.6f}, std={lr_k.std():.6f}")
    print(f"  log_r_k range: [{lr_k.min():.4f}, {lr_k.max():.4f}]")

    total_log_r += lr_k

print(f"\n{'='*60}")
print(f"Total log-ratio: mean={total_log_r.mean():.4f}, std={total_log_r.std():.4f}")
print(f"Total log-ratio range: [{total_log_r.min():.4f}, {total_log_r.max():.4f}]")
ratio = torch.exp(total_log_r.clamp(-10, 10))
print(f"Ratio: mean={ratio.mean():.6f}, range=[{ratio.min():.6f}, {ratio.max():.6f}]")

print(f"\nDiagnosis:")
if total_log_r.mean() < -5:
    print("  PROBLEM: Total log-ratio is very negative → ratio ≈ 0")
    print("  The product of K ratios underflows because each step contributes")
    print("  a moderately negative log-ratio that accumulates.")
    print()
    print("  Potential fixes:")
    print("  1. Use fewer denoising steps K (reduce accumulation)")
    print("  2. Normalize per-step log-ratios by K (geometric mean)")
    print("  3. Use per-step PPO (DPPO-style) instead of product ratio")
    print("  4. Clamp per-step log-ratios more aggressively")
