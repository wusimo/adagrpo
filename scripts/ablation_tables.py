#!/usr/bin/env python3
"""Generate ablation tables by directly importing the WORKING code."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import the KNOWN WORKING classes and functions
from grpo_diffusion_test import (
    DiffPolicy, ReachingEnv,
    strategy_A_product_ratio, strategy_B_per_step_ppo, strategy_C_mean_log_ratio,
    run_one, DEVICE
)
import copy, json, numpy as np, torch
from pathlib import Path

RESULTS_DIR = Path("results/synthetic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {}

# ---- Table 1: Main results (4 tasks) ----
print("="*70)
print("TABLE 1: Main Synthetic Results")
print("="*70)

# Reaching (from run_one)
print("\n  reaching_2d (sigma_min=0.05):")
il, gr = run_one("product", strategy_A_product_ratio, sigma_min=0.05)
results['reaching_2d'] = {'il': il, 'grpo': gr}

# Biased expert
print("\n  biased_expert:")
import torch
class BiasedEnv:
    obs_dim, action_dim = 4, 2
    def get_target(self, obs): return obs[:, :2] * 0.5
    def reward(self, obs, action):
        return -(action - self.get_target(obs)).pow(2).sum(-1)

torch.manual_seed(42)
env_b = BiasedEnv()
policy_b = DiffPolicy(env_b.action_dim, env_b.obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)
policy_b.train()
opt = torch.optim.Adam(policy_b.parameters(), lr=1e-3)
for s in range(2000):
    o = torch.randn(64, env_b.obs_dim, device=DEVICE)
    # BIASED expert: +0.3 offset
    e = env_b.get_target(o) + 0.3 + torch.randn(64, env_b.action_dim, device=DEVICE) * 0.05
    l = policy_b.denoising_loss(o, e); opt.zero_grad(); l.backward(); opt.step()
policy_b.eval()
with torch.no_grad(): il_b = env_b.reward(torch.randn(500,4,device=DEVICE), policy_b.sample(torch.randn(500,4,device=DEVICE))).mean().item()
il_sd_b = copy.deepcopy(policy_b.state_dict())

# GRPO on biased
G,clip_eps,lr,iters,ref_freq,clamp_v,num_groups = 8,0.2,3e-5,300,5,2.0,4
policy_b.load_state_dict(il_sd_b)
po_b = DiffPolicy(env_b.action_dim, env_b.obs_dim, T=50, K=5, hidden=128, eta=1.0, sigma_min=0.05)
po_b.load_state_dict(il_sd_b); po_b.eval()
rl_opt = torch.optim.Adam(policy_b.parameters(), lr=lr)
import torch.nn as nn
for it in range(iters):
    policy_b.train()
    tl = torch.tensor(0., device=DEVICE); nu = 0
    for _ in range(num_groups):
        o = torch.randn(1, env_b.obs_dim, device=DEVICE).expand(G,-1).contiguous()
        with torch.no_grad(): a,inp,out,mo,sig = po_b.sample_with_path(o)
        rw = env_b.reward(o, a)
        if rw.std() < 1e-8: continue
        adv = (rw - rw.mean())/(rw.std()+1e-8); nu += 1
        loss, ratio = strategy_A_product_ratio(policy_b, po_b, o, a, inp, out, mo, sig, adv, clip_eps)
        tl = tl + loss
    if nu == 0: continue
    tl = tl / nu
    rl_opt.zero_grad(); tl.backward()
    nn.utils.clip_grad_norm_(policy_b.parameters(), 1.0); rl_opt.step()
    if (it+1)%ref_freq==0: po_b.load_state_dict(policy_b.state_dict())
    if (it+1)%100==0:
        policy_b.eval()
        with torch.no_grad(): er=env_b.reward(torch.randn(500,4,device=DEVICE), policy_b.sample(torch.randn(500,4,device=DEVICE))).mean().item()
        print(f"    Iter {it+1}: {er:.4f}")
policy_b.eval()
with torch.no_grad(): gr_b = env_b.reward(torch.randn(1000,4,device=DEVICE), policy_b.sample(torch.randn(1000,4,device=DEVICE))).mean().item()
d = gr_b - il_b
print(f"  biased: IL={il_b:.4f} GRPO={gr_b:.4f} D={d:+.4f} ({d/abs(il_b)*100:+.1f}%)")
results['biased_expert'] = {'il': il_b, 'grpo': gr_b}

# ---- Table 2: Sigma_min sweep ----
print(f"\n{'='*70}")
print("TABLE 2: Sigma_min Ablation")
print(f"{'='*70}")
for sm in [0.01, 0.05, 0.1, 0.2]:
    il, gr = run_one("product", strategy_A_product_ratio, sigma_min=sm)
    results[f'sm_{sm}'] = {'il': il, 'grpo': gr}

# ---- Table 3: Strategy comparison ----
print(f"\n{'='*70}")
print("TABLE 3: Ratio Strategy Comparison (sigma_min=0.1)")
print(f"{'='*70}")
il_a, gr_a = run_one("product", strategy_A_product_ratio, sigma_min=0.1)
results['strat_product'] = {'il': il_a, 'grpo': gr_a}
il_b2, gr_b2 = run_one("per_step", strategy_B_per_step_ppo, sigma_min=0.1)
results['strat_perstep'] = {'il': il_b2, 'grpo': gr_b2}
il_c, gr_c = run_one("mean", strategy_C_mean_log_ratio, sigma_min=0.1)
results['strat_mean'] = {'il': il_c, 'grpo': gr_c}

# ---- Summary ----
print(f"\n{'='*70}")
print("ALL RESULTS")
print(f"{'='*70}")
print(f"{'Config':<25} {'IL':>10} {'GRPO':>10} {'Δ':>10} {'%':>8}")
print("-"*63)
for k, v in results.items():
    d = v['grpo'] - v['il']
    p = d/abs(v['il'])*100 if v['il'] != 0 else 0
    print(f"{k:<25} {v['il']:>10.4f} {v['grpo']:>10.4f} {d:>+10.4f} {p:>+7.1f}%")

with open(RESULTS_DIR / "ablation_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS_DIR}/ablation_results.json")
