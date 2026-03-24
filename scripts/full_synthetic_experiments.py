#!/usr/bin/env python3
"""Complete synthetic experiments — calls the working code directly.

For each experiment, we call run_one() from grpo_diffusion_test.py with
minimal modifications, ensuring we get the SAME results that were verified.
"""

import copy, json, sys, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/synthetic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import working building blocks
from grpo_diffusion_test import (
    DiffPolicy, ReachingEnv,
    strategy_A_product_ratio, strategy_B_per_step_ppo, strategy_C_mean_log_ratio,
)


def run_one_with_logging(env, expert_fn, strategy_fn, sigma_min=0.05, eta=1.0,
                          K=5, il_steps=5000, rl_iters=300, G=8, lr_rl=3e-5,
                          ref_freq=5, num_groups=4, clip_eps=0.2):
    """Exact copy of the working run_one() but with learning curve logging."""
    torch.manual_seed(42)

    obs_fn = lambda n: torch.randn(n, env.obs_dim, device=DEVICE)

    policy = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=K,
                         hidden=128, eta=eta, sigma_min=sigma_min)

    # IL pretrain
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    # Train until convergence or max steps
    best_loss = float('inf')
    for step in range(il_steps):
        obs = obs_fn(64)
        expert = expert_fn(obs)
        loss = policy.denoising_loss(obs, expert)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
        # Early convergence: if loss is very low, we're done
        if step > 1000 and best_loss < 0.01:
            break

    policy.eval()
    with torch.no_grad():
        _eo = obs_fn(500); il_reward = env.reward(_eo, policy.sample(_eo)).mean().item()
    il_state = copy.deepcopy(policy.state_dict())

    # GRPO
    policy.load_state_dict(il_state)
    policy_old = DiffPolicy(env.action_dim, env.obs_dim, T=50, K=K,
                              hidden=128, eta=eta, sigma_min=sigma_min)
    policy_old.load_state_dict(il_state); policy_old.eval()
    rl_opt = torch.optim.Adam(policy.parameters(), lr=lr_rl)

    log = defaultdict(list)

    for iteration in range(rl_iters):
        policy.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_used = 0; iter_rewards = []; iter_ratios = []

        for _ in range(num_groups):
            obs = obs_fn(1).expand(G, -1).contiguous()
            with torch.no_grad():
                actions, inp, out, means_old, sigmas = policy_old.sample_with_path(obs)
            rewards = env.reward(obs, actions)
            iter_rewards.extend(rewards.tolist())
            if rewards.std() < 1e-8: continue
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            n_used += 1

            loss, ratio = strategy_fn(policy, policy_old, obs, actions,
                                       inp, out, means_old, sigmas, adv, clip_eps)
            total_loss = total_loss + loss
            iter_ratios.extend(ratio.detach().cpu().tolist())

        if n_used == 0: continue
        total_loss = total_loss / n_used
        rl_opt.zero_grad(); total_loss.backward()
        gn = sum(p.grad.norm().item()**2 for p in policy.parameters()
                 if p.grad is not None)**0.5
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        rl_opt.step()

        if (iteration + 1) % ref_freq == 0:
            policy_old.load_state_dict(policy.state_dict())

        log['reward'].append(float(np.mean(iter_rewards)))
        log['grad'].append(float(gn))
        log['ratio_mean'].append(float(np.mean(iter_ratios)) if iter_ratios else 1.0)
        log['ratio_std'].append(float(np.std(iter_ratios)) if iter_ratios else 0.0)

        if (iteration + 1) % 10 == 0:
            policy.eval()
            with torch.no_grad():
                _eo2 = obs_fn(500); er = env.reward(_eo2, policy.sample(_eo2)).mean().item()
            log['eval_r'].append(float(er))
            log['eval_it'].append(iteration + 1)

    policy.eval()
    with torch.no_grad():
        _eo3 = obs_fn(1000); final_r = env.reward(_eo3, policy.sample(_eo3)).mean().item()

    return il_reward, final_r, dict(log)


# ============================================================
# Environments and expert functions
# ============================================================

class BiasedExpertEnv:
    obs_dim, action_dim = 4, 2
    def get_target(self, obs): return obs[:, :2] * 0.5
    def reward(self, obs, action): return -(action - self.get_target(obs)).pow(2).sum(-1)

class HighDim7DEnv:
    obs_dim, action_dim = 10, 7
    def __init__(self):
        torch.manual_seed(999); self.W = torch.randn(10, 7, device=DEVICE) * 0.3
    def get_target(self, obs): return obs @ self.W
    def reward(self, obs, action): return -(action - self.get_target(obs)).pow(2).sum(-1)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    all_results = {}
    env = ReachingEnv()

    def reach_expert(o):
        return env.get_target(o) + torch.randn(o.shape[0], 2, device=DEVICE) * 0.1

    # ================================================================
    # TABLE 1: Main results — AdaGRPO vs DPPO-style on Reaching
    # ================================================================
    print("=" * 70)
    print("TABLE 1: AdaGRPO (product) vs DPPO (per-step) vs Mean ratio")
    print("=" * 70)

    il_a, gr_a, log_a = run_one_with_logging(env, reach_expert, strategy_A_product_ratio, sigma_min=0.05)
    print(f"  Product:  IL={il_a:.4f} → GRPO={gr_a:.4f} ({(gr_a-il_a)/abs(il_a)*100:+.1f}%)")

    il_b, gr_b, log_b = run_one_with_logging(env, reach_expert, strategy_B_per_step_ppo, sigma_min=0.05)
    print(f"  PerStep:  IL={il_b:.4f} → DPPO={gr_b:.4f} ({(gr_b-il_b)/abs(il_b)*100:+.1f}%)")

    il_c, gr_c, log_c = run_one_with_logging(env, reach_expert, strategy_C_mean_log_ratio, sigma_min=0.05)
    print(f"  Mean:     IL={il_c:.4f} → Mean={gr_c:.4f} ({(gr_c-il_c)/abs(il_c)*100:+.1f}%)")

    all_results['main_reaching'] = {
        'product': {'il': il_a, 'final': gr_a, 'log': log_a},
        'per_step': {'il': il_b, 'final': gr_b, 'log': log_b},
        'mean': {'il': il_c, 'final': gr_c, 'log': log_c},
    }

    # ================================================================
    # TABLE 1 continued: Biased Expert
    # ================================================================
    print(f"\n  --- Biased Expert ---")
    env_b = BiasedExpertEnv()
    biased_expert = lambda o: env_b.get_target(o) + 0.3 + torch.randn(o.shape[0], 2, device=DEVICE) * 0.05

    il_ba, gr_ba, log_ba = run_one_with_logging(env_b, biased_expert, strategy_A_product_ratio, sigma_min=0.05)
    print(f"  Biased Product: IL={il_ba:.4f} → GRPO={gr_ba:.4f} ({(gr_ba-il_ba)/abs(il_ba)*100:+.1f}%)")

    il_bb, gr_bb, log_bb = run_one_with_logging(env_b, biased_expert, strategy_B_per_step_ppo, sigma_min=0.05)
    print(f"  Biased PerStep: IL={il_bb:.4f} → DPPO={gr_bb:.4f} ({(gr_bb-il_bb)/abs(il_bb)*100:+.1f}%)")

    all_results['main_biased'] = {
        'product': {'il': il_ba, 'final': gr_ba, 'log': log_ba},
        'per_step': {'il': il_bb, 'final': gr_bb, 'log': log_bb},
    }

    # 7D
    print(f"\n  --- 7D Actions ---")
    env_7 = HighDim7DEnv()
    exp7 = lambda o: env_7.get_target(o) + 0.1 + torch.randn(o.shape[0], 7, device=DEVICE) * 0.15

    il_7a, gr_7a, log_7a = run_one_with_logging(env_7, exp7, strategy_A_product_ratio,
                                                  sigma_min=0.05, il_steps=5000)
    print(f"  7D Product: IL={il_7a:.4f} → GRPO={gr_7a:.4f} ({(gr_7a-il_7a)/abs(il_7a)*100:+.1f}%)")

    il_7b, gr_7b, log_7b = run_one_with_logging(env_7, exp7, strategy_B_per_step_ppo,
                                                  sigma_min=0.05, il_steps=5000)
    print(f"  7D PerStep: IL={il_7b:.4f} → DPPO={gr_7b:.4f} ({(gr_7b-il_7b)/abs(il_7b)*100:+.1f}%)")

    all_results['main_7d'] = {
        'product': {'il': il_7a, 'final': gr_7a, 'log': log_7a},
        'per_step': {'il': il_7b, 'final': gr_7b, 'log': log_7b},
    }

    # ================================================================
    # TABLE 2: Sigma_min ablation
    # ================================================================
    print(f"\n{'='*70}")
    print("TABLE 2: sigma_min ablation (Reaching, product ratio)")
    print(f"{'='*70}")
    sm_res = {}
    for sm in [0.01, 0.05, 0.1, 0.2]:
        il_v, gr_v, lg = run_one_with_logging(env, reach_expert, strategy_A_product_ratio,
                                               sigma_min=sm, rl_iters=200)
        p = (gr_v - il_v) / abs(il_v) * 100 if il_v != 0 else 0
        print(f"  σ_min={sm}: IL={il_v:.4f} GRPO={gr_v:.4f} ({p:+.1f}%)")
        sm_res[str(sm)] = {'il': il_v, 'grpo': gr_v, 'log': lg}
    all_results['sigma_min'] = sm_res

    # ================================================================
    # TABLE 3: Eta ablation
    # ================================================================
    print(f"\n{'='*70}")
    print("TABLE 3: eta ablation (Reaching, product ratio, σ_min=0.05)")
    print(f"{'='*70}")
    eta_res = {}
    for eta in [0.0, 0.3, 0.5, 1.0]:
        il_v, gr_v, lg = run_one_with_logging(env, reach_expert, strategy_A_product_ratio,
                                               sigma_min=0.05, eta=eta, rl_iters=200)
        ag = np.mean(lg['grad']) if lg['grad'] else 0
        arm = np.mean(lg['ratio_mean']) if lg['ratio_mean'] else 1
        p = (gr_v - il_v) / abs(il_v) * 100 if il_v != 0 else 0
        print(f"  η={eta}: IL={il_v:.4f} GRPO={gr_v:.4f} grad={ag:.3f} ratio={arm:.3f} ({p:+.1f}%)")
        eta_res[str(eta)] = {'il': il_v, 'grpo': gr_v, 'avg_grad': ag, 'avg_ratio': arm, 'log': lg}
    all_results['eta'] = eta_res

    # ================================================================
    # TABLE 4: K scaling
    # ================================================================
    print(f"\n{'='*70}")
    print("TABLE 4: K scaling (Reaching, product ratio, σ_min=0.05)")
    print(f"{'='*70}")
    k_res = {}
    for K in [3, 5, 8, 10]:
        il_v, gr_v, lg = run_one_with_logging(env, reach_expert, strategy_A_product_ratio,
                                               sigma_min=0.05, K=K, rl_iters=200)
        ars = np.mean(lg['ratio_std']) if lg['ratio_std'] else 0
        p = (gr_v - il_v) / abs(il_v) * 100 if il_v != 0 else 0
        print(f"  K={K}: IL={il_v:.4f} GRPO={gr_v:.4f} ratio_std={ars:.4f} ({p:+.1f}%)")
        k_res[str(K)] = {'il': il_v, 'grpo': gr_v, 'ratio_std': ars, 'log': lg}
    all_results['K'] = k_res

    # ================================================================
    # SAVE
    # ================================================================
    elapsed = time.time() - t0

    def clean(obj):
        if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [clean(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)): return float(obj)
        elif isinstance(obj, defaultdict): return clean(dict(obj))
        return obj

    with open(RESULTS_DIR / "full_results.json", 'w') as f:
        json.dump(clean(all_results), f, indent=2)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"COMPLETE SUMMARY ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    print("\n--- Table 1: Main Results ---")
    print(f"{'Task':<15} {'IL':>10} {'AdaGRPO':>10} {'DPPO':>10} {'GRPO%':>8} {'DPPO%':>8}")
    print("-" * 62)
    for task, key in [('Reaching', 'main_reaching'), ('Biased', 'main_biased'), ('7D', 'main_7d')]:
        r = all_results[key]
        il = r['product']['il']
        g = r['product']['final']
        d = r['per_step']['final']
        gp = (g-il)/abs(il)*100 if il != 0 else 0
        dp = (d-il)/abs(il)*100 if il != 0 else 0
        print(f"{task:<15} {il:>10.4f} {g:>10.4f} {d:>10.4f} {gp:>+7.1f}% {dp:>+7.1f}%")

    print("\n--- Table 2: σ_min ---")
    for sm, r in all_results['sigma_min'].items():
        p = (r['grpo']-r['il'])/abs(r['il'])*100 if r['il'] != 0 else 0
        print(f"  σ_min={sm:<5} IL={r['il']:.4f} GRPO={r['grpo']:.4f} ({p:+.1f}%)")

    print("\n--- Table 3: η ---")
    for eta, r in all_results['eta'].items():
        p = (r['grpo']-r['il'])/abs(r['il'])*100 if r['il'] != 0 else 0
        print(f"  η={eta:<4} IL={r['il']:.4f} GRPO={r['grpo']:.4f} grad={r['avg_grad']:.3f} ({p:+.1f}%)")

    print("\n--- Table 4: K ---")
    for K, r in all_results['K'].items():
        p = (r['grpo']-r['il'])/abs(r['il'])*100 if r['il'] != 0 else 0
        print(f"  K={K:<3} IL={r['il']:.4f} GRPO={r['grpo']:.4f} ratio_std={r['ratio_std']:.4f} ({p:+.1f}%)")

    print(f"\nResults saved to {RESULTS_DIR}/full_results.json")


if __name__ == "__main__":
    main()
