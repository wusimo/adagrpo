#!/usr/bin/env python3
"""Generate paper figures from experiment results."""

import json
from pathlib import Path
import numpy as np

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, generating ASCII tables only")

RESULTS_DIR = Path("results/synthetic")
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_DIR / "full_results.json") as f:
    results = json.load(f)


def smooth(y, window=5):
    """Simple moving average."""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid').tolist()


if HAS_MPL:
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

    # ================================================================
    # Figure 1: Learning curves — AdaGRPO vs DPPO vs IL
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (task, key) in enumerate([('Reaching 2D', 'main_reaching'),
                                        ('Biased Expert', 'main_biased'),
                                        ('7D Actions', 'main_7d')]):
        ax = axes[idx]
        r = results[key]

        # Product (AdaGRPO)
        g = r['product']['log']
        ax.plot(g['eval_it'], g['eval_r'], 'b-o', markersize=3, label='AdaGRPO (product)')

        # Per-step (DPPO)
        d = r['per_step']['log']
        ax.plot(d['eval_it'], d['eval_r'], 'r-s', markersize=3, label='DPPO (per-step)')

        # IL baseline
        il = r['product']['il']
        ax.axhline(y=il, color='gray', linestyle='--', label=f'IL baseline ({il:.4f})')

        ax.set_xlabel('RL Iteration')
        ax.set_ylabel('Eval Reward')
        ax.set_title(task)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'learning_curves.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved learning_curves.pdf/png")

    # ================================================================
    # Figure 2: Ratio statistics over training (Reaching)
    # ================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    g = results['main_reaching']['product']['log']

    # Ratio mean
    ax1.plot(smooth(g['ratio_mean']), 'b-', alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Ratio')
    ax1.set_title('Importance Ratio Mean')
    ax1.grid(True, alpha=0.3)

    # Ratio std
    ax2.plot(smooth(g['ratio_std']), 'r-', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Ratio Std')
    ax2.set_title('Importance Ratio Std')
    ax2.grid(True, alpha=0.3)

    # Gradient norm
    ax3.plot(smooth(g['grad']), 'g-', alpha=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm (pre-clip)')
    ax3.set_title('Gradient Norm')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ratio_diagnostics.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'ratio_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"Saved ratio_diagnostics.pdf/png")

    # ================================================================
    # Figure 3: sigma_min ablation
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sm_vals = []
    improvements = []
    for sm, r in results['sigma_min'].items():
        il, grpo = r['il'], r['grpo']
        pct = (grpo - il) / abs(il) * 100
        sm_vals.append(float(sm))
        improvements.append(pct)
    ax.bar(range(len(sm_vals)), improvements, tick_label=[str(s) for s in sm_vals], color='steelblue')
    ax.set_xlabel('$\\sigma_{\\min}$')
    ax.set_ylabel('Improvement over IL (%)')
    ax.set_title('Effect of Variance Floor $\\sigma_{\\min}$')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sigma_min_ablation.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'sigma_min_ablation.png', dpi=150, bbox_inches='tight')
    print(f"Saved sigma_min_ablation.pdf/png")

    # ================================================================
    # Figure 4: K scaling
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ks = []
    grpo_rewards = []
    ratio_stds = []
    for K, r in results['K'].items():
        ks.append(int(K))
        grpo_rewards.append(r['grpo'])
        ratio_stds.append(r['ratio_std'])

    ax1.plot(ks, grpo_rewards, 'bo-', markersize=8)
    ax1.set_xlabel('K (denoising steps)')
    ax1.set_ylabel('GRPO Reward')
    ax1.set_title('Performance vs K')
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, ratio_stds, 'ro-', markersize=8)
    ax2.set_xlabel('K (denoising steps)')
    ax2.set_ylabel('Mean Ratio Std')
    ax2.set_title('Ratio Variance vs K')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'k_scaling.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'k_scaling.png', dpi=150, bbox_inches='tight')
    print(f"Saved k_scaling.pdf/png")

    # ================================================================
    # Figure 5: Eta ablation
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    etas = []
    grads = []
    imprs = []
    for eta, r in results['eta'].items():
        etas.append(float(eta))
        grads.append(r['avg_grad'])
        pct = (r['grpo'] - r['il']) / abs(r['il']) * 100
        imprs.append(pct)

    ax1.bar(range(len(etas)), imprs, tick_label=[str(e) for e in etas], color='coral')
    ax1.set_xlabel('$\\eta$ (DDIM stochasticity)')
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('Performance vs $\\eta$')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(range(len(etas)), grads, tick_label=[str(e) for e in etas], color='seagreen')
    ax2.set_xlabel('$\\eta$')
    ax2.set_ylabel('Mean Gradient Norm')
    ax2.set_title('Gradient Norm vs $\\eta$')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'eta_ablation.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'eta_ablation.png', dpi=150, bbox_inches='tight')
    print(f"Saved eta_ablation.pdf/png")

    print(f"\nAll figures saved to {FIG_DIR}/")

else:
    print("\n=== RESULTS SUMMARY (ASCII) ===")
    print("\nTable 1: Main Results")
    print(f"{'Task':<15} {'IL':>10} {'AdaGRPO':>10} {'DPPO':>10}")
    for task, key in [('Reaching', 'main_reaching'), ('Biased', 'main_biased'), ('7D', 'main_7d')]:
        r = results[key]
        print(f"{task:<15} {r['product']['il']:>10.4f} {r['product']['final']:>10.4f} {r['per_step']['final']:>10.4f}")


if __name__ == "__main__":
    pass
