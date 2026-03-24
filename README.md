# AdaGRPO: Adaptive Group Relative Policy Optimization for Diffusion-Based VLA Models

This repository contains the implementation and experiments for **AdaGRPO**, a framework that makes GRPO (Group Relative Policy Optimization) tractable and efficient for diffusion-based Vision-Language-Action (VLA) action heads.

## Key Idea

GRPO requires an importance ratio $\pi_\theta(a|s) / \pi_{\theta_{old}}(a|s)$ which is **intractable** for diffusion policies because marginalizing over the denoising chain is a high-dimensional integral. We decompose this into a product of per-step Gaussian ratios (following DPPO) but replace the value function with GRPO's group-relative advantages, preserving the **value-function-free** simplicity.

### Critical Findings

Through systematic experimentation, we identified two requirements that prior work leaves implicit:

1. **Stochastic reverse sampling is mandatory** — Deterministic DDIM (eta=0) makes the ratio formula degenerate (ratio = 1.0 exactly, zero gradient). You *must* use eta > 0 for RL rollouts.

2. **A per-step variance floor is necessary** — The cosine schedule produces near-zero posterior variance at late denoising steps, causing `1/(2*sigma^2)` to explode to ~10^7. A minimum sigma (e.g. 0.05) prevents ratio collapse.

## Results

### Synthetic Tasks

| Task | IL Baseline | AdaGRPO | DPPO-style | Improvement |
|------|------------|---------|------------|-------------|
| Reaching (2D) | -0.0049 | **-0.0001** | -0.0001 | +98.2% |
| Biased Expert (2D) | -0.1655 | **-0.0002** | -0.0002 | +99.9% |
| High-Dim (7D) | -0.0886 | **-0.0017** | -0.0017 | +98.0% |
| Multi-Step (H=5) | -6.70 | **-5.72** | — | +14.6% |

### Robosuite Lift (Real Robot Simulator)

| Method | Reward | Improvement |
|--------|--------|-------------|
| IL (200 human demos) | 0.992 | — |
| AdaGRPO (100 iters) | **1.506** | **+51.8%** |

### Ablations

| sigma_min | Improvement | | eta | Improvement | | K | Improvement |
|-----------|-------------|---|-----|-------------|---|---|-------------|
| 0.01 | +99.1% | | 0.0 | +96.3%* | | 3 | +92.4% |
| 0.05 | +98.1% | | 0.3 | +97.7% | | 5 | +98.1% |
| 0.10 | +94.7% | | 0.5 | +97.8% | | 8 | +97.6% |
| 0.20 | +90.5% | | 1.0 | +98.1% | | 10 | +96.0% |

*eta=0 works only when sigma_min > 0 provides a fallback stochasticity.

## Installation

```bash
pip install -e .
```

For robosuite experiments:
```bash
pip install robosuite robomimic
# Download demo data:
python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph
```

## Quick Start

### Run the core validation (synthetic, ~3 min)
```bash
python scripts/grpo_diffusion_test.py
```

### Run all ablations with DPPO comparison (~3 min)
```bash
python scripts/full_synthetic_experiments.py
```

### Run robosuite Lift experiment (~15 min, needs demo data)
```bash
python scripts/robosuite_lift.py
```

### Run harder tasks (biased expert, 7D, multi-step, ~5 min)
```bash
python scripts/grpo_hard_test.py
```

### Generate paper figures
```bash
python scripts/generate_plots.py
```

### Compile the paper
```bash
cd paper && tectonic main.tex
```

## Project Structure

```
adagrpo/
├── adagrpo/                    # Core library
│   ├── core/                   # Algorithm components
│   │   ├── ratio.py            # Per-step Gaussian log-ratio computation
│   │   ├── aln.py              # Adaptive Loss Network (importance weights)
│   │   ├── grpo.py             # AdaGRPO clipped surrogate loss
│   │   ├── advantages.py       # Group-relative advantage computation
│   │   └── group_sampler.py    # Hard-trajectory mining
│   ├── policy/                 # Policy implementations
│   │   ├── action_head.py      # Abstract interface + DenoisingPath dataclass
│   │   ├── diffusion_policy.py # Diffusion policy with path recording
│   │   └── flow_policy.py      # Flow-matching variant
│   ├── scheduling/             # Stage-aware rollout scheduling
│   │   ├── hvts.py             # Hierarchical Vision Task Segmenter
│   │   ├── stage_classifier.py # Distilled stage classifier
│   │   └── budget_allocator.py # Dynamic N_d, N_a allocation
│   ├── envs/                   # Environment wrappers
│   │   ├── libero_wrapper.py
│   │   ├── robomimic_wrapper.py
│   │   └── maniskill_wrapper.py
│   ├── training/               # Training loops
│   │   ├── il_trainer.py       # IL pretraining with ALN co-training
│   │   ├── rl_trainer.py       # AdaGRPO RL post-training
│   │   ├── rollout.py          # Stage-aware rollout collection
│   │   └── metrics.py          # Success rate, ratio diagnostics
│   └── utils/                  # Utilities
│       ├── diffusion_utils.py  # DDPM/DDIM schedulers
│       ├── logging.py          # Console + W&B logging
│       └── checkpointing.py    # Save/load helpers
├── configs/                    # Hydra configs
│   ├── algo/                   # adagrpo, dppo, vanilla_grpo, fpo
│   ├── env/                    # libero, robomimic, maniskill
│   └── policy/                 # diffusion_cnn, diffusion_transformer, flow_matching
├── scripts/                    # Entry points
│   ├── grpo_diffusion_test.py  # Core validation (verified working)
│   ├── grpo_hard_test.py       # Harder tasks (biased, 7D, multi-step)
│   ├── full_synthetic_experiments.py  # All ablations + DPPO baseline
│   ├── robosuite_lift.py       # Robosuite Lift with robomimic demos
│   ├── generate_plots.py       # Paper figures
│   ├── feasibility_validation.py  # 18-test validation suite
│   ├── train_il.py             # Hydra-based IL training
│   ├── train_rl.py             # Hydra-based RL training
│   └── eval.py                 # Evaluation
├── tests/                      # Unit tests
├── paper/                      # LaTeX paper draft
│   ├── main.tex
│   ├── main.pdf
│   └── figures/                # Generated figures (PDF + PNG)
├── results/                    # Experiment results (JSON)
│   ├── synthetic/
│   └── robosuite/
└── EXPERIMENT_REPORT.md        # Detailed feasibility analysis
```

## Key Equations

**Per-step Gaussian log-ratio** (the tractable building block):

```
log r_k(θ) = (1 / 2σ_k²) * (||a^{k-1} - μ_{θ_old}||² - ||a^{k-1} - μ_θ||²)
```

**Stabilized path-conditioned ratio** (with sigma floor + clamping):

```
log r(θ) = Σ_k clamp( log r_k(θ), -c, c )    where σ_k = max(σ_k, σ_min)
```

**AdaGRPO objective** (value-function-free):

```
L = -E[ min( r(θ) * Â_i, clip(r(θ), 1-ε, 1+ε) * Â_i ) ]
```

where `Â_i = (R_i - mean(R_{1:G})) / (std(R_{1:G}) + δ)` is the group-relative advantage.

## Known Limitations

1. **Multi-step ratio product** accumulates variance: with H=5 steps × K=5 denoising steps = 25 terms, we get +14.6%. For H=300 (real robotics), additional stabilization (ALN weighting, chunk-level factorization) is needed.
2. **Auxiliary denoising loss** is important to prevent policy drift during RL fine-tuning.
3. **sigma_min** is a manually tuned hyperparameter; learned per-step weights (ALN) would be more principled but our ALN training collapsed in early experiments.

## Citation

```bibtex
@article{adagrpo2025,
  title={AdaGRPO: Adaptive Group Relative Policy Optimization for Diffusion-Based Vision-Language-Action Models},
  author={Anonymous},
  year={2025}
}
```
