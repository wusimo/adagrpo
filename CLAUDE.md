# AdaGRPO

Adaptive Group Relative Policy Optimization for Diffusion-Based VLA Models.

## Quick Start

```bash
pip install -e .
python scripts/train_il.py env=libero policy=diffusion_cnn
python scripts/train_rl.py algo=adagrpo env=libero
python scripts/eval.py env=libero checkpoint.resume_from=checkpoints/rl/iter_2000.pt
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

- `adagrpo/core/` — Core algorithms (ratio computation, ALN, GRPO loss, group sampling)
- `adagrpo/policy/` — Diffusion and flow-matching policy wrappers with path recording
- `adagrpo/scheduling/` — HVTS task segmenter, stage classifier, budget allocator
- `adagrpo/envs/` — Environment wrappers (LIBERO, RoboMimic, ManiSkill)
- `adagrpo/training/` — IL and RL training loops, rollout collection, metrics
- `configs/` — Hydra configs (algo, env, policy)
- `scripts/` — Entry points (train_il, setup_hvts, train_rl, eval, ablation)

## Key Design Decisions

- Path-conditioned ratios decompose the intractable marginal into per-step Gaussians (DPPO-style)
- ALN weights suppress uninformative denoising steps to reduce ratio variance
- GRPO group-relative advantages eliminate the need for a value function
- Stage-aware rollout scheduling reduces compute by dynamically adjusting N_d and N_a
- Hard-trajectory mining biases groups toward maximally informative states
