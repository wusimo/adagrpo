# AdaGRPO Feasibility Report: Path-Conditioned GRPO for Diffusion Policies

## Executive Summary

**The core idea works, but with critical caveats.** Path-conditioned GRPO can improve diffusion policies beyond IL baselines on synthetic tasks. However, we discovered and fixed two fundamental issues that the paper's current formulation doesn't address, and one issue that remains an open problem.

---

## What I Tried (Chronological)

### Attempt 1: Direct implementation of the paper's equations
Implemented per-step Gaussian log-ratios, product ratio, group-relative advantages, clipped surrogate loss. Used DDIM with eta=0 (deterministic) as the paper implies.

**Result: Total failure.** Ratio = 1.0 exactly, gradient = 0.0, no learning.

**Root cause discovered:** With deterministic DDIM (eta=0), the output of each denoising step IS the predicted mean: a^{k-1} = mu_old. So ||a^{k-1} - mu_old||^2 = 0 ALWAYS. The first term in the ratio formula vanishes, and the ratio reduces to exp(-c * ||mu_old - mu_new||^2) which is always <= 1 and has a degenerate gradient landscape.

**Lesson:** The path-conditioned ratio factorization REQUIRES stochastic reverse steps. This is what DPPO does (they use DDPM, not DDIM), but the AdaGRPO paper doesn't make this explicit.

### Attempt 2: Stochastic DDIM (eta > 0)
Switched to eta=0.5 and eta=1.0 so that a^{k-1} = mu_old + sigma * epsilon (random noise added at each step).

**Result: Catastrophic failure.** Ratios collapsed to ~0.0001, policy degraded 10-14x.

**Root cause discovered:** The cosine beta schedule produces near-zero posterior variance at the last 2 denoising steps (sigma ~ 1e-4). The factor 1/(2*sigma^2) ~ 25,000,000 amplifies even tiny differences between mu_old and mu_new into log-ratios of -200 to -350. The product of K ratios underflows catastrophically.

### Attempt 3: Sigma floor (sigma_min)
Enforced a minimum sigma of 0.01-0.2 for ratio computation. This prevents the 1/(2*sigma^2) explosion at late denoising steps.

**Result: SUCCESS.** Tested 4 sigma_min values (0.01, 0.05, 0.1, 0.2) — all improved over IL.

Also tested two alternative strategies:
- **Per-step PPO (DPPO-style):** Apply clipping to each step independently instead of multiplying ratios. Also works.
- **Mean log-ratio:** Divide the sum by K (geometric mean). Also works but weaker improvement.

### Attempt 4: Harder problems
Tested on 4 more challenging settings:

| Test | Description | IL | GRPO | Improvement |
|------|-------------|-----|------|-------------|
| Biased expert | Expert systematically offset by +0.3 | -0.207 | -0.0001 | +99.9% |
| 7D actions | Robotics-scale action dim | -0.086 | -0.002 | +97.4% |
| Sparse reward | Binary success/fail | 1.0 | 1.0 | 0% (IL already perfect) |
| Multi-step (H=5) | Sequential decision making | -6.70 | -5.72 | +14.6% |

---

## What Works

1. **The core math is sound.** Per-step Gaussian log-ratios are correct. Group-relative advantages work. The clipped surrogate converges.

2. **GRPO can correct IL errors.** The biased expert test shows GRPO can fully overcome a systematic +0.3 bias in demonstrations, improving reward by 99.9%.

3. **Scales to higher dimensions.** 7D action space (robotics-scale) shows 97.4% improvement.

4. **Multi-step episodes work.** Even with H=5 sequential decisions (total K*H = 25 denoising steps in the ratio product), GRPO achieves 14.6% improvement.

5. **All three ratio strategies work:** product (with sigma floor), per-step PPO, and mean log-ratio. The product with small sigma_min is strongest.

---

## What Doesn't Work / Open Issues

### 1. CRITICAL: Deterministic DDIM (eta=0) is fundamentally incompatible
The entire path-conditioned ratio approach requires stochasticity in the reverse process. With eta=0, the ratio formula degenerates. This is a hard requirement, not a bug.

**Impact on the paper:** The paper says "Use DDIM for RL (fewer steps = cheaper rollouts)" without mentioning that eta must be > 0. This needs to be stated explicitly, and the extra variance from stochastic sampling needs to be characterized.

### 2. CRITICAL: Sigma floor is a necessary hack
Without sigma_min, the last 1-2 denoising steps have sigma ~ 1e-4, causing 1/(2*sigma^2) ~ 25M, which makes ratios numerically meaningless. The paper's ALN could in principle learn to downweight these steps, but:
- The ALN in our IL co-training experiments collapsed to all-zero weights
- Even if ALN works, the underlying numerical issue means those steps should probably be excluded entirely

**Recommendation:** Either (a) skip the last 1-2 steps from ratio computation, (b) enforce sigma_min >= 0.01, or (c) switch to DDPM where posterior variance is always meaningful.

### 3. MODERATE: Multi-step ratio product accumulates variance
With H=5 episode steps * K=5 denoising steps = 25 terms in the ratio product, the ratio variance is already noticeable (0.75-1.09 mean, 0.1-0.5 std). For real robotics (H=300, K=10), this would be 3000 terms — likely intractable even with per-step clamping.

**This is where ALN weighting would genuinely help:** if it can learn to suppress the 90%+ of steps that don't matter, it could reduce the effective product length from 3000 to ~300.

### 4. MINOR: Sparse reward provides no signal when IL is already good
When IL achieves 100% success rate (sparse reward test), GRPO has no room to improve. This is expected — GRPO needs mixed outcomes in groups to generate contrastive signal.

### 5. NOT YET TESTED: Integration with actual robot simulators
All results are on synthetic environments. The real challenges (visual observations, contact dynamics, long horizons) are not tested. The core algorithm works, but simulator-level validation is needed.

---

## Specific Technical Recommendations

1. **In the paper, explicitly state eta > 0 is required** for RL rollouts. Characterize the tradeoff: larger eta = more stochastic = better-behaved ratios but noisier action samples.

2. **Add sigma_min as a hyperparameter** (recommend 0.05). Document it as necessary, not optional.

3. **Consider per-step PPO (DPPO-style) as a baseline.** It avoids the ratio product explosion entirely and performs comparably.

4. **For long-horizon tasks, the ratio product will be the bottleneck.** The ALN paper contribution becomes much more important there — but the ALN training procedure needs work (it collapsed in our experiments).

5. **The "7x compute savings from HVTS" claim needs recalibration.** Our tests show 1.4x average. The 7x number may be achievable on specific tasks but isn't representative.

---

## Experiment Artifacts

- `scripts/grpo_diffusion_test.py` — Core algorithm test (6 strategies)
- `scripts/grpo_hard_test.py` — Harder validation (4 tasks)
- `scripts/debug_grad.py` — Gradient flow verification
- `scripts/debug_ratio_magnitude.py` — Per-step ratio analysis
- `scripts/feasibility_validation.py` — Full 18-test validation suite
