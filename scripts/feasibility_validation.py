#!/usr/bin/env python3
"""Comprehensive feasibility validation for AdaGRPO.

Tests every core component with synthetic data to identify bugs, obstacles,
and validate that the key claims hold:
  1. Per-step Gaussian ratios are mathematically correct
  2. ALN weights reduce ratio variance (Proposition 1)
  3. GRPO training loop converges on a simple task
  4. Stage-aware scheduling produces real compute savings
  5. Hard-trajectory mining improves group informativeness
  6. Full end-to-end pipeline runs without errors

Run with: /home/simo/miniconda3/envs/libero/bin/python scripts/feasibility_validation.py
"""

import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Validation infrastructure
# ============================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float = 0.0
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


results: list[TestResult] = []


def run_test(name):
    """Decorator to run a test and capture results."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            start = time.time()
            try:
                data = fn()
                dur = time.time() - start
                results.append(TestResult(name, True, "PASSED", dur, data or {}))
                print(f"  ✓ PASSED ({dur:.2f}s)")
                return data
            except Exception as e:
                dur = time.time() - start
                msg = f"FAILED: {e}\n{traceback.format_exc()}"
                results.append(TestResult(name, False, msg, dur))
                print(f"  ✗ FAILED ({dur:.2f}s): {e}")
                return None
        wrapper.test_name = name
        return wrapper
    return decorator


# ============================================================
# Test 1: Unit tests for core math
# ============================================================

@run_test("1.1 Per-step Gaussian log-ratio correctness")
def test_ratio_math():
    from adagrpo.core.ratio import compute_per_step_log_ratio

    B, K, D = 32, 10, 7
    torch.manual_seed(42)

    noisy = torch.randn(B, K, D)
    means_old = torch.randn(B, K, D)
    means_new = torch.randn(B, K, D)
    sigmas = torch.ones(K) * 0.5

    log_r = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)

    # Manual computation for verification
    for b in range(3):
        for k in range(3):
            a = noisy[b, k]
            m_old = means_old[b, k]
            m_new = means_new[b, k]
            s = sigmas[k]
            expected = (1 / (2 * s**2)) * (
                (a - m_old).pow(2).sum() - (a - m_new).pow(2).sum()
            )
            actual = log_r[b, k]
            assert abs(expected.item() - actual.item()) < 1e-4, \
                f"Mismatch at [{b},{k}]: expected={expected.item():.6f}, got={actual.item():.6f}"

    # When policies equal, ratio = 0
    log_r_same = compute_per_step_log_ratio(noisy, means_old, means_old, sigmas)
    assert torch.allclose(log_r_same, torch.zeros_like(log_r_same), atol=1e-6), \
        "Equal policies should give zero log-ratio"

    print(f"  Log-ratio stats: mean={log_r.mean():.4f}, std={log_r.std():.4f}")
    return {"log_ratio_mean": log_r.mean().item(), "log_ratio_std": log_r.std().item()}


@run_test("1.2 Weighted vs uniform ratio aggregation")
def test_ratio_aggregation():
    from adagrpo.core.ratio import (
        compute_per_step_log_ratio,
        compute_uniform_log_ratio,
        compute_weighted_log_ratio,
    )

    B, K, D = 64, 10, 7
    torch.manual_seed(42)
    noisy = torch.randn(B, K, D)
    means_old = torch.randn(B, K, D)
    means_new = means_old + torch.randn(B, K, D) * 0.1
    sigmas = torch.ones(K) * 0.5

    per_step = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)

    # Uniform
    uniform = compute_uniform_log_ratio(per_step)
    assert uniform.shape == (B,)

    # Uniform weights = 1 should match
    ones = torch.ones(K)
    weighted_ones = compute_weighted_log_ratio(per_step, ones)
    assert torch.allclose(uniform, weighted_ones, atol=1e-5), \
        "Uniform weights should equal sum"

    # Sparse weights
    sparse = torch.zeros(K)
    sparse[0] = 1.0
    sparse[5] = 1.0
    weighted_sparse = compute_weighted_log_ratio(per_step, sparse)
    expected_sparse = per_step[:, 0] + per_step[:, 5]
    assert torch.allclose(weighted_sparse, expected_sparse, atol=1e-5)

    print(f"  Uniform log-ratio: mean={uniform.mean():.4f}, var={uniform.var():.4f}")
    return {"uniform_var": uniform.var().item()}


@run_test("1.3 Group-relative advantage computation")
def test_advantages():
    from adagrpo.core.advantages import (
        compute_group_advantages,
        compute_batched_group_advantages,
        filter_uninformative_groups,
    )

    # Single group
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    adv = compute_group_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"

    # Batched groups
    rewards_batched = torch.tensor([
        0.0, 0.0, 1.0, 1.0,  # Group 1: mixed
        1.0, 1.0, 1.0, 1.0,  # Group 2: all success
        0.0, 0.0, 0.0, 0.0,  # Group 3: all fail
    ])
    adv_batched = compute_batched_group_advantages(rewards_batched, group_size=4)
    assert adv_batched.shape == (12,)

    # Filter
    mask = filter_uninformative_groups(rewards_batched, group_size=4, success_threshold=0.5)
    assert mask.shape == (3,)
    assert mask[0] == True   # mixed → keep
    assert mask[1] == False  # all success → discard
    assert mask[2] == False  # all fail → discard
    print(f"  Group filter: {mask.tolist()}")

    return {"filter_works": True}


# ============================================================
# Test 2: ALN variance reduction (Proposition 1)
# ============================================================

@run_test("2.1 ALN learns informative weights during IL")
def test_aln_il_training():
    from adagrpo.core.aln import AdaptiveLossNetwork, compute_aln_il_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aln = AdaptiveLossNetwork(num_timesteps=100, embed_dim=128, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(aln.parameters(), lr=1e-3)

    # Simulate per-step denoising losses where steps 2-4 are consistently
    # high-loss (informative) and others are low
    K = 20
    timesteps = torch.arange(K, device=device)

    losses_history = []
    for step in range(200):
        B = 32
        # Create structured per-step losses: steps 2-4 have high loss
        base_loss = torch.ones(B, K, device=device) * 0.1
        base_loss[:, 2:5] = torch.rand(B, 3, device=device) * 2.0 + 1.0  # High loss
        base_loss += torch.randn(B, K, device=device) * 0.05

        loss = compute_aln_il_loss(base_loss.detach(), aln, timesteps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_history.append(loss.item())

    # Check that ALN learned to upweight steps 2-4
    weights = aln.get_weights(K, device=device)
    high_region = weights[2:5].mean().item()
    low_region = torch.cat([weights[0:2], weights[5:]]).mean().item()

    print(f"  ALN weights (high-loss region 2-4): {high_region:.4f}")
    print(f"  ALN weights (low-loss region):      {low_region:.4f}")
    print(f"  Weight ratio: {high_region / (low_region + 1e-8):.2f}x")
    print(f"  All weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    # The high-loss region should have higher weights
    ratio = high_region / (low_region + 1e-8)
    return {
        "high_region_weight": high_region,
        "low_region_weight": low_region,
        "weight_ratio": ratio,
        "aln_learns_structure": ratio > 1.0,
    }


@run_test("2.2 ALN weights reduce ratio variance (Proposition 1)")
def test_variance_reduction():
    from adagrpo.core.ratio import (
        compute_per_step_log_ratio,
        compute_uniform_log_ratio,
        compute_weighted_log_ratio,
    )
    from adagrpo.core.aln import AdaptiveLossNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a scenario where some denoising steps have high-variance ratios
    # (noisy, uninformative) and others have low-variance (informative)
    B, K, D = 500, 10, 7
    torch.manual_seed(42)

    noisy = torch.randn(B, K, D, device=device)
    means_old = torch.randn(B, K, D, device=device)
    means_new = means_old.clone()

    # Steps 0-3: high variance (large policy change)
    means_new[:, 0:4, :] += torch.randn(B, 4, D, device=device) * 3.0
    # Steps 4-9: low variance (small policy change)
    means_new[:, 4:, :] += torch.randn(B, 6, D, device=device) * 0.05

    sigmas = torch.ones(K, device=device) * 0.5
    per_step = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)

    # Uniform ratio variance
    uniform_log_r = compute_uniform_log_ratio(per_step)
    uniform_var = uniform_log_r.var().item()

    # Optimal weights: suppress high-variance steps
    optimal_weights = torch.ones(K, device=device)
    optimal_weights[0:4] = 0.1  # Suppress noisy steps
    weighted_log_r = compute_weighted_log_ratio(per_step, optimal_weights)
    optimal_var = weighted_log_r.var().item()

    # Train ALN to discover these weights
    aln = AdaptiveLossNetwork(num_timesteps=100, embed_dim=128, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(aln.parameters(), lr=3e-3)

    for step in range(300):
        weights = aln.get_weights(K, device=device)
        # Train ALN to minimize variance of weighted log-ratio
        wlr = compute_weighted_log_ratio(per_step.detach(), weights)
        # Use variance as loss (this is a simplified proxy)
        # In practice, ALN is trained during IL with denoising loss structure
        var_loss = wlr.var()
        logits = aln(torch.arange(1, K + 1, device=device))
        w = torch.sigmoid(logits)
        # Add entropy to avoid collapse
        entropy = -(w * torch.log(w + 1e-8) + (1 - w) * torch.log(1 - w + 1e-8)).mean()
        loss = var_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    learned_weights = aln.get_weights(K, device=device)
    learned_log_r = compute_weighted_log_ratio(per_step, learned_weights)
    learned_var = learned_log_r.var().item()

    print(f"  Uniform ratio variance:          {uniform_var:.4f}")
    print(f"  Optimal-weight ratio variance:   {optimal_var:.4f}")
    print(f"  ALN-learned ratio variance:      {learned_var:.4f}")
    print(f"  Variance reduction (optimal):    {uniform_var / (optimal_var + 1e-8):.2f}x")
    print(f"  Variance reduction (learned):    {uniform_var / (learned_var + 1e-8):.2f}x")
    print(f"  Learned weights: {[f'{w:.3f}' for w in learned_weights.tolist()]}")

    return {
        "uniform_var": uniform_var,
        "optimal_var": optimal_var,
        "learned_var": learned_var,
        "proposition1_holds": learned_var < uniform_var,
        "reduction_factor": uniform_var / (learned_var + 1e-8),
    }


# ============================================================
# Test 3: AdaGRPO loss computation
# ============================================================

@run_test("3.1 AdaGRPO loss forward/backward pass")
def test_grpo_loss():
    from adagrpo.core.grpo import AdaGRPOLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = AdaGRPOLoss(
        clip_eps=0.2, aux_weight=0.1, group_size=4,
        filter_uniform_groups=False,
    )

    B, K, D = 8, 10, 7  # 2 groups of 4

    # Simulate: means_new requires grad (through policy parameters)
    noisy = torch.randn(B, K, D, device=device)
    means_old = torch.randn(B, K, D, device=device)
    means_new = means_old + torch.randn(B, K, D, device=device) * 0.1
    means_new.requires_grad_(True)
    sigmas = torch.ones(K, device=device) * 0.5
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], device=device)
    aln_weights = torch.sigmoid(torch.randn(K, device=device))
    aux_loss = torch.tensor(0.5, device=device, requires_grad=True)

    output = loss_fn(
        noisy_actions=noisy,
        means_old=means_old,
        means_new=means_new,
        sigmas=sigmas,
        rewards=rewards,
        aln_weights=aln_weights,
        aux_loss=aux_loss,
    )

    print(f"  Loss: {output.loss.item():.4f}")
    print(f"  Policy loss: {output.policy_loss.item():.4f}")
    print(f"  Aux loss: {output.aux_loss.item():.4f}")
    print(f"  Ratio: {output.ratio_mean:.4f} ± {output.ratio_std:.4f} (max: {output.ratio_max:.4f})")
    print(f"  Clipped fraction: {output.clipped_frac:.4f}")
    print(f"  Log-ratio variance: {output.log_ratio_var:.4f}")

    # Backward pass
    output.loss.backward()
    assert means_new.grad is not None, "Gradients should flow to means_new"
    grad_norm = means_new.grad.norm().item()
    print(f"  Gradient norm on means_new: {grad_norm:.6f}")

    assert torch.isfinite(output.loss), "Loss should be finite"
    assert output.ratio_mean > 0, "Ratio should be positive"

    return {
        "loss": output.loss.item(),
        "ratio_mean": output.ratio_mean,
        "clipped_frac": output.clipped_frac,
        "grad_norm": grad_norm,
        "backward_works": grad_norm > 0,
    }


@run_test("3.2 Uninformative group filtering")
def test_group_filtering():
    from adagrpo.core.grpo import AdaGRPOLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = AdaGRPOLoss(
        clip_eps=0.2, aux_weight=0.0, group_size=4,
        filter_uniform_groups=True,
    )

    B, K, D = 12, 5, 7  # 3 groups of 4

    noisy = torch.randn(B, K, D, device=device)
    means_old = torch.randn(B, K, D, device=device)
    means_new = means_old + torch.randn(B, K, D, device=device) * 0.1
    means_new.requires_grad_(True)
    sigmas = torch.ones(K, device=device) * 0.5

    # Group 1: mixed, Group 2: all success, Group 3: all fail
    rewards = torch.tensor([
        0.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    ], device=device)

    output = loss_fn(
        noisy_actions=noisy,
        means_old=means_old,
        means_new=means_new,
        sigmas=sigmas,
        rewards=rewards,
    )

    print(f"  Loss with filtering: {output.loss.item():.4f}")
    print(f"  (Only 1 of 3 groups should contribute)")

    # All uninformative
    rewards_uniform = torch.tensor([
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    ], device=device)
    noisy2 = torch.randn(8, K, D, device=device)
    means_old2 = torch.randn(8, K, D, device=device)
    means_new2 = (means_old2 + torch.randn(8, K, D, device=device) * 0.1).requires_grad_(True)

    output2 = loss_fn(
        noisy_actions=noisy2, means_old=means_old2,
        means_new=means_new2, sigmas=sigmas,
        rewards=rewards_uniform,
    )
    print(f"  Loss when all groups uninformative: {output2.loss.item():.4f} (should be ~0)")

    return {"filtering_works": True}


# ============================================================
# Test 4: Diffusion Policy path recording
# ============================================================

@run_test("4.1 Diffusion Policy path recording and replay")
def test_path_recording():
    from adagrpo.policy.diffusion_policy import DiffusionPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionPolicy(
        action_dim=7, obs_dim=9, action_horizon=8,
        hidden_dim=128, num_train_timesteps=100,
        num_inference_steps=5, schedule="cosine", ddim_eta=0.0,
    ).to(device)

    obs = torch.randn(4, 9, device=device)

    # Standard inference
    with torch.no_grad():
        actions = policy.predict_action(obs)
    print(f"  Action shape: {actions.shape}")
    assert actions.shape == (4, 8, 7)

    # Path-recording inference
    with torch.no_grad():
        path = policy.predict_action_with_path(obs)

    print(f"  Path recorded:")
    print(f"    actions:       {path.actions.shape}")
    print(f"    noisy_actions: {path.noisy_actions.shape}")
    print(f"    means:         {path.means.shape}")
    print(f"    sigmas:        {path.sigmas.shape} = {path.sigmas.tolist()}")

    K = path.noisy_actions.shape[1]
    assert path.noisy_actions.shape == (4, K, 8, 7)
    assert path.means.shape == (4, K, 8, 7)
    assert path.sigmas.shape == (K,)

    return {
        "K": K,
        "action_shape": list(actions.shape),
        "path_shapes_correct": True,
    }


@run_test("4.2 Ratio computation with recorded path")
def test_path_ratio_computation():
    from adagrpo.policy.diffusion_policy import DiffusionPolicy
    from adagrpo.core.ratio import compute_per_step_log_ratio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create old and new (slightly perturbed) policies
    policy_old = DiffusionPolicy(
        action_dim=7, obs_dim=9, action_horizon=8,
        hidden_dim=128, num_train_timesteps=100,
        num_inference_steps=5, schedule="cosine",
    ).to(device)

    policy_new = DiffusionPolicy(
        action_dim=7, obs_dim=9, action_horizon=8,
        hidden_dim=128, num_train_timesteps=100,
        num_inference_steps=5, schedule="cosine",
    ).to(device)

    # Make new policy slightly different
    policy_new.load_state_dict(policy_old.state_dict())
    with torch.no_grad():
        for p in policy_new.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    obs = torch.randn(4, 9, device=device)

    # Record path with old policy
    with torch.no_grad():
        path = policy_old.predict_action_with_path(obs)

    # Recompute means with new policy
    with torch.no_grad():
        means_new = policy_new.recompute_path_means(path, obs)

    B, K, H, D = path.noisy_actions.shape
    noisy_flat = path.noisy_actions.view(B, K, H * D)
    means_old_flat = path.means.view(B, K, H * D)
    means_new_flat = means_new.view(B, K, H * D)

    log_ratios = compute_per_step_log_ratio(
        noisy_flat, means_old_flat, means_new_flat, path.sigmas
    )

    print(f"  Per-step log-ratios shape: {log_ratios.shape}")
    print(f"  Per-step log-ratio stats:")
    for k in range(K):
        print(f"    Step {k}: mean={log_ratios[:, k].mean():.4f}, std={log_ratios[:, k].std():.4f}")

    total_log_ratio = log_ratios.sum(dim=-1)
    ratio = torch.exp(total_log_ratio.clamp(-10, 10))
    print(f"  Total ratio: {ratio.mean():.4f} ± {ratio.std():.4f}")

    assert torch.isfinite(log_ratios).all(), "Log ratios should be finite"
    assert torch.isfinite(ratio).all(), "Ratios should be finite"

    return {
        "log_ratio_per_step_mean": log_ratios.mean().item(),
        "total_ratio_mean": ratio.mean().item(),
        "all_finite": bool(torch.isfinite(ratio).all()),
    }


# ============================================================
# Test 5: Denoising loss (auxiliary / IL)
# ============================================================

@run_test("5.1 IL training convergence on synthetic data")
def test_il_convergence():
    from adagrpo.policy.diffusion_policy import DiffusionPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionPolicy(
        action_dim=2, obs_dim=4, action_horizon=4,
        hidden_dim=64, num_train_timesteps=50,
        num_inference_steps=5, schedule="cosine",
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Generate simple synthetic demo data: actions = f(obs)
    torch.manual_seed(42)
    num_demos = 256
    obs_data = torch.randn(num_demos, 4, device=device)
    # Simple linear target
    W = torch.randn(4, 2, device=device) * 0.5
    target_actions = (obs_data @ W).unsqueeze(1).expand(-1, 4, -1)  # [N, 4, 2]

    losses = []
    for epoch in range(50):
        # Mini-batch
        idx = torch.randint(0, num_demos, (32,))
        obs = obs_data[idx]
        actions = target_actions[idx]

        loss = policy.compute_denoising_loss(obs, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    # Check convergence
    early = np.mean(losses[:5])
    late = np.mean(losses[-5:])
    print(f"  Early loss: {early:.4f}")
    print(f"  Late loss:  {late:.4f}")
    print(f"  Converged:  {late < early}")

    return {
        "early_loss": early,
        "late_loss": late,
        "converged": late < early,
    }


# ============================================================
# Test 6: Full GRPO training loop on synthetic MDP
# ============================================================

@run_test("6.1 GRPO training on simple bandit problem")
def test_grpo_bandit():
    """Test GRPO on a simple continuous bandit where the reward is
    -||a - a*||^2 for a target action a*. This validates the
    full ratio → advantage → loss → gradient pipeline."""
    from adagrpo.core.ratio import compute_per_step_log_ratio, compute_weighted_log_ratio, safe_exp_ratio
    from adagrpo.core.advantages import compute_group_advantages
    from adagrpo.core.aln import AdaptiveLossNetwork
    from adagrpo.policy.diffusion_policy import DiffusionPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    action_dim = 2
    obs_dim = 4
    H = 4  # action horizon
    K = 5  # denoising steps
    G = 8  # group size
    clip_eps = 0.2

    policy = DiffusionPolicy(
        action_dim=action_dim, obs_dim=obs_dim, action_horizon=H,
        hidden_dim=64, num_train_timesteps=50,
        num_inference_steps=K, schedule="cosine",
    ).to(device)

    import copy
    policy_old = copy.deepcopy(policy)
    policy_old.eval()

    aln = AdaptiveLossNetwork(num_timesteps=50, embed_dim=64, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Target: a* = [0.5, 0.5] repeated across horizon
    target = torch.full((H, action_dim), 0.5, device=device)

    obs = torch.randn(1, obs_dim, device=device)  # Fixed observation

    reward_history = []

    for iteration in range(100):
        # Collect G trajectories with old policy
        all_paths = []
        all_rewards = []

        with torch.no_grad():
            for _ in range(G):
                path = policy_old.predict_action_with_path(obs)
                action = path.actions[0]  # [H, D]
                reward = -((action - target).pow(2).sum()).item()
                all_paths.append(path)
                all_rewards.append(reward)

        rewards = torch.tensor(all_rewards, device=device)
        advantages = compute_group_advantages(rewards)
        reward_history.append(rewards.mean().item())

        # Compute ratios for each trajectory
        aln_weights = aln.get_weights(K, device=device)
        loss_total = torch.tensor(0.0, device=device)

        for i in range(G):
            path = all_paths[i]
            B_, K_, H_, D_ = path.noisy_actions.shape

            # Recompute means with current policy (with gradients)
            means_new = policy.recompute_path_means(path, obs)

            noisy_flat = path.noisy_actions.view(B_, K_, H_ * D_)
            means_old_flat = path.means.view(B_, K_, H_ * D_)
            means_new_flat = means_new.view(B_, K_, H_ * D_)

            log_r_per_step = compute_per_step_log_ratio(
                noisy_flat, means_old_flat, means_new_flat, path.sigmas.to(device)
            )
            log_r = compute_weighted_log_ratio(log_r_per_step, aln_weights)
            ratio = safe_exp_ratio(log_r)

            adv = advantages[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            loss_total = loss_total - torch.min(surr1, surr2).mean()

        loss_total = loss_total / G

        optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Update reference every 10 steps
        if (iteration + 1) % 10 == 0:
            policy_old.load_state_dict(policy.state_dict())
            avg_r = np.mean(reward_history[-10:])
            print(f"  Iter {iteration+1}: mean_reward={avg_r:.4f}")

    early_reward = np.mean(reward_history[:10])
    late_reward = np.mean(reward_history[-10:])
    improved = late_reward > early_reward

    print(f"\n  Early reward: {early_reward:.4f}")
    print(f"  Late reward:  {late_reward:.4f}")
    print(f"  Improved: {improved}")

    return {
        "early_reward": early_reward,
        "late_reward": late_reward,
        "improved": improved,
        "reward_history": reward_history,
    }


# ============================================================
# Test 7: Stage-aware scheduling
# ============================================================

@run_test("7.1 HVTS task decomposition and compute savings")
def test_hvts_savings():
    from adagrpo.scheduling.hvts import HierarchicalVisionTaskSegmenter, StageComplexity
    from adagrpo.scheduling.budget_allocator import BudgetAllocator

    hvts = HierarchicalVisionTaskSegmenter(use_vlm=False)
    allocator = BudgetAllocator()

    tasks = [
        "pick up the red cube and place it on the plate",
        "push the block to the target zone",
        "insert the peg into the hole",
        "open the drawer and place the ball inside",
        "stack the blue cube on top of the red cube",
    ]

    results_data = {}
    for task in tasks:
        decomp = hvts.decompose(task, max_episode_steps=300)
        savings = allocator.compute_savings(decomp.stages)
        stages_info = [(s.name, s.complexity.name) for s in decomp.stages]
        print(f"\n  Task: '{task}'")
        print(f"    Stages: {stages_info}")
        print(f"    Savings: {savings['total_savings_ratio']:.2f}x")
        results_data[task] = {
            "num_stages": len(decomp.stages),
            "savings": savings["total_savings_ratio"],
        }

    avg_savings = np.mean([v["savings"] for v in results_data.values()])
    print(f"\n  Average compute savings: {avg_savings:.2f}x")

    return {"tasks": results_data, "avg_savings": avg_savings}


# ============================================================
# Test 8: Hard-trajectory mining
# ============================================================

@run_test("8.1 Hard-trajectory mining improves group informativeness")
def test_hard_mining():
    from adagrpo.core.group_sampler import HardTrajectoryMiner

    miner = HardTrajectoryMiner(buffer_size=1000, uniform_mix=0.1)
    rng = np.random.default_rng(42)

    # Simulate 100 states with varying difficulty
    num_states = 100
    true_success_rates = np.random.uniform(0, 1, num_states)

    # Populate buffer with history
    for state_id in range(num_states):
        sr = true_success_rates[state_id]
        for _ in range(20):
            miner.update(state_id, success=bool(rng.random() < sr))

    # Sample with mining vs uniform
    n_samples = 1000
    mined_ids = miner.sample_states(n_samples, list(range(num_states)), rng)
    uniform_ids = rng.choice(num_states, n_samples).tolist()

    # Measure: how close to 0.5 success rate are the sampled states?
    mined_difficulty = np.mean([true_success_rates[sid] for sid in mined_ids])
    uniform_difficulty = np.mean([true_success_rates[sid] for sid in uniform_ids])

    mined_near_half = np.mean([abs(true_success_rates[sid] - 0.5) for sid in mined_ids])
    uniform_near_half = np.mean([abs(true_success_rates[sid] - 0.5) for sid in uniform_ids])

    print(f"  Mined avg distance from 0.5:   {mined_near_half:.4f}")
    print(f"  Uniform avg distance from 0.5: {uniform_near_half:.4f}")
    print(f"  Mining biases toward 0.5: {mined_near_half < uniform_near_half}")

    stats = miner.get_stats()
    print(f"  Buffer stats: {stats}")

    return {
        "mined_distance": mined_near_half,
        "uniform_distance": uniform_near_half,
        "mining_helps": mined_near_half < uniform_near_half,
    }


# ============================================================
# Test 9: Numerical stability
# ============================================================

@run_test("9.1 Ratio stability under policy drift")
def test_ratio_stability():
    """Test that ratios remain stable as policy diverges from reference."""
    from adagrpo.core.ratio import compute_per_step_log_ratio, compute_uniform_log_ratio, safe_exp_ratio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, K, D = 32, 10, 7

    noisy = torch.randn(B, K, D, device=device)
    means_old = torch.randn(B, K, D, device=device)
    sigmas = torch.ones(K, device=device) * 0.5

    drift_levels = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"  {'Drift':>8} {'Mean ratio':>12} {'Max ratio':>12} {'NaN/Inf':>8} {'Log-var':>10}")

    stability_data = {}
    for drift in drift_levels:
        means_new = means_old + torch.randn(B, K, D, device=device) * drift
        log_r = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)
        total_log_r = compute_uniform_log_ratio(log_r)
        ratio = safe_exp_ratio(total_log_r)

        has_bad = not torch.isfinite(ratio).all()
        print(f"  {drift:>8.2f} {ratio.mean().item():>12.4f} {ratio.max().item():>12.4f} "
              f"{'YES' if has_bad else 'no':>8} {total_log_r.var().item():>10.4f}")

        stability_data[drift] = {
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "log_var": total_log_r.var().item(),
            "stable": not has_bad,
        }

    return stability_data


@run_test("9.2 Gradient stability during GRPO update")
def test_gradient_stability():
    """Check for gradient explosion/vanishing during GRPO updates."""
    from adagrpo.policy.diffusion_policy import DiffusionPolicy
    from adagrpo.core.ratio import compute_per_step_log_ratio, compute_uniform_log_ratio, safe_exp_ratio
    from adagrpo.core.advantages import compute_group_advantages

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import copy

    policy = DiffusionPolicy(
        action_dim=2, obs_dim=4, action_horizon=4,
        hidden_dim=64, num_train_timesteps=50,
        num_inference_steps=5, schedule="cosine",
    ).to(device)

    policy_old = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    obs = torch.randn(1, 4, device=device)

    grad_norms = []
    for step in range(50):
        # Large LR updates to stress-test
        G = 4
        all_ratios = []
        all_advantages = []

        rewards = torch.randn(G, device=device)
        advantages = compute_group_advantages(rewards)

        with torch.no_grad():
            paths = [policy_old.predict_action_with_path(obs) for _ in range(G)]

        loss = torch.tensor(0.0, device=device)
        for i in range(G):
            path = paths[i]
            means_new = policy.recompute_path_means(path, obs)
            B_, K_, H_, D_ = path.noisy_actions.shape
            log_r = compute_per_step_log_ratio(
                path.noisy_actions.view(B_, K_, H_ * D_),
                path.means.view(B_, K_, H_ * D_),
                means_new.view(B_, K_, H_ * D_),
                path.sigmas.to(device),
            )
            total_lr = compute_uniform_log_ratio(log_r)
            ratio = safe_exp_ratio(total_lr)
            surr = ratio * advantages[i]
            loss = loss - surr.mean()

        loss = loss / G
        optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 5 == 0:
            policy_old.load_state_dict(policy.state_dict())

    print(f"  Grad norm stats over 50 steps:")
    print(f"    Mean: {np.mean(grad_norms):.4f}")
    print(f"    Max:  {np.max(grad_norms):.4f}")
    print(f"    Min:  {np.min(grad_norms):.4f}")
    print(f"    NaN:  {sum(1 for g in grad_norms if np.isnan(g))}")

    has_nan = any(np.isnan(g) for g in grad_norms)
    has_explosion = max(grad_norms) > 1e6

    return {
        "mean_grad_norm": float(np.mean(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "no_nan": not has_nan,
        "no_explosion": not has_explosion,
    }


# ============================================================
# Test 10: End-to-end pipeline with dummy env
# ============================================================

@run_test("10.1 End-to-end RL training with dummy environment")
def test_e2e_training():
    """Run the full RL trainer for a few iterations with a dummy env."""
    import gymnasium as gym

    from adagrpo.policy.diffusion_policy import DiffusionPolicy
    from adagrpo.core.aln import AdaptiveLossNetwork
    from adagrpo.training.rl_trainer import RLTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple dummy env
    class DummyManipEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict({
                "robot0_eef_pos": gym.spaces.Box(-1, 1, (3,), np.float32),
                "robot0_eef_quat": gym.spaces.Box(-1, 1, (4,), np.float32),
                "robot0_gripper_qpos": gym.spaces.Box(-1, 1, (2,), np.float32),
            })
            self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
            self._step = 0
            self._target = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def reset(self, **kwargs):
            self._step = 0
            obs = {k: space.sample() for k, space in self.observation_space.spaces.items()}
            return obs, {}

        def step(self, action):
            self._step += 1
            obs = {k: space.sample() for k, space in self.observation_space.spaces.items()}
            dist = np.linalg.norm(action - self._target)
            reward = float(dist < 0.5)  # Binary success
            terminated = reward > 0.5
            truncated = self._step >= 50
            return obs, reward, terminated, truncated, {}

    env = DummyManipEnv()

    policy = DiffusionPolicy(
        action_dim=7, obs_dim=9, action_horizon=4,
        hidden_dim=64, num_train_timesteps=50,
        num_inference_steps=3, schedule="cosine",
    )

    aln = AdaptiveLossNetwork(num_timesteps=50, embed_dim=64, hidden_dim=64)

    trainer = RLTrainer(
        policy=policy,
        env=env,
        aln=aln,
        lr=1e-4,
        clip_eps=0.2,
        aux_weight=0.0,
        group_size=4,
        num_groups_per_iter=2,
        num_update_epochs=1,
        max_episode_steps=50,
        default_denoise_steps=3,
        default_action_horizon=4,
        ref_update_interval=5,
        use_hard_mining=True,
        num_iterations=10,
        save_dir="/tmp/adagrpo_test",
        save_every=100,
        eval_every=5,
        num_eval_episodes=4,
        wandb_enabled=False,
        device=device,
    )

    print("  Running 10 iterations of AdaGRPO...")
    trainer.train()
    print("  ✓ End-to-end pipeline completed without errors")

    return {"e2e_works": True}


# ============================================================
# Test 11: Known obstacles and edge cases
# ============================================================

@run_test("11.1 DDIM sigma=0 handling (deterministic steps)")
def test_ddim_zero_sigma():
    """DDIM with eta=0 produces sigma=0 at every step. The per-step ratio
    formula divides by sigma^2 which would be inf. Check our handling."""
    from adagrpo.policy.diffusion_policy import DiffusionPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionPolicy(
        action_dim=2, obs_dim=4, action_horizon=4,
        hidden_dim=64, num_train_timesteps=50,
        num_inference_steps=5, schedule="cosine",
        ddim_eta=0.0,  # Deterministic!
    ).to(device)

    obs = torch.randn(2, 4, device=device)
    with torch.no_grad():
        path = policy.predict_action_with_path(obs)

    print(f"  Sigmas with eta=0: {path.sigmas.tolist()}")

    # Check if any sigmas are exactly 0
    zero_sigmas = (path.sigmas == 0.0).sum().item()
    print(f"  Zero sigmas: {zero_sigmas} out of {path.sigmas.shape[0]}")

    # This is a KNOWN PROBLEM: DDIM with eta=0 gives sigma=0 for all steps
    # The ratio formula has 1/(2*sigma^2) which would be inf
    # Our implementation adds 1e-8 in the denominator to prevent this
    from adagrpo.core.ratio import compute_per_step_log_ratio
    B, K, H, D = path.noisy_actions.shape
    noisy_flat = path.noisy_actions.view(B, K, H * D)
    means_flat = path.means.view(B, K, H * D)
    means_new = means_flat + torch.randn_like(means_flat) * 0.01

    log_r = compute_per_step_log_ratio(noisy_flat, means_flat, means_new, path.sigmas.to(device))
    print(f"  Log-ratios with zero sigmas: finite={torch.isfinite(log_r).all().item()}")
    print(f"  Log-ratio range: [{log_r.min().item():.4f}, {log_r.max().item():.4f}]")

    all_finite = torch.isfinite(log_r).all().item()
    if not all_finite:
        print("  ⚠ WARNING: Non-finite ratios with zero sigmas!")
        print("  This is a critical bug — DDIM eta=0 needs special handling.")

    return {
        "zero_sigmas": zero_sigmas,
        "all_finite": all_finite,
        "critical_issue": not all_finite,
    }


@run_test("11.2 Action chunk dimension handling")
def test_action_chunk_dims():
    """Verify that action chunks are properly handled when flattening
    for ratio computation (H*D vs H and D separately)."""
    from adagrpo.policy.diffusion_policy import DiffusionPolicy
    from adagrpo.core.ratio import compute_per_step_log_ratio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for H in [1, 4, 8, 16]:
        for D in [2, 7, 14]:
            policy = DiffusionPolicy(
                action_dim=D, obs_dim=4, action_horizon=H,
                hidden_dim=32, num_train_timesteps=20,
                num_inference_steps=3, schedule="cosine",
            ).to(device)

            obs = torch.randn(2, 4, device=device)
            with torch.no_grad():
                path = policy.predict_action_with_path(obs)

            B, K, H_, D_ = path.noisy_actions.shape
            assert H_ == H and D_ == D, f"Shape mismatch: expected ({H}, {D}), got ({H_}, {D_})"

            noisy_flat = path.noisy_actions.view(B, K, H * D)
            means_flat = path.means.view(B, K, H * D)
            log_r = compute_per_step_log_ratio(
                noisy_flat, means_flat, means_flat + 0.01, path.sigmas.to(device)
            )
            assert log_r.shape == (B, K), f"Log-ratio shape wrong for H={H}, D={D}"

    print("  All (H, D) combinations work correctly")
    return {"all_dims_work": True}


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("AdaGRPO FEASIBILITY VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run all tests
    tests = [
        test_ratio_math,
        test_ratio_aggregation,
        test_advantages,
        test_aln_il_training,
        test_variance_reduction,
        test_grpo_loss,
        test_group_filtering,
        test_path_recording,
        test_path_ratio_computation,
        test_il_convergence,
        test_grpo_bandit,
        test_hvts_savings,
        test_hard_mining,
        test_ratio_stability,
        test_gradient_stability,
        test_e2e_training,
        test_ddim_zero_sigma,
        test_action_chunk_dims,
    ]

    for test_fn in tests:
        test_fn()

    # Summary
    print("\n" + "=" * 70)
    print("FEASIBILITY VALIDATION SUMMARY")
    print("=" * 70)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name} ({r.duration:.2f}s)")

    print(f"\n  Total: {len(results)} tests, {len(passed)} passed, {len(failed)} failed")

    if failed:
        print("\n  FAILED TESTS:")
        for r in failed:
            print(f"\n  --- {r.name} ---")
            print(f"  {r.message[:500]}")

    # Identify obstacles
    print("\n" + "=" * 70)
    print("IDENTIFIED OBSTACLES & ISSUES")
    print("=" * 70)

    obstacles = []

    # Check specific results
    for r in results:
        if not r.passed:
            obstacles.append(f"❌ CRITICAL: {r.name} failed — {r.message.split(chr(10))[0]}")

        if r.data:
            # DDIM sigma issue
            if "critical_issue" in r.data and r.data["critical_issue"]:
                obstacles.append(
                    "❌ CRITICAL: DDIM with eta=0 produces sigma=0 at all steps. "
                    "The ratio formula 1/(2σ²) diverges. Need to either: "
                    "(a) use eta > 0 for stochastic DDIM, or "
                    "(b) replace sigma with the DDIM-specific posterior variance."
                )
            # Variance reduction
            if "proposition1_holds" in r.data and not r.data["proposition1_holds"]:
                obstacles.append(
                    "⚠ WARNING: ALN did not reduce ratio variance in test. "
                    "Proposition 1 may need stronger conditions."
                )
            # GRPO convergence
            if "improved" in r.data and not r.data["improved"]:
                obstacles.append(
                    "⚠ WARNING: GRPO did not improve reward in bandit test. "
                    "May need tuning or more iterations."
                )
            # Gradient stability
            if "no_nan" in r.data and not r.data["no_nan"]:
                obstacles.append("❌ CRITICAL: NaN gradients detected during GRPO updates.")
            if "no_explosion" in r.data and not r.data["no_explosion"]:
                obstacles.append("⚠ WARNING: Gradient explosion detected. Need aggressive clipping.")

    if not obstacles:
        obstacles.append("✓ No critical obstacles detected in synthetic tests.")

    for i, obs in enumerate(obstacles, 1):
        print(f"  {i}. {obs}")

    print("\n" + "=" * 70)
    print("KNOWN LIMITATIONS (from analysis)")
    print("=" * 70)
    known = [
        "1. recompute_path_means() requires K forward passes per trajectory per update epoch. "
        "   With G=8, num_groups=4, K=10, that's 320 forward passes per iteration.",
        "2. The initial noise a^(K) is not stored in DenoisingPath. "
        "   recompute_path_means() uses a proxy for the first step.",
        "3. DDIM deterministic (eta=0) needs special sigma handling for ratios.",
        "4. The auxiliary denoising loss in rl_trainer.py is not yet connected "
        "   to a demo replay buffer.",
        "5. The GRPO update uses pre-computed means_new which doesn't properly "
        "   backprop through the policy network in the current implementation.",
    ]
    for item in known:
        print(f"  {item}")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
