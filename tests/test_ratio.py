"""Unit tests for path-conditioned ratio computation."""

import torch
import pytest

from adagrpo.core.ratio import (
    compute_per_step_log_ratio,
    compute_uniform_log_ratio,
    compute_weighted_log_ratio,
    safe_exp_ratio,
)


@pytest.fixture
def batch_data():
    """Generate test data: B=4 trajectories, K=5 denoising steps, D=7 action dim."""
    B, K, D = 4, 5, 7
    torch.manual_seed(42)
    noisy_actions = torch.randn(B, K, D)
    means_old = torch.randn(B, K, D)
    means_new = torch.randn(B, K, D)
    sigmas = torch.ones(K) * 0.5
    return noisy_actions, means_old, means_new, sigmas


class TestPerStepLogRatio:
    def test_shape(self, batch_data):
        noisy, m_old, m_new, sigmas = batch_data
        log_r = compute_per_step_log_ratio(noisy, m_old, m_new, sigmas)
        assert log_r.shape == (4, 5)

    def test_zero_when_policies_equal(self):
        """If θ == θ_old, all ratios should be zero."""
        B, K, D = 3, 4, 6
        noisy = torch.randn(B, K, D)
        means = torch.randn(B, K, D)
        sigmas = torch.ones(K)
        log_r = compute_per_step_log_ratio(noisy, means, means, sigmas)
        assert torch.allclose(log_r, torch.zeros_like(log_r), atol=1e-6)

    def test_sign_correct(self):
        """If new policy is closer to target, log-ratio should be positive."""
        B, K, D = 2, 3, 4
        noisy = torch.zeros(B, K, D)  # Target is at origin
        means_old = torch.ones(B, K, D) * 2.0  # Old policy far away
        means_new = torch.ones(B, K, D) * 0.1  # New policy close
        sigmas = torch.ones(K)
        log_r = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)
        assert (log_r > 0).all()

    def test_batched_sigmas(self, batch_data):
        """Test with per-batch sigmas [B, K]."""
        noisy, m_old, m_new, _ = batch_data
        B, K, D = noisy.shape
        sigmas_batched = torch.ones(B, K) * 0.5
        log_r = compute_per_step_log_ratio(noisy, m_old, m_new, sigmas_batched)
        assert log_r.shape == (B, K)


class TestUniformLogRatio:
    def test_sum_over_steps(self, batch_data):
        noisy, m_old, m_new, sigmas = batch_data
        per_step = compute_per_step_log_ratio(noisy, m_old, m_new, sigmas)
        uniform = compute_uniform_log_ratio(per_step)
        assert uniform.shape == (4,)
        assert torch.allclose(uniform, per_step.sum(dim=-1))


class TestWeightedLogRatio:
    def test_with_uniform_weights(self, batch_data):
        """Uniform weights should give same result as uniform ratio."""
        noisy, m_old, m_new, sigmas = batch_data
        per_step = compute_per_step_log_ratio(noisy, m_old, m_new, sigmas)
        K = per_step.shape[1]
        weights = torch.ones(K)
        weighted = compute_weighted_log_ratio(per_step, weights)
        uniform = compute_uniform_log_ratio(per_step)
        assert torch.allclose(weighted, uniform, atol=1e-6)

    def test_zero_weights_zero_ratio(self, batch_data):
        """Zero weights should give zero log-ratio."""
        noisy, m_old, m_new, sigmas = batch_data
        per_step = compute_per_step_log_ratio(noisy, m_old, m_new, sigmas)
        K = per_step.shape[1]
        weights = torch.zeros(K)
        weighted = compute_weighted_log_ratio(per_step, weights)
        assert torch.allclose(weighted, torch.zeros(4), atol=1e-6)

    def test_variance_reduction(self):
        """Downweighting high-variance steps should reduce overall variance."""
        torch.manual_seed(123)
        B, K, D = 100, 10, 7
        noisy = torch.randn(B, K, D)
        means_old = torch.randn(B, K, D)
        means_new = means_old + torch.randn(B, K, D) * 0.1

        # Make some steps high-variance
        means_new[:, 0:3, :] = means_old[:, 0:3, :] + torch.randn(B, 3, D) * 2.0

        sigmas = torch.ones(K) * 0.5
        per_step = compute_per_step_log_ratio(noisy, means_old, means_new, sigmas)

        # Uniform ratio variance
        uniform_var = compute_uniform_log_ratio(per_step).var().item()

        # Weighted: suppress the high-variance steps
        weights = torch.ones(K)
        weights[0:3] = 0.1
        weighted_var = compute_weighted_log_ratio(per_step, weights).var().item()

        assert weighted_var < uniform_var


class TestSafeExpRatio:
    def test_clamps_extreme_values(self):
        log_ratio = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        ratio = safe_exp_ratio(log_ratio)
        assert torch.isfinite(ratio).all()
        assert ratio[2].item() == pytest.approx(1.0, abs=1e-5)

    def test_identity_near_zero(self):
        log_ratio = torch.tensor([0.0])
        assert safe_exp_ratio(log_ratio).item() == pytest.approx(1.0, abs=1e-6)
