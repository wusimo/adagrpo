"""Unit tests for the Adaptive Loss Network."""

import torch
import pytest

from adagrpo.core.aln import (
    AdaptiveLossNetwork,
    compute_aln_il_loss,
    compute_aln_rl_reward,
)


@pytest.fixture
def aln():
    return AdaptiveLossNetwork(num_timesteps=100, embed_dim=64, hidden_dim=64)


class TestALN:
    def test_forward_shape(self, aln):
        timesteps = torch.arange(10)
        logits = aln(timesteps)
        assert logits.shape == (10,)

    def test_get_weights_shape(self, aln):
        weights = aln.get_weights(K=10)
        assert weights.shape == (10,)

    def test_weights_in_range(self, aln):
        weights = aln.get_weights(K=20)
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_state_conditioned(self):
        aln = AdaptiveLossNetwork(num_timesteps=100, embed_dim=64, hidden_dim=64, state_dim=32)
        timesteps = torch.arange(5)
        state_emb = torch.randn(5, 32)
        logits = aln(timesteps, state_emb)
        assert logits.shape == (5,)

    def test_gradient_flows(self, aln):
        timesteps = torch.arange(10)
        logits = aln(timesteps)
        loss = logits.sum()
        loss.backward()
        for p in aln.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestALNILLoss:
    def test_returns_scalar(self, aln):
        per_step_losses = torch.randn(8, 10).abs()
        timesteps = torch.arange(10)
        loss = compute_aln_il_loss(per_step_losses, aln, timesteps)
        assert loss.dim() == 0

    def test_gradient_flows(self, aln):
        per_step_losses = torch.randn(8, 10).abs()
        timesteps = torch.arange(10)
        loss = compute_aln_il_loss(per_step_losses, aln, timesteps)
        loss.backward()
        for p in aln.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestALNRLReward:
    def test_shape(self):
        B, K = 16, 10
        per_step_losses = torch.randn(B, K).abs()
        per_step_log_ratios = torch.randn(B, K)
        rewards = compute_aln_rl_reward(per_step_losses, per_step_log_ratios, beta=0.1)
        assert rewards.shape == (B, K)

    def test_beta_scaling(self):
        B, K = 16, 10
        losses = torch.ones(B, K)
        log_ratios = torch.ones(B, K) * 5.0

        reward_low = compute_aln_rl_reward(losses, log_ratios, beta=0.0)
        reward_high = compute_aln_rl_reward(losses, log_ratios, beta=1.0)
        # Higher beta → higher reward from ratio magnitude
        assert reward_high.mean() > reward_low.mean()
