"""Unit tests for AdaGRPO loss and advantage computation."""

import torch
import pytest

from adagrpo.core.advantages import (
    compute_group_advantages,
    compute_batched_group_advantages,
    filter_uninformative_groups,
)
from adagrpo.core.grpo import AdaGRPOLoss


class TestGroupAdvantages:
    def test_zero_mean(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = compute_group_advantages(rewards)
        assert abs(adv.mean().item()) < 1e-5

    def test_unit_variance(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        adv = compute_group_advantages(rewards)
        # Should have approximately unit variance
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_ordering_preserved(self):
        rewards = torch.tensor([1.0, 5.0, 3.0])
        adv = compute_group_advantages(rewards)
        # Highest reward → highest advantage
        assert adv[1] > adv[2] > adv[0]


class TestBatchedAdvantages:
    def test_shape(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        adv = compute_batched_group_advantages(rewards, group_size=4)
        assert adv.shape == (8,)

    def test_per_group_normalisation(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        adv = compute_batched_group_advantages(rewards, group_size=4)
        # Each group should be independently normalised
        group1 = adv[:4]
        group2 = adv[4:]
        assert abs(group1.mean().item()) < 1e-5
        assert abs(group2.mean().item()) < 1e-5


class TestFilterUninformativeGroups:
    def test_all_success_filtered(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])  # All success
        mask = filter_uninformative_groups(rewards, group_size=4, success_threshold=0.5)
        assert mask.sum() == 0

    def test_all_fail_filtered(self):
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])  # All fail
        mask = filter_uninformative_groups(rewards, group_size=4, success_threshold=0.5)
        assert mask.sum() == 0

    def test_mixed_kept(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Mixed
        mask = filter_uninformative_groups(rewards, group_size=4, success_threshold=0.5)
        assert mask.sum() == 1

    def test_multiple_groups(self):
        rewards = torch.tensor([
            1.0, 1.0, 1.0, 1.0,  # Group 1: all success → filtered
            0.0, 0.0, 1.0, 1.0,  # Group 2: mixed → kept
            0.0, 0.0, 0.0, 0.0,  # Group 3: all fail → filtered
        ])
        mask = filter_uninformative_groups(rewards, group_size=4, success_threshold=0.5)
        assert mask.shape == (3,)
        assert mask[0] == False
        assert mask[1] == True
        assert mask[2] == False


class TestAdaGRPOLoss:
    @pytest.fixture
    def loss_fn(self):
        return AdaGRPOLoss(
            clip_eps=0.2,
            aux_weight=0.1,
            group_size=4,
            filter_uniform_groups=False,  # Disable for testing
        )

    def test_returns_correct_type(self, loss_fn):
        B, K, D = 8, 5, 7
        output = loss_fn(
            noisy_actions=torch.randn(B, K, D),
            means_old=torch.randn(B, K, D),
            means_new=torch.randn(B, K, D),
            sigmas=torch.ones(K) * 0.5,
            rewards=torch.randn(B),
        )
        assert hasattr(output, "loss")
        assert hasattr(output, "ratio_mean")
        assert hasattr(output, "clipped_frac")

    def test_loss_zero_when_policies_equal(self, loss_fn):
        B, K, D = 8, 5, 7
        noisy = torch.randn(B, K, D)
        means = torch.randn(B, K, D)
        output = loss_fn(
            noisy_actions=noisy,
            means_old=means,
            means_new=means,
            sigmas=torch.ones(K) * 0.5,
            rewards=torch.randn(B),
        )
        # Ratios should all be 1.0
        assert abs(output.ratio_mean - 1.0) < 1e-3

    def test_with_aln_weights(self, loss_fn):
        B, K, D = 8, 5, 7
        aln_weights = torch.sigmoid(torch.randn(K))
        output = loss_fn(
            noisy_actions=torch.randn(B, K, D),
            means_old=torch.randn(B, K, D),
            means_new=torch.randn(B, K, D),
            sigmas=torch.ones(K) * 0.5,
            rewards=torch.randn(B),
            aln_weights=aln_weights,
        )
        assert torch.isfinite(output.loss)

    def test_with_aux_loss(self, loss_fn):
        B, K, D = 8, 5, 7
        aux = torch.tensor(0.5, requires_grad=True)
        output = loss_fn(
            noisy_actions=torch.randn(B, K, D),
            means_old=torch.randn(B, K, D),
            means_new=torch.randn(B, K, D),
            sigmas=torch.ones(K) * 0.5,
            rewards=torch.randn(B),
            aux_loss=aux,
        )
        assert output.loss.item() != output.policy_loss.item()
