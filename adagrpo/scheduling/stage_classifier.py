"""Distilled lightweight stage classifier.

After HVTS produces task decompositions offline, we distil the stage labels
into a lightweight ResNet-18-based classifier that can run in real-time
during RL rollouts. The classifier maps an observation image to a stage
index, avoiding the cost of querying the VLM at every step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageClassifier(nn.Module):
    """ResNet-18-based stage classifier distilled from HVTS labels."""

    def __init__(
        self,
        num_stages: int = 4,
        obs_dim: Optional[int] = None,
        image_input: bool = True,
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            num_stages: number of task stages to classify.
            obs_dim: if not using image input, dimension of flat obs vector.
            image_input: if True, use ResNet-18 backbone for image observations.
            pretrained_backbone: use ImageNet-pretrained weights.
        """
        super().__init__()
        self.num_stages = num_stages
        self.image_input = image_input

        if image_input:
            try:
                from torchvision.models import resnet18, ResNet18_Weights

                weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
                backbone = resnet18(weights=weights)
            except ImportError:
                from torchvision.models import resnet18

                backbone = resnet18(pretrained=pretrained_backbone)

            # Replace final FC
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = nn.Linear(feat_dim, num_stages)
        else:
            assert obs_dim is not None
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.head = nn.Linear(128, num_stages)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict stage logits.

        Args:
            obs: [B, C, H, W] image or [B, obs_dim] flat vector.

        Returns:
            logits: [B, num_stages]
        """
        features = self.backbone(obs)
        return self.head(features)

    @torch.no_grad()
    def predict_stage(self, obs: torch.Tensor) -> torch.Tensor:
        """Return predicted stage index.

        Args:
            obs: [B, ...] observation.

        Returns:
            stage_idx: [B] integer stage indices.
        """
        logits = self.forward(obs)
        return logits.argmax(dim=-1)

    @staticmethod
    def distillation_loss(
        student_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for distilling HVTS labels.

        Args:
            student_logits: [B, num_stages] predicted logits.
            teacher_labels: [B] integer stage labels from HVTS.

        Returns:
            Scalar loss.
        """
        return F.cross_entropy(student_logits, teacher_labels)
