"""模块 15 示例模型：一个能在 CPU 上几秒跑完的小 CNN。

面试常问「你的模型输入输出是什么形状」——答案应该能在这一个文件里看到。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """28x28 单通道图像二分类模型。

    Args:
        in_channels: 输入通道数，PneumoniaMNIST 为 1
        num_classes: 输出类别数，二分类为 2
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: (B, 1, 28, 28) float32，值域 [0, 1]

        Returns:
            (B, 2) logits
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
