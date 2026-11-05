"""指标函数：单元测试要测的就是这种「有确定数值答案」的纯函数。"""
from __future__ import annotations

import torch


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> float:
    """计算二值掩码的 Dice 系数（按 batch 求平均）。

    Args:
        pred: 预测掩码，形状 (B, 1, H, W)，值域 0-1
        target: 真实掩码，形状同 pred
        smooth: 平滑项，防止分母为 0

    Returns:
        batch 内平均 Dice，范围 [0, 1]

    Raises:
        ValueError: 形状不一致或张量为空
    """
    if pred.shape != target.shape:
        raise ValueError(f"形状不一致: {pred.shape} vs {target.shape}")
    if pred.numel() == 0:
        raise ValueError("输入为空张量")

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + smooth) / (union + smooth)
    return float(dice.mean())


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """分类准确率。

    Args:
        logits: (B, C) 未过 softmax 的输出
        labels: (B,) int64 类别下标

    Returns:
        准确率，范围 [0, 1]
    """
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean())
