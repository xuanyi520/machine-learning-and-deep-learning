"""指标函数单元测试：每个测试都能手算出期望值。"""
from __future__ import annotations

import pytest
import torch

from metrics import accuracy, dice_score


def test_dice_perfect() -> None:
    """预测和真值完全一样，Dice 必须是 1。"""
    pred = torch.ones(1, 1, 4, 4)
    assert dice_score(pred, pred) == pytest.approx(1.0, abs=1e-4)


def test_dice_no_overlap() -> None:
    """完全不相交，Dice 必须是 0（有 smooth 时接近 0）。"""
    pred = torch.zeros(1, 1, 4, 4)
    pred[..., :2, :] = 1      # 上半 2 行
    target = torch.zeros(1, 1, 4, 4)
    target[..., 2:, :] = 1    # 下半 2 行
    assert dice_score(pred, target) < 1e-3


def test_dice_known_value() -> None:
    """手算：pred 4 个 1，target 4 个 1，重叠 2 个。

    Dice = 2*2 / (4+4) = 0.5
    """
    pred = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).view(1, 1, 2, 4)
    target = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32).view(1, 1, 2, 4)
    assert dice_score(pred, target) == pytest.approx(0.5, abs=1e-4)


def test_dice_shape_mismatch_raises() -> None:
    """形状不一致必须报错，而不是悄悄算出一个错数字。"""
    with pytest.raises(ValueError, match="形状不一致"):
        dice_score(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))


def test_accuracy_hand_computed() -> None:
    """手算：4 个样本，预测对 3 个，准确率 0.75。"""
    logits = torch.tensor([
        [0.9, 0.1],   # 预测 0，真值 0 -> 对
        [0.2, 0.8],   # 预测 1，真值 1 -> 对
        [0.6, 0.4],   # 预测 0，真值 1 -> 错
        [0.1, 0.9],   # 预测 1，真值 1 -> 对
    ])
    labels = torch.tensor([0, 1, 1, 1])
    assert accuracy(logits, labels) == pytest.approx(0.75)
