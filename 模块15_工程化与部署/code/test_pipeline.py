"""数据与模型测试：测「形状」和「值域」，这两条最容易在重构时被改坏。"""
from __future__ import annotations

import numpy as np
import torch

from dataset import NpzDataset
from model import SmallCNN


def test_dataset_shape_and_range() -> None:
    """预处理后必须是 (N, 1, 28, 28)、float32、值域 [0, 1]。"""
    images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    labels = np.random.randint(0, 2, size=(10, 1), dtype=np.uint8)
    ds = NpzDataset(images, labels)

    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert y.dtype == torch.int64
    assert len(ds) == 10


def test_model_output_shape() -> None:
    """模型前向：输入 (B, 1, 28, 28)，输出 (B, 2)。"""
    model = SmallCNN(in_channels=1, num_classes=2).eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()
