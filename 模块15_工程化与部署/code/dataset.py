"""数据读取：把 MedMNIST 的 npz 包成 Dataset。

单独放一个文件的原因：训练脚本依赖 Hydra，测试不应该为了 import 数据集
把 Hydra、TensorBoard 全拉进来。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    """MedMNIST npz 数据集的 PyTorch 包装。

    Args:
        images: (N, 28, 28) uint8
        labels: (N, 1) uint8
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32，并归一化到 [0, 1]
        self.images = torch.from_numpy(images).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(labels.ravel()).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[i], self.labels[i]
