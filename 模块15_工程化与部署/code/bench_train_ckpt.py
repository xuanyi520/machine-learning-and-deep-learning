"""梯度检查点实测：3D CNN（医学影像形状），看显存和时间的真实取舍。"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class ConvBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet3d(nn.Module):
    """4 个 stage 的 3D 卷积网络，模拟医学分割模型的显存结构。"""

    def __init__(self, cin: int = 1, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock3d(cin, width)
        self.stages = nn.Sequential(
            ConvBlock3d(width, width),
            ConvBlock3d(width, width),
            ConvBlock3d(width, width),
            ConvBlock3d(width, width),
        )
        self.head = nn.Conv3d(width, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = TinyUNet3d().to(device).train()
    x = torch.randn(4, 1, 64, 64, 64, device=device)
    y = torch.randint(0, 2, (4, 64, 64, 64), device=device)
    crit = nn.CrossEntropyLoss()

    def step(ckpt: bool):
        model.zero_grad(set_to_none=True)
        h = model.stem(x)
        if ckpt:
            h = checkpoint_sequential(model.stages, 4, h, use_reentrant=False)
        else:
            h = model.stages(h)
        out = model.head(h)
        loss = crit(out, y)
        loss.backward()
        return loss

    def measure(ckpt: bool):
        torch.cuda.empty_cache()
        for _ in range(2):
            step(ckpt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(5):
            step(ckpt)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return ms, peak

    t0, m0 = measure(False)
    t1, m1 = measure(True)
    print(f"输入 {tuple(x.shape)}  模型参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"普通        {t0:7.1f} ms/step   峰值显存 {m0:7.1f} MB")
    print(f"梯度检查点  {t1:7.1f} ms/step   峰值显存 {m1:7.1f} MB")
    print(f"时间 {t1/t0:.2f}x   显存节省 {100*(1-m1/m0):.1f}%")


if __name__ == "__main__":
    main()
