"""推理优化实测：no_grad / 半精度 / 批处理 / channels_last / torch.compile。

运行：
    python bench_infer.py
输出一张可直接抄进报告的耗时表。所有数字都是本机实测（RTX 5060 Laptop 8GB）。
"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn
from torchvision.models import resnet18


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup: int = 5, iters: int = 30) -> float:
    """返回单次调用的中位耗时（毫秒）。"""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    model = resnet18(num_classes=2).to(device).eval()

    # ---------- 1. no_grad 有没有用 ----------
    x1 = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        t_nograd = bench(lambda: model(x1))
    t_grad = bench(lambda: model(x1))
    print(f"\n[1] batch=1 fp32   no_grad={t_nograd:.2f} ms   with_grad={t_grad:.2f} ms"
          f"   加速 {t_grad / t_nograd:.2f}x")

    # ---------- 2. 半精度 ----------
    model_h = resnet18(num_classes=2).to(device).half().eval()
    x1_h = x1.half()
    with torch.no_grad():
        t_fp16 = bench(lambda: model_h(x1_h))
    print(f"[2] batch=1 fp16   {t_fp16:.2f} ms   vs fp32 {t_nograd:.2f} ms"
          f"   加速 {t_nograd / t_fp16:.2f}x")

    # autocast 版本（不用改模型权重）
    def autocast_call():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return model(x1)
    t_autocast = bench(autocast_call)
    print(f"    autocast 版      {t_autocast:.2f} ms   加速 {t_nograd / t_autocast:.2f}x")

    # ---------- 3. 批处理吞吐 ----------
    print("\n[3] 批处理（no_grad fp32）")
    for bs in (1, 8, 32, 128):
        x = torch.randn(bs, 3, 224, 224, device=device)
        with torch.no_grad():
            t = bench(lambda: model(x), warmup=3, iters=15)
        print(f"    batch={bs:3d}  {t:7.2f} ms/批  {bs / (t / 1000):8.1f} 张/秒")

    # ---------- 4. channels_last ----------
    model_cl = resnet18(num_classes=2).to(device).to(memory_format=torch.channels_last).eval()
    x_cl = torch.randn(32, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    with torch.no_grad():
        t_cl = bench(lambda: model_cl(x_cl), warmup=3, iters=15)
    x_nc = torch.randn(32, 3, 224, 224, device=device)
    with torch.no_grad():
        t_nc = bench(lambda: model(x_nc), warmup=3, iters=15)
    print(f"\n[4] batch=32 channels_last={t_cl:.2f} ms  contiguous={t_nc:.2f} ms"
          f"   {t_nc / t_cl:.2f}x")

    # ---------- 5. 梯度检查点（训练） ----------
    if device.type == "cuda":
        from torch.utils.checkpoint import checkpoint_sequential

        model_ck = resnet18(num_classes=2).to(device).train()
        model_plain = resnet18(num_classes=2).to(device).train()
        xb = torch.randn(32, 3, 224, 224, device=device)
        yb = torch.randint(0, 2, (32,), device=device)
        crit = nn.CrossEntropyLoss()

        def train_step(m, use_ckpt=False):
            m.zero_grad(set_to_none=True)
            if use_ckpt:
                # 把 layer1-4 切成 4 段做 checkpoint
                out = m.conv1(xb); out = m.bn1(out); out = m.relu(out); out = m.maxpool(out)
                out = checkpoint_sequential(
                    nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4), 4, out
                )
                out = m.avgpool(out)
                out = torch.flatten(out, 1)
                out = m.fc(out)
            else:
                out = m(xb)
            loss = crit(out, yb)
            loss.backward()
            return loss

        def measure_train(use_ckpt: bool):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = model_ck if use_ckpt else model_plain
            for _ in range(2):
                train_step(m, use_ckpt)
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(5):
                train_step(m, use_ckpt)
            sync()
            ms = (time.perf_counter() - t0) / 5 * 1000
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return ms, peak

        t0_ms, mem0 = measure_train(False)
        t1_ms, mem1 = measure_train(True)
        print(f"\n[5] 训练 batch=32  resnet18")
        print(f"    普通        {t0_ms:7.1f} ms/step  峰值显存 {mem0:7.1f} MB")
        print(f"    梯度检查点  {t1_ms:7.1f} ms/step  峰值显存 {mem1:7.1f} MB"
              f"   时间 {t1_ms / t0_ms:.2f}x  显存 {mem0 / mem1:.2f}x")


if __name__ == "__main__":
    main()
