# -*- coding: utf-8 -*-
"""
环境自检 —— 随时确认 ai 环境是好的。

用法（在 Anaconda Prompt 里）：
    conda activate ai
    python "E:\AI学习\check_env.py"
"""
import sys

line = "=" * 50
print(line)
print("Python  :", sys.version.split()[0])
print("解释器  :", sys.executable)

# 检查是不是在 ai 环境里
if "envs\\ai" not in sys.executable.replace("/", "\\"):
    print("!! 警告：当前不在 ai 环境里，先执行 conda activate ai")
print(line)

mods = ["numpy", "matplotlib", "pandas", "sklearn", "torch", "torchvision"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK    {m:12s} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  缺失  {m}")

# pydicom 是模块 4 才需要的，现在缺了没关系
try:
    import pydicom
    print(f"  OK    {'pydicom':12s} {pydicom.__version__}")
except ImportError:
    print(f"  --    {'pydicom':12s} 还没装（模块4 再说）")

print("-" * 50)

try:
    import torch
    print("CUDA 可用 :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("设备      :", torch.cuda.get_device_name(0))
        p = torch.cuda.get_device_properties(0)
        print(f"显存      : {p.total_memory / 1024**3:.1f} GB")
        print(f"计算能力  : sm_{p.major}{p.minor}")
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU 实测  : OK（2000 万次浮点乘加跑通）")
    else:
        print("!! GPU 没接上，去查显卡驱动或 torch 版本")
except ImportError:
    print("!! torch 没装")

print(line)
print("如果上面全是 OK、CUDA 可用是 True，环境就没问题，可以开始学了。")
