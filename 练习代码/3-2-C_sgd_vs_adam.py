# -*- coding: utf-8 -*-
"""
练习 3-2-C：SGD vs Adam 优化器对比

同一个网络、同一份数据，只换优化器：
    SGD (lr=0.01)   vs   Adam (lr=0.001)
画出两条 loss 曲线，看谁收敛得快。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 解决 matplotlib 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)

# ---------- 1. 数据（只造一次，两个优化器共用）----------
X, y = make_classification(n_samples=2000, n_features=10,
                           n_informative=5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)     # (N,) -> (N,1)


# ---------- 2. 网络：参数相同、初始点相同 ----------
def make_model():
    # 每次用同一个种子建模型，保证 SGD / Adam 从同一起点出发，
    # 这样曲线差异完全来自优化器本身
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),            # BCEWithLogitsLoss 自己带 Sigmoid
    )


criterion = nn.BCEWithLogitsLoss()

# ---------- 3. 训练函数 ----------
def train(opt_cls, lr, steps=500):
    model = make_model()
    optimizer = opt_cls(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


results = {}
configs = [("SGD (lr=0.01)", torch.optim.SGD, 0.01),
           ("Adam (lr=0.001)", torch.optim.Adam, 0.001)]

for name, opt_cls, lr in configs:
    model, losses = train(opt_cls, lr)
    with torch.no_grad():
        acc = ((model(X) > 0).float() == y).float().mean().item()
    results[name] = losses
    print(f"{name:16s} 起始 loss={losses[0]:.4f}  最终 loss={losses[-1]:.4f}  "
          f"准确率={acc:.4f}")

# ---------- 4. 画对比曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 左：线性坐标，看整体趋势
for name, losses in results.items():
    axes[0].plot(losses, label=name, linewidth=1.8)
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].set_title("损失曲线（线性坐标）")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右：对数纵坐标，看最后那段细节
for name, losses in results.items():
    axes[1].plot(losses, label=name, linewidth=1.8)
axes[1].set_yscale("log")
axes[1].set_xlabel("step")
axes[1].set_ylabel("loss (log)")
axes[1].set_title("损失曲线（对数坐标，看细节）")
axes[1].legend()
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("3-2-C_sgd_vs_adam.png", dpi=150)
plt.show()

# ---------- 5. 定量对比：谁先到达某个 loss 水平 ----------
print("\n到达指定 loss 所需步数：")
for threshold in [0.5, 0.3, 0.25]:
    row = []
    for name, losses in results.items():
        hit = next((i for i, v in enumerate(losses) if v < threshold), None)
        row.append(f"{name}={'未达到' if hit is None else hit}步")
    print(f"  loss < {threshold}: " + "  |  ".join(row))
