# -*- coding: utf-8 -*-
"""
练习 3-2-C：SGD vs Adam 优化器对比（make_moons 版）

同一个网络、同一份数据，只换优化器：
    SGD (lr=0.01)   vs   Adam (lr=0.001)
画出两条 loss 曲线对比。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 造数据 ----------
torch.manual_seed(42)
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)      # numpy -> Tensor，float32
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # (N,) -> (N,1)


# ---------- 2. 网络结构：末层不加 Sigmoid ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # 不加 Sigmoid，BCEWithLogitsLoss 内部会处理
        )

    def forward(self, x):
        return self.net(x)


# ---------- 3. 训练函数 ----------
def train(optimizer_name, lr, epochs=500):
    torch.manual_seed(42)          # 保证两组起点相同
    model = Net()
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []                    # 空本子，记录每一步的 loss
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)          # 大写 X
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # logit > 0 等价于 sigmoid(logit) > 0.5
    with torch.no_grad():
        acc = ((model(X) > 0).float() == y).float().mean().item()
    return losses, acc


# ---------- 4. 跑两组 ----------
loss_sgd, acc_sgd = train("SGD", lr=0.01)
loss_adam, acc_adam = train("Adam", lr=0.001)
print(f"SGD  (lr=0.01) : 最终 loss={loss_sgd[-1]:.4f}  acc={acc_sgd:.4f}")
print(f"Adam (lr=0.001): 最终 loss={loss_adam[-1]:.4f}  acc={acc_adam:.4f}")

# ---------- 5. 画图 ----------
plt.figure(figsize=(8, 5))
plt.plot(loss_sgd, label="SGD  lr=0.01", linewidth=2)
plt.plot(loss_adam, label="Adam lr=0.001", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("SGD vs Adam on make_moons")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3-2-C_sgd_vs_adam_moons.png", dpi=150)
plt.show()
