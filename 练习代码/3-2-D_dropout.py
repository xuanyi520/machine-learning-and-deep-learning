# -*- coding: utf-8 -*-
"""
练习 3-2-D：加 Dropout 看过拟合

对照两组（结构完全一样，只差一个 Dropout）：
    A 组：64 -> 128 -> 64 -> 10
    B 组：64 -> 128 -> 64 -> Dropout(0.3) -> 10

三种难度，看训练集和测试集准确率的差距怎么变：
    场景1：数据充足（1257 个）      → 过拟合轻微
    场景2：数据很少（150 个）       → 过拟合中等
    场景3：数据充足但标签有 25% 噪声 → 过拟合严重

数据用 sklearn 自带的 8x8 手写数字（64 维、10 类），不需要联网。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 数据 ----------
digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)   # 像素 0~16 -> 0~1
y = torch.tensor(digits.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 造一份"标签被弄脏"的训练集：随机把 25% 的标签改成别的类
g = torch.Generator().manual_seed(7)
y_train_noisy = y_train.clone()
bad = torch.randperm(len(y_train_noisy), generator=g)[: int(0.25 * len(y_train_noisy))]
y_train_noisy[bad] = torch.randint(0, 10, (len(bad),), generator=g)


# ---------- 2. 两个网络：只差一个 Dropout ----------
class NetNoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),      # 10 类，输出 10 个分数
        )

    def forward(self, x):
        return self.net(x)


class NetDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),        # 只多这一行
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------- 3. 准确率工具 ----------
@torch.no_grad()
def acc_of(model, X, y):
    model.eval()                     # 关掉 Dropout，用完整网络评估
    pred = model(X).argmax(dim=1)    # 10 个分数里最大的下标 = 预测类别
    return (pred == y).float().mean().item()


# ---------- 4. 训练 ----------
def train(model_cls, X_tr, y_tr, epochs, lr=1e-3):
    torch.manual_seed(0)                     # 两组初始权重相同
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hist = {"train": [], "test": []}

    for _ in range(epochs):
        model.train()                        # 打开 Dropout（只在训练时随机丢弃）
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        hist["train"].append(acc_of(model, X_tr, y_tr))
        hist["test"].append(acc_of(model, X_test, y_test))
    return hist


# ---------- 5. 三种场景各跑一遍 ----------
scenarios = [
    ("场景1：数据充足（1257 个）", X_train, y_train, 300),
    ("场景2：数据很少（150 个）", X_train[:150], y_train[:150], 300),
    ("场景3：标签有 25% 噪声", X_train, y_train_noisy, 400),
]

fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 4.3 * len(scenarios)))

for row, (title, X_tr, y_tr, epochs) in enumerate(scenarios):
    hist_a = train(NetNoDropout, X_tr, y_tr, epochs)
    hist_b = train(NetDropout, X_tr, y_tr, epochs)

    print(f"\n===== {title} =====")
    print("                训练集acc   测试集acc   差距(过拟合程度)")
    for name, h in [("不加 Dropout", hist_a), ("加 Dropout0.3", hist_b)]:
        tr, te = h["train"][-1], h["test"][-1]
        print(f"{name:12s}   {tr:.4f}      {te:.4f}      {tr - te:+.4f}")

    ax = axes[row][0]
    ax.plot(hist_a["train"], "--", color="tab:blue", label="不加 Dropout · 训练集")
    ax.plot(hist_a["test"], "-", color="tab:blue", label="不加 Dropout · 测试集")
    ax.plot(hist_b["train"], "--", color="tab:orange", label="加 Dropout · 训练集")
    ax.plot(hist_b["test"], "-", color="tab:orange", label="加 Dropout · 测试集")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[row][1]
    gaps = [hist_a["train"][-1] - hist_a["test"][-1],
            hist_b["train"][-1] - hist_b["test"][-1]]
    bars = ax.bar(["不加 Dropout", "加 Dropout(0.3)"], gaps,
                  color=["tab:blue", "tab:orange"], width=0.5)
    for b, gap in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, gap, f"{gap:+.3f}",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("训练集 acc - 测试集 acc")
    ax.set_title("过拟合差距（越小越好）")
    ax.axhline(0, color="gray", linewidth=1)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("3-2-D_dropout.png", dpi=150)
plt.show()
