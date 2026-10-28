# -*- coding: utf-8 -*-
"""练习 3-2-D：给 3-2-A 的 MyNet 加 Dropout(0.3)，对比训练集/测试集准确率"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 数据：8x8 手写数字，64 维 ----------
d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32)
y = torch.tensor(d.target, dtype=torch.long)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)


# ---------- 2. 你的 MyNet，多一个开关 ----------
class MyNet(nn.Module):
    def __init__(self, use_dropout):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)          # 3-2-A 是 784，这里数据是 64 维
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)         # 新增
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)                # 新增：fc2 之后、fc3 之前
        return self.fc3(x)


# ---------- 3. 训练 ----------
@torch.no_grad()
def acc(model, X, y):
    model.eval()                               # 算分时关掉 Dropout
    return (model(X).argmax(dim=1) == y).float().mean().item()


def train(use_dropout, epochs=300, lr=1e-3):
    torch.manual_seed(0)                       # 两组起点相同
    model = MyNet(use_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tr_hist, te_hist = [], []

    for _ in range(epochs):
        model.train()                          # 训练时打开 Dropout
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        tr_hist.append(acc(model, X_tr, y_tr))
        te_hist.append(acc(model, X_te, y_te))
    return tr_hist, te_hist


# ---------- 4. 跑两组 ----------
tr_no, te_no = train(use_dropout=False)
tr_yes, te_yes = train(use_dropout=True)

print("                训练集acc   测试集acc   差距")
for name, tr, te in [("不加 Dropout", tr_no, te_no), ("加 Dropout0.3", tr_yes, te_yes)]:
    print(f"{name:12s}   {tr[-1]:.4f}      {te[-1]:.4f}      {tr[-1] - te[-1]:+.4f}")

# ---------- 5. 画图 ----------
plt.figure(figsize=(8, 5))
plt.plot(tr_no, "--", color="tab:blue", label="不加 Dropout · 训练集")
plt.plot(te_no, "-", color="tab:blue", label="不加 Dropout · 测试集")
plt.plot(tr_yes, "--", color="tab:orange", label="加 Dropout · 训练集")
plt.plot(te_yes, "-", color="tab:orange", label="加 Dropout · 测试集")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Dropout 对过拟合的影响")
plt.ylim(0, 1.02)
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3-2-D_dropout_MyNet.png", dpi=150)
plt.show()
