# -*- coding: utf-8 -*-
# 练习 4-2-A：完整跑通，10 轮后记录测试集全部指标

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve, auc)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)                       # 固定种子，结果可复现
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# ---------- 1. 数据 ----------
ROOT = Path(r'E:\AI学习\练习代码\data\medmnist')
ROOT.mkdir(parents=True, exist_ok=True)

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
n_classes = len(info['label'])
label_names = info['label']

# ToTensor 把像素压到 0~1，Normalize(0.5,0.5) 再压到 -1~1
# 训练/验证/测试必须用同一套变换，否则分布不一致
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_set = DataClass(split='train', transform=train_transform, download=True, root=str(ROOT))
val_set   = DataClass(split='val',   transform=eval_transform,  download=True, root=str(ROOT))
test_set  = DataClass(split='test',  transform=eval_transform,  download=True, root=str(ROOT))

print(f"训练 {len(train_set)} / 验证 {len(val_set)} / 测试 {len(test_set)}")
print("标签:", label_names)

# 训练集打乱，验证和测试不打乱（评估结果要稳定可复现）
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)


# ---------- 2. 模型 ----------
class MedCNN(nn.Module):
    def __init__(self, in_ch, n_cls):
        super().__init__()
        # 特征提取：通道 1→32→64，尺寸 28→14→7
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2),
        )
        # 分类：3136 = 64 通道 × 7 × 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
            nn.Dropout(0.3),                       # 训练时随机丢弃 30%，防过拟合
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MedCNN(info['n_channels'], n_classes).to(device)
print("参数量:", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()                    # 多分类交叉熵，内部自带 softmax
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ---------- 3. 一个 epoch 的通用流程 ----------
def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.squeeze().long().to(device)      # (64,1) → (64,)

            if train:
                optimizer.zero_grad()

            out = model(xb)
            loss = criterion(out, yb)

            if train:
                loss.backward()
                optimizer.step()

            # 每个 batch 的 loss 是平均值，乘回 batch 大小才是总和
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, correct / total


# ---------- 4. 训练 10 轮 ----------
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(10):
    tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
    va_loss, va_acc = run_epoch(model, val_loader,   train=False)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)

    print(f"Epoch {epoch+1:2d}  "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val loss={va_loss:.4f} acc={va_acc:.4f}")


# ---------- 5. 测试集：只跑这一次 ----------
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]    # 取"肺炎"的概率
        y_prob.extend(prob.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(yb.squeeze().long().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

acc  = accuracy_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)                  # 召回率 = TP/(TP+FN)
prec = precision_score(y_true, y_pred)               # 精确率 = TP/(TP+FP)
spec = tn / (tn + fp)                                # 特异度 = TN/(TN+FP)
f1   = f1_score(y_true, y_pred)
aucc = roc_auc_score(y_true, y_prob)

print("\n===== 测试集（10 轮，阈值 0.5）=====")
print(f"混淆矩阵:\n{cm}")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"准确率 Accuracy     : {acc:.4f}")
print(f"召回率 Recall       : {rec:.4f}")
print(f"精确率 Precision    : {prec:.4f}")
print(f"特异度 Specificity  : {spec:.4f}")
print(f"F1                  : {f1:.4f}")
print(f"AUC                 : {aucc:.4f}")

# 手算验证，确保每个数都能对上
print("\n===== 手算校验 =====")
print(f"准确率 = (TN+TP)/总数 = ({tn}+{tp})/{len(y_true)} = {(tn+tp)/len(y_true):.4f}")
print(f"召回率 = TP/(TP+FN)   = {tp}/({tp}+{fn}) = {tp/(tp+fn):.4f}")
print(f"精确率 = TP/(TP+FP)   = {tp}/({tp}+{fp}) = {tp/(tp+fp):.4f}")
print(f"特异度 = TN/(TN+FP)   = {tn}/({tn}+{fp}) = {tn/(tn+fp):.4f}")

# ---------- 6. 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['train_loss'], label='训练', marker='o', ms=4)
axes[0].plot(history['val_loss'],   label='验证', marker='s', ms=4)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('损失曲线'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['train_acc'], label='训练', marker='o', ms=4)
axes[1].plot(history['val_acc'],   label='验证', marker='s', ms=4)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('准确率')
axes[1].set_title('准确率曲线'); axes[1].legend(); axes[1].grid(alpha=0.3)

fpr, tpr, _ = roc_curve(y_true, y_prob)
axes[2].plot(fpr, tpr, linewidth=2, label=f'AUC = {aucc:.4f}')
axes[2].plot([0, 1], [0, 1], 'k--', label='随机猜测')
axes[2].set_xlabel('假阳性率 FPR'); axes[2].set_ylabel('真阳性率 TPR')
axes[2].set_title('ROC 曲线'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 混淆矩阵画图
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(['预测 normal', '预测 pneumonia'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['真实 normal', '真实 pneumonia'])
ax.set_title('混淆矩阵')
plt.tight_layout()
plt.show()
