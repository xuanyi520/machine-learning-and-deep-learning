# 4-1 读 DICOM 与医学图像

**建议用时：6-8 小时**

前面用的都是普通图片（PNG/JPG）。**医学影像不一样**，你得先学会怎么读它。

这节的产出：读出一张真实 CT，理解它的像素值代表什么。

***

## 一、DICOM 是什么

医院里 CT、MRI、超声出来的文件格式，扩展名是 `.dcm`。

它不只是图像，还是**图像 + 元数据**：

- 像素数据（图像本体）
- 患者信息、检查日期
- 扫描参数（层厚、管电压、像素间距）
- 设备信息

**元数据里最关键的几个：**

| 标签 | 含义 | 为什么重要 |
|---|---|---|
| `PixelSpacing` | 每个像素代表的实际物理尺寸（mm） | 没有它，你无法知道病灶真实大小 |
| `SliceThickness` | 层厚 | 重建三维体积必须 |
| `RescaleSlope/Intercept` | 换算 HU 值的参数 | 见下文 |
| `Modality` | CT / MR / US / ... | 不同类型处理方式不同 |
| `WindowCenter/Width` | 窗宽窗位 | 显示的默认对比度设置 |

**这些是你以后写论文时要报告的参数。** 医学 AI 论文的 Methods 部分必须交代数据来源和预处理，比普通 CV 论文严格得多。

***

## 二、安装并读取

```python
# 如果还没装
# pip install pydicom
```

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 这里需要你有一张 .dcm 文件
ds = pydicom.dcmread("你的文件路径.dcm")

print("模态:", ds.Modality)
print("图像尺寸:", ds.Rows, "x", ds.Columns)
print("像素间距:", ds.PixelSpacing)          # [行间距, 列间距]，单位 mm
print("层厚:", getattr(ds, 'SliceThickness', '无'))
print("位深:", ds.BitsStored)

# 拿到像素数组
pixels = ds.pixel_array
print("数组形状:", pixels.shape, " 数据类型:", pixels.dtype)
print("取值范围:", pixels.min(), "~", pixels.max())

plt.imshow(pixels, cmap='gray')
plt.colorbar()
plt.title('原始像素值')
plt.show()
```

**注意 `pixel_array`** 是 pydicom 帮你把压缩的像素数据解出来的结果。

***

## 三、HU 值——CT 的核心概念

CT 的原始像素值不是随便的数，是**Hounsfield Unit（HU）**，衡量组织对 X 射线的吸收程度。

换算公式（`RescaleSlope` 通常是 1，`RescaleIntercept` 通常是 -1024）：

```python
hu = pixels * ds.RescaleSlope + ds.RescaleIntercept
```

**固定参考值（背下来）：**

| 组织 | HU 范围 |
|---|---|
| 空气 | -1000 |
| 肺 | -500 ~ -800 |
| 脂肪 | -100 ~ -50 |
| 水 | 0 |
| 软组织 | 20 ~ 60 |
| 肝脏 | 40 ~ 70 |
| 增强后的血管/肝 | 100 ~ 300 |
| 骨 | 400 ~ 1000 |
| 金属 | > 1000 |

**这些数字有实际用途**：写代码按阈值分割器官时（比如"HU 在 40-70 之间的是肝脏实质"），用的就是这张表。

***

## 四、窗宽窗位（Window/Level）

CT 的 HU 范围是 -1024 到 3000+，但人眼只能分辨约 256 级灰阶。

**所以显示时必须"开窗"**——只显示某个区间，把它映射到 0-255。

```python
def apply_window(img, center, width):
    """把 HU 值映射到显示范围"""
    low = center - width / 2
    high = center + width / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low) * 255
    return img.astype(np.uint8)

# 常见的窗（放射科医生日常用这几套）
windows = {
    '肺窗':     (-600, 1500),    # 看肺
    '纵隔窗':   (40, 400),        # 看血管、软组织
    '肝窗':     (60, 160),        # 看肝脏 ← 你的领域
    '骨窗':     (400, 1800),      # 看骨
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (c, w)) in zip(axes, windows.items()):
    ax.imshow(apply_window(hu, c, w), cmap='gray')
    ax.set_title(f'{name}\n(窗位{c}, 窗宽{w})')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**同一张 CT，用不同窗看到的东西完全不同。** 这是医学影像和普通图像处理的第一个大区别。

**对你的意义**：做肝脏 AI，预处理时要选对窗。选了肺窗，肝脏的细节就全丢了。

***

## 五、没有 DICOM 文件怎么办

你可能手头暂时没有。两个办法：

**办法 1：从公开数据集下载**

- **TCIA**（The Cancer Imaging Archive）：海量公开医学影像
- **LiTS**（肝脏肿瘤分割挑战赛数据）：直接对你的方向
- **MedMNIST**：轻量级，适合练手（下面演示）

**办法 2：用 MedMNIST 先跑通流程**（推荐先用这个）

MedMNIST 把医学影像做成了和 MNIST 一样的格式，非常适合入门。

```python
# pip install medmnist
import medmnist
from medmnist import INFO, Evaluator
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

data_flag = 'pneumoniamnist'      # 肺炎胸片数据集
info = INFO[data_flag]
print("任务:", info['task'])       # 分类
print("类别:", info['label'])
print("图像尺寸:", info['n_channels'], info['n_channels'] and medmnist.INFO[data_flag]['n_channels'])

DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([transforms.ToTensor()])

train_set = DataClass(split='train', transform=transform, download=True)
test_set = DataClass(split='test', transform=transform, download=True)

print(train_set)
x, y = train_set[0]
print("单张图像形状:", x.shape, " 标签:", y)
```

**MedMNIST 有很多医学数据集**，你可以挨个试：`pathmnist`、`bloodmnist`、`organamnist`、`breastmnist`……换 `data_flag` 就行。

***

## 六、跟着做：先跑通 MedMNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
n_classes = len(info['label'])

transform = transforms.Compose([transforms.ToTensor()])
train_set = DataClass(split='train', transform=transform, download=True)
test_set = DataClass(split='test', transform=transform, download=True)

print(f"训练集: {len(train_set)}  测试集: {len(test_set)}")
print(f"类别数: {n_classes}  标签含义: {info['label']}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 复用 3-4 的 CNN 思路
class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_cls):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN(info['n_channels'], n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb.squeeze().long())
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(1)
            correct += (pred == yb.squeeze().long()).sum().item()
            total += yb.size(0)
    print(f"Epoch {epoch+1}  测试准确率={correct/total:.4f}")
```

**这是你第一个医学影像模型。** 虽然是简化的数据集，但流程和真实项目完全一样。

```python
# MedMNIST 自带专业评估器，能算 AUC 等指标
from medmnist import Evaluator
evaluator = Evaluator(data_flag, 'test')
# 用法见 medmnist 官方文档
```

***

## 七、练习

### 练习 4-1-A（读文件）

找一张 DICOM（医院拷一张，或从 TCIA 下载一张），用 pydicom 读出来。

打印：模态、尺寸、像素间距、层厚、HU 范围。

### 练习 4-1-B（HU 换算）

对读到的像素数据，做 HU 换算，打印换算前后的取值范围。

**验证**：空气应该接近 -1000。

### 练习 4-1-C（开窗）

对同一张 CT，用肝窗、肺窗、骨窗各显示一次，画 2×2 的图（加一张原始）。

写几句话描述你看到的差异。

### 练习 4-1-D（MedMNIST 换数据集）

把 `data_flag` 换成 `bloodmnist`，重新训练。

1. 图像是几通道？几类？
2. 准确率多少？
3. 和你 3-4 的 MNIST 结果比如何？（**注意：医学图像通常更难，准确率更低是正常的**）

### 练习 4-1-E（思考题）

你读到一张 CT，像素间距是 `[0.7, 0.7]`，层厚是 `5.0`。

1. 一个在图上占 100×100 像素的肝区，实际有多大（mm²）？
2. **层厚 5mm 意味着什么**？对三维重建和病灶检出有什么影响？

（这题涉及医学常识，答不出来就查资料，别跳过——这类问题面试会问。）

***

## 八、自测题

1. DICOM 和普通图片格式的区别是什么？
2. HU 值是什么？水的 HU 是多少？
3. 窗宽窗位的作用是什么？
4. 为什么同一张 CT 用不同窗看差别巨大？
5. `PixelSpacing` 为什么重要？

***

## 出口标准

- [ ] 能用 pydicom 读出一张 DICOM 并打印关键元数据
- [ ] 会做 HU 换算，知道常见组织的 HU 范围
- [ ] 会用窗宽窗位显示，知道肝窗的参数
- [ ] 在 MedMNIST 上跑通一个医学影像分类任务
- [ ] 知道 PixelSpacing 和层厚的临床意义

做到了进 4-2，最后一个模块。
