# 3-4 卷积网络 CNN

**建议用时：8-10 小时**

CNN 是图像领域的基石。你现在做医学影像，**躲不开它**。

本节结束，MNIST 准确率应该到 99% 以上。

***

## 一、全连接层处理图像的三个问题

上节你把 28×28 展平成 784 再喂给全连接层。这有三个毛病：

**1. 参数爆炸**

一张 512×512 的彩色 CT 图，展平后是 `512×512×3 = 786432` 维。接一个 1000 维的隐藏层：

```
786432 × 1000 = 7860 万个参数   ← 一层而已
```

**2. 丢失空间信息**

展平后，第 1 个像素和第 784 个像素在向量里挨着，但它们在图上可能一个在左上角、一个在右下角。**空间关系被抹掉了。**

**3. 平移不敏感**

把图里的肝区往右挪 10 个像素，全连接层会认为这是完全不同的输入。但**病灶挪个位置，它还是同一个病灶**。

***

## 二、卷积的三个核心思想

### 1. 局部连接

一个卷积核只盯着一小块区域（比如 3×3）。检测边缘、纹理这种低级特征，本来就不需要看全图。

### 2. 权值共享

同一个卷积核在整张图上滑动。这意味着：

- 参数极少（3×3 的核只有 9 个参数）
- **平移不变**：不管病灶在图的哪个位置，同一个核都能检测到

### 3. 层次化特征

堆多层卷积，特征从简单到复杂：

```
第1层：边缘、线条
第2层：纹理、角点
第3层：形状、局部结构
第4层：器官轮廓、病灶
```

**这就是深度学习的"深"的意义。**

***

## 三、卷积怎么算

```python
import torch
import torch.nn as nn

# 一张 1 通道 5x5 的"图"
x = torch.tensor([[[
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [0., 0., 0., 1., 1.],
    [0., 0., 0., 1., 1.],
]]])

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
print(conv.weight.shape)      # (1, 1, 3, 3)

out = conv(x)
print(out.shape)              # (1, 1, 3, 3)
```

**输出尺寸公式：**

```
输出 = (输入 - 卷积核 + 2×padding) / stride + 1
```

`5 - 3 + 0 = 2`，`2/1 + 1 = 3`。对上。

**`padding`** 是在边缘补 0，让输出和输入尺寸一致：

```python
conv_p = nn.Conv2d(1, 1, 3, padding=1)
print(conv_p(x).shape)        # (1, 1, 5, 5)  尺寸不变
```

***

## 四、通道

真实图像处理里，通道数表示"提取了多少种不同的特征"。

```python
# 输入 1 通道，输出 32 个通道（32 个不同的卷积核，提取 32 种特征）
conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
print(conv(torch.randn(8, 1, 28, 28)).shape)    # (8, 32, 28, 28)
```

- `8` = batch size
- `32` = 提取了 32 个特征图
- `28, 28` = 空间尺寸不变（用了 padding=1）

**规律**：越往后的层，通道数越多（特征越丰富），空间尺寸越小（感受野越大）。这是 CNN 的标准设计模式。

***

## 五、池化

下采样，把特征图的尺寸减半：

```python
pool = nn.MaxPool2d(2)         # 2x2 窗口取最大值
x = torch.randn(8, 32, 28, 28)
print(pool(x).shape)            # (8, 32, 14, 14)
```

作用：

- 减少计算量
- 增大感受野（让后面的层"看到"更大的范围）
- 提供一定的平移鲁棒性

现在也有用 stride 卷积替代池化的做法，但 MaxPool 最直观，先学它。

***

## 六、完整的 CNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 (N, 1, 28, 28)
            nn.Conv2d(1, 32, 3, padding=1),    # -> (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (N, 32, 14, 14)

            nn.Conv2d(32, 64, 3, padding=1),    # -> (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (N, 64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # -> (N, 64*7*7 = 3136)
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
print(model)
print("参数量:", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)

    print(f"Epoch {epoch+1}  loss={running_loss/len(train_loader):.4f}  测试准确率={correct/total:.4f}")
```

**3 轮就能到 99%。** 对比上节全连接网络的 97%——CNN 用更懂图像的结构，拿到了更好的结果。

***

## 七、为什么 3136 维

`64 通道 × 7 × 7 = 3136`。

这个数字怎么来的：

- 28×28 输入
- 第一次池化 → 14×14
- 第二次池化 → 7×7
- 通道变成 64

**每次改网络结构，这个数字都要重算。** 算错就会报形状不匹配的错。这是 CNN 最常见的报错，遇到就回来查这里。

***

## 八、跟着做：把卷积核画出来

**这一步能让你真正"看见"卷积在学什么。**

```python
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 取第一层卷积的 32 个核
w = model.features[0].weight.data.cpu()      # (32, 1, 3, 3)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(w[i, 0], cmap='gray')
    ax.axis('off')
plt.suptitle('第一层卷积核（32 个）')
plt.tight_layout()
plt.show()
```

你看到的会是一些检测边缘、方向的模板。

**再看第一层的特征图**：

```python
xb, _ = next(iter(test_loader))
xb = xb.to(device)

with torch.no_grad():
    feats = model.features[0](xb)     # (N, 32, 28, 28)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(feats[0, i].cpu(), cmap='viridis')
    ax.axis('off')
plt.suptitle('第 0 张图经 32 个卷积核后的响应')
plt.tight_layout()
plt.show()
```

不同通道对图的不同部分有响应。**这就是 CNN 的"可解释性"入口**——做医学影像时，这类可视化经常用来验证模型有没有看对地方。

***

## 九、练习

### 练习 3-4-A（跑通）

跑通上面的 CNN。记录：
- 参数量是多少？
- 3 轮后测试准确率是多少？
- 和 3-3 的全连接网络比，参数多了还是少了？（**注意这个反直觉的结果**）

### 练习 3-4-B（尺寸计算）

计算下列每步的输出形状：

1. 输入 `(N, 3, 32, 32)`，`Conv2d(3, 16, 3, padding=1)` 后？
2. 接 `MaxPool2d(2)` 后？
3. 再接 `Conv2d(16, 32, 3, padding=1)` 后？
4. 再接 `MaxPool2d(2)` 后？
5. 展平后是多少维？

### 练习 3-4-C（CIFAR-10）

把数据集换成 CIFAR-10（3 通道彩色图，32×32，10 类），改一下网络的输入通道数和分类头。

```python
train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
```

**CIFAR-10 比 MNIST 难得多。** 准确率可能只有 70% 左右。记录你的结果。

（提示：这个数据集用 `transform.ToTensor()` 就够，但加个标准化会更好：
`transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))`）

### 练习 3-4-D（数据增强）

给 CIFAR-10 训练集加数据增强，看准确率提升多少：

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.ToTensor(),
])
```

**为什么这样能提升性能？** 用"模型见到更多变化→更不容易过拟合"来解释。

（注意：医学影像的增强要小心——**水平翻转对胸片可能不合理**，因为左右肺有解剖差异。这是医学 AI 的特殊性，先记住。）

### 练习 3-4-E（迁移学习，挑战）

用预训练的 ResNet18 做迁移学习：

```python
from torchvision import models
model = models.resnet18(pretrained=True)
# 冻结前面所有层，只替换最后的全连接层
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(512, 10)
```

**这是工业界最常用的做法**：借用别人在大数据上训好的模型，只在你的小数据集上微调最后一层。

在 MNIST 或 CIFAR 上试试。（ResNet 需要 3 通道输入，MNIST 要先把 1 通道复制成 3 通道）

***

## 十、自测题

1. 全连接层处理图像的三个问题是什么？
2. 卷积的"权值共享"带来什么好处？
3. 卷积输出的尺寸怎么算？
4. 池化的作用是什么？
5. 为什么 CNN 的参数量比全连接网络少，效果还更好？

***

## 出口标准

- [ ] CNN 在 MNIST 上准确率 > 98%
- [ ] 能手算卷积和池化后的形状变化
- [ ] 能画出卷积核和特征图，并说出你的观察
- [ ] 理解数据增强的原理，并知道医学影像增强的注意事项
- [ ] 知道迁移学习是什么

***

## 模块 3 结束

你现在已经能：**搭网络、训练、评估、可视化、调参。**

这一套流程，换上医学数据集就是医学 AI。下一模块开始换数据。
