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

### 1. 先搞清楚一张图有几个维度

一张灰度 CT 切片，不是一个 28×28 的矩阵，而是：

```
(通道, 高, 宽)
   1    28   28      ← 灰度图
   3    28   28      ← 彩色图（红绿蓝）
```

PyTorch 还会在最前面加一维 batch，所以**进网络的数据永远是四维**：

```
(批次, 通道, 高, 宽)
  8     1     28   28
```

**顺序 `NCHW` 要背下来**。numpy / OpenCV 习惯是 `(批次, 高, 宽, 通道)`，顺序不一样，迁移时必踩坑（模块 4 读 DICOM 时会遇到）。

### 2. "通道"在输入层和在中间层，含义不同

- **输入层**：通道 = 图片本身的颜色通道。灰度 1，彩色 3。
- **中间层**：通道 = **这一层提取出了多少种特征**。

`nn.Conv2d(1, 32, 3)` 不是"把 1 张图变成 32 张一样的图"，而是：

> **用 32 个不同的卷积核，各扫一遍，得到 32 张不同的特征图。**

```
核1  → "竖边在哪里"    → 特征图 1
核2  → "横边在哪里"    → 特征图 2
...
核32 → "某种纹理在哪"  → 特征图 32
```

这 32 张图摞在一起，就是输出的 32 个通道。

**记住这条：输出通道数 = 用了几个卷积核。**

```python
conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
print(conv.weight.shape)                        # (32, 1, 3, 3)
print(conv(torch.randn(8, 1, 28, 28)).shape)    # (8, 32, 28, 28)
```

- `weight` 的第一个 32 = 32 个核
- 输出的 32 = 32 张特征图
- `8` 是 batch，`28, 28` 是空间尺寸（padding=1 保住了）

### 3. 最容易误解的一点：一个核不止 9 个数

输入只有 1 通道时，一个核确实是 3×3 = 9 个数。
**输入有 4 通道时，一个核就是 4 个 3×3 摞起来 = 36 个数。**

```python
conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
print(conv.weight.shape)     # (8, 4, 3, 3)
```

```
(输出通道, 输入通道, 核高, 核宽)
    8         4        3    3
```

**读法**：8 个核，每个核有 4 层（对应 4 个输入通道），每层 3×3。

### 4. 多通道是怎么合成一张图的

一个卷积核的计算分三步：

```
第 1 步：核的第 1 层 和 输入的第 1 通道 卷积  → 得到一个数 a
第 2 步：核的第 2 层 和 输入的第 2 通道 卷积  → 得到一个数 b
        ...
最后：  把 a + b + ... 全部加起来，再加上偏置
```

**关键：所有输入通道的结果是「相加」成一个数。**

所以不管输入多少通道，**每个卷积核永远只输出 1 张图**。

### 5. 手算一遍

输入 2 个通道，各是 3×3：

```
通道 0:          通道 1:
1 2 3            1 1 1
4 5 6            1 1 1
7 8 9            1 1 1
```

核 A 的两层分别是：

```
对通道0:         对通道1:
1 0 0            0 0 0
0 1 0            0 0 0
0 0 1            0 0 0
```

计算：

```
通道 0：1×1 + 5×1 + 9×1 = 15
通道 1：全是 0，结果 0
合计：15 + 0 = 15
```

把核 A 的第二层改成全 1：

```
通道 0：15
通道 1：9 个数各乘 1，合计 9
合计：15 + 9 = 24
```

验证：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[
    [[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],      # 通道 0
    [[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]],      # 通道 1
]])

w_a = torch.tensor([[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                     [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]]])
print(F.conv2d(x, w_a).item())    # 15.0

w_b = torch.tensor([[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                     [[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]]])
print(F.conv2d(x, w_b).item())    # 24.0
```

### 6. 参数量怎么算

```
一个核的参数 = 核高 × 核宽 × 输入通道数
总参数       = 一个核的参数 × 输出通道数 + 输出通道个偏置
```

```python
conv = nn.Conv2d(2, 3, 3)
print(sum(p.numel() for p in conv.parameters()))   # 57
```

```
一个核：3×3 × 2 通道 = 18
3 个核：18 × 3 = 54
偏置：  3 个
合计：  57
```

对比全连接：输入 28×28×2 = 1568 维，输出 3 维，`Linear(1568, 3)` 需要 4707 个参数。**卷积用 57 个就做到了。**

### 7. 通道数怎么定

| 位置 | 通道数 | 依据 |
|---|---|---|
| 第一层输入 | 1 或 3 | 看数据：灰度=1，彩色=3 |
| 中间层 | 16 → 32 → 64 → 128 | 凭经验，逐层变多 |
| 输出层 | 类别数 | 10 类就是 10 |

**唯一硬性规则**：这一层的 `in_channels` 必须等于上一层的 `out_channels`。

```python
nn.Conv2d(1, 32, 3)      # 输入 1
nn.Conv2d(32, 64, 3)     # 输入 32 —— 接得上
nn.Conv2d(16, 64, 3)     # 输入 16 —— 接不上，运行时报错
```

报错长这样：

```
RuntimeError: Given groups=1, weight of size [64, 32, 3, 3],
expected input[8, 16, 28, 28] to have 32 channels, but got 16 channels instead
```

**看到 `have X channels, but got Y channels`，就是这一层接错了。**

### 8. 常见误区

| 误区 | 正确理解 |
|---|---|
| 通道数 = 图片张数 | 通道数 = 特征种类数；张数是 batch |
| 5 个通道就是 5 张图 | 是「每张图都有 5 个通道」 |
| 卷积核只有 3×3 = 9 个数 | 完整形状是 `(out, in, 3, 3)` |
| 输入通道可以随便写 | 必须和上一层输出对上 |

**规律**：越往后的层，通道数越多（特征越丰富），空间尺寸越小（感受野越大）。通道靠卷积加，尺寸靠池化减。

***

## 五、池化

### 1. 先问：卷积为什么不缩小尺寸

`Conv2d(1, 32, 3, padding=1)` 输出还是 28×28。

**卷积只增加通道，不缩小空间。** 堆 10 层，尺寸还是 28×28，计算量一点没少。

而且每层只看 3×3，神经元的"视野"太小，看不到整张图。

**池化就是来解决这两件事的：把图缩小。**

### 2. MaxPool 怎么算

取一个 2×2 的窗口，**只保留里面最大的那个数**，窗口每次跳 2 格。

输入 4×4：

```
1  3 | 2  4
5  6 | 1  2
-----+-----
7  2 | 8  3
4  1 | 9  6
```

四个窗口：

```
左上 max(1,3,5,6) = 6       右上 max(2,4,1,2) = 4
左下 max(7,2,4,1) = 7       右下 max(8,3,9,6) = 9
```

输出 2×2：

```
6  4
7  9
```

```python
p = torch.tensor([[[[1.,3.,2.,4.],
                    [5.,6.,1.,2.],
                    [7.,2.,8.,3.],
                    [4.,1.,9.,6.]]]])
print(nn.MaxPool2d(2)(p))
# tensor([[[[6., 4.],
#           [7., 9.]]]])
```

### 3. 池化只改空间，不改通道

```python
x = torch.randn(8, 32, 28, 28)
print(nn.MaxPool2d(2)(x).shape)    # (8, 32, 14, 14)
```

- `8` 张图 —— 不变
- `32` 个通道 —— **不变**
- `28×28 → 14×14` —— 变小了

**为什么通道不变？** 因为池化是**在每个通道上各自独立进行的**，通道之间互不干扰。

这一点和卷积正好相反，记成一对：

| 操作 | 会不会混合通道 | 会不会改尺寸 |
|---|---|---|
| 卷积 | **会**（所有输入通道相加） | 看 padding |
| 池化 | **不会**（各通道各算各的） | **会**（一定缩小） |

### 4. 尺寸公式

```
输出 = floor((输入 - 窗口) / 步长) + 1
```

`MaxPool2d(2)` 的默认步长 = 窗口大小 = 2，所以：

```
28 → (28-2)/2 + 1 = 14
14 → (14-2)/2 + 1 = 7
 7 → (7-2)/2 + 1 = 3.5 → 3    ← 奇数向下取整，最后一行一列被扔掉
```

**所以网络里的尺寸一般设计成 28、14、7 这种能一直除以 2 的数。**

### 5. 三种池化

| 层 | 怎么算 | 用在哪 |
|---|---|---|
| `MaxPool2d(2)` | 窗口取最大值 | 图像分类，最常用 |
| `AvgPool2d(2)` | 窗口取平均值 | 有些网络用来平滑 |
| `AdaptiveAvgPool2d(1)` | 每个通道整张图取平均 → 1×1 | 分类网络末尾，替代全连接层 |

第三种现在很常用：

```python
x = torch.randn(8, 512, 7, 7)
print(nn.AdaptiveAvgPool2d(1)(x).shape)    # (8, 512, 1, 1)
```

**不管输入是 7×7 还是 13×13，都压成 1×1**，然后 `Flatten` 成 `(8, 512)` 接 `Linear(512, 类别数)`。

好处：输入尺寸变了也不报错，参数还少一大截。ResNet 就是这么做的。

### 6. 池化的三个作用

1. **减少计算量**：14×14 变成 7×7，后面的计算量直接少 4 倍
2. **扩大感受野**：缩过一次之后，再叠 3×3 卷积，每个神经元能"看到"原图更大一块
3. **轻微平移鲁棒**：病灶往右挪 1 个像素，窗口里的最大值通常不变

### 7. 池化没有参数

`MaxPool2d`、`AvgPool2d` 里**没有任何可学的数字**，它们不是参数，只是固定规则。

```python
sum(p.numel() for p in nn.MaxPool2d(2).parameters())   # 0
```

**所以池化不参与训练，`loss.backward()` 和它无关。**

### 8. 医学影像里要小心

做**分类**（这张 CT 是肺炎还是正常）时，池化多几次没关系，反正最后只输出一个类别。

但做**分割**（把每个像素标成病灶/正常）时，池化会丢掉边界信息，缩得太小就再也找不回精确边界。

所以 U-Net 是「下采样（池化）到很小 → 再上采样（放大）回原尺寸」，中间还要把浅层的细节"接"过去。这个结构到模块 5 会看到。

### 9. 现代趋势

池化不是必须的：

- 有的网络用 `Conv2d(..., stride=2)` 直接降采样，让网络自己学怎么缩
- 有的网络完全不用池化，靠 `AdaptiveAvgPool2d` 收尾

但**池化最简单、最好理解、最好手算**，先把 `MaxPool2d(2)` 学透。

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
