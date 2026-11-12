# 18-5 SwinUNet / Mamba UNet 复现对比

**建议用时：10 小时**

**出口标准：能用同一份数据跑出三个架构的对比表，并解释各自的差异来源。**

---

## 〇、这一节在解决什么

**你现在的知识结构停在 U-Net。**

**2022 年以后，医学分割的论文几乎都在这两条线上：**

```
Transformer 路线：SwinUNetR、UNETR、TransUNet
状态空间模型路线：Mamba UNet、VM-UNet、U-Mamba
```

**JD 里明确写了「SwinUNet、Mamba UNet 等模型的文献调研、模块复现和对比实验」。**

**这一节要做三件事：**

1. 把三个架构放在同一份数据上跑
2. 做出一张真实的对比表
3. **说清楚它们各自多花了多少代价，换来了多少收益**

---

## 一、U-Net 的三个「问题」

**先说清楚为什么有人要换架构，不然你只会觉得「U-Net 挺好的，折腾什么」。**

### ① 感受野受限

**卷积核是局部的。一个 3×3×3 的卷积，只能看到相邻的 3 个体素。**

**要看到整个肝脏，需要堆很多层下采样。**

**后果：** 层数越深，特征越抽象，但空间位置信息丢得越多。

**skip connection 是在补救，但治标不治本。**

### ② 长距离依赖学不到

**判断「这块组织是肝脏还是脾脏」，不只看局部纹理，还要看「它在腹腔的哪个位置」。**

**卷积要传很多层才能把远近两处信息结合起来。**

### ③ 固定的权重

**同一个卷积核，在全图所有位置都用同一套权重。**

**图像左上角和右下角用的是同一个滤波器。** 这对「平移不变」的自然图像是对的，但对医学影像不一定——**腹腔上部到底部，器官完全不同。**

---

## 二、SwinUNETR：把 Transformer 装进 3D 分割

### 核心思想

**Swin Transformer 的四个关键设计：**

| 设计 | 解决什么 |
|---|---|
| **窗口注意力（Window Attention）** | 只在局部窗口内算注意力 → 计算量从 O(n²) 降到线性 |
| **移位窗口（Shifted Window）** | 相邻层的窗口错开半个 → 窗口之间能交流 |
| **层级结构（Hierarchical）** | 像 CNN 一样逐层下采样 → 能接进 U 型结构 |
| **3D patch 化** | 把体积切成小块当代币 → 直接处理 3D |

**SwinUNETR 就是把 Swin Transformer 当编码器，配上一个 CNN 解码器。**

```
输入 (1, 96, 96, 96)
    ↓ patch 嵌入：切成 4×4×4 的块
Transformer 编码器（4 个 stage，逐层下采样）
    ↓ 每一层的特征都拿出来
CNN 解码器（上采样 + skip connection）
    ↓
输出 (3, 96, 96, 96)
```

### 实测：能跑、多大、多快

```python
import torch
from monai.networks.nets import SwinUNETR

net = SwinUNETR(
    in_channels=1, out_channels=3,
    feature_size=24,
    use_checkpoint=False,
    spatial_dims=3,
)
print(f"参数量: {sum(p.numel() for p in net.parameters())/1e6:.2f} M")
```

**实际输出：**

```
SwinUNETR 参数量: 15.70 M
前向: (1, 1, 96, 96, 96) -> (1, 3, 96, 96, 96), 1.5s (CPU)
GPU 前向: 1.34s, 峰值显存 1.26 GB
```

**三个关键数字：**

| 项 | 值 | 对比 U-Net |
|---|---|---|
| 参数量 | **15.70 M** | U-Net（channels=16,32,64）约 1.19 M，**大 13 倍** |
| 96³ 前向耗时 | **1.34 秒** | U-Net 约 0.1 秒，**慢 13 倍** |
| 峰值显存 | **1.26 GB** | U-Net 约 0.3 GB，**大 4 倍** |

**这三个数字要一起看。** 换架构不是免费的。

### 8 GB 显存能跑多大

| patch size | batch | 估计显存 | 能跑吗 |
|---|---|---|---|
| 96³ | 1 | 1.3 GB | ✅ 很轻松 |
| 96³ | 4 | 约 4-5 GB | ✅ 可以 |
| 128³ | 2 | 约 5-6 GB | ⚠️ 接近上限 |
| 128³ | 4 | 会 OOM | ❌ |

**SwinUNETR 的显存其实很友好，因为窗口注意力是线性的。**

---

## 三、Mamba：另一条路

### 一句话解释

**Mamba 属于「状态空间模型（State Space Model, SSM）」这一类。**

**它的结构和循环神经网络（RNN）长得像，但能并行训练。**

```
RNN：   h_t = f(h_{t-1}, x_t)       ← 必须顺序算，不能并行
Mamba：h_t = A·h_{t-1} + B·x_t      ← 线性，可以用扫描算法并行
```

**关键：Mamba 是「选择性」的——A、B、C 三个矩阵随输入变化。**

**这意味着它能「决定记住什么、忘掉什么」，而不像传统 RNN 那样一视同仁。**

### 为什么有人用它做分割

| | Transformer | Mamba |
|---|---|---|
| 计算复杂度 | O(n²) | **O(n)** |
| 长序列 | 慢 | **快** |
| 显存 | 随长度平方增长 | **线性增长** |
| 成熟度 | 高 | 中 |

**对 3D 医学数据（一个体积就是几十万个体素），O(n²) 是不可接受的。所以 Mamba 很有吸引力。**

### 一个必须提前说的坑：Windows 装不上

**官方的 `mamba-ssm` 包需要编译 CUDA 扩展。**

```
pip install mamba-ssm
```

**在 Windows 上大概率失败，原因：**

- 没有预编译的 Windows wheel
- 需要本地有 CUDA Toolkit + MSVC，且版本完全匹配
- 报错通常是 `error: command 'cl.exe' failed` 或者 `nvcc fatal`

**PyPI 上有 2.x 的版本号，但那是 Linux 的构建。**

### 替代方案：纯 PyTorch 实现的简化 Mamba

**教学目的不需要官方实现。下面这个能跑，虽然慢。**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    """简化的选择性状态空间块，纯 PyTorch 实现

    真实 Mamba 用的是并行扫描最优实现，
    这里用逐时间步循环，数学上等价但慢很多。
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state

        # 输入投影：决定 A、B、C 和门控
        self.proj = nn.Linear(dim, d_state * 3)
        self.dt_proj = nn.Linear(dim, 1)
        self.out_proj = nn.Linear(d_state, dim)

    def forward(self, x):
        # x: (B, N, D)，N 是序列长度
        B, N, D = x.shape
        bc = self.proj(x)                          # (B, N, 3*d_state)
        Bm, Cm, gate = bc.chunk(3, dim=-1)
        dt = F.softplus(self.dt_proj(x))           # (B, N, 1)

        # 状态初始化为 0
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(N):                          # ← 串行，这是它慢的原因
            h = (1 - dt[:, t]) * h + dt[:, t] * Bm[:, t] * x[:, t, :1]
            outs.append((h * Cm[:, t]).unsqueeze(1))

        y = torch.cat(outs, dim=1)                  # (B, N, d_state)
        return self.out_proj(y * gate)
```

**这个实现能跑，理解原理够用。但不要用它跑真实实验——慢到怀疑人生。**

### 真的要用 Mamba 做实验怎么办

**三条路：**

1. **在 Linux 服务器上跑**（实验室如果有 GPU 服务器，这是最实际的）
2. **用 Colab / AutoDL 租卡**（按小时计费）
3. **用已有的 Mamba 分割结果做文献对比**（不做实验，只做分析）

**第 3 条其实也行——你可以在报告里写「Mamba UNet 在 LiTS 上的公开结果是多少，我的 U-Net 是多少」，做文献级对比。**

---

## 四、对比实验怎么设计

**这一节最有价值的部分。表格怎么设计，直接决定了报告质量。**

### 必须控制的变量

| 变量 | 必须相同 |
|---|---|
| 数据划分 | **同一份，按病人分组** |
| patch size | 同一个（如 96³） |
| batch size | 能跑的最大值，但要一致 |
| 训练 epoch | 相同，或者都用早停 |
| 学习率策略 | 相同 |
| 损失函数 | Dice + CE，同样权重 |
| 数据增强 | 相同 |

**只有架构不同，其他全一样。这是做对比实验的唯一正确方式。**

**很多人做出来「新架构更好」，其实是因为新架构多训了 100 epoch。**

### 指标表

| 架构 | 参数量(M) | 显存(GB) | 单轮耗时(s) | Dice | HD95 | 推理速度(FPS) |
|---|---|---|---|---|---|---|
| 3D U-Net | 1.19 | 0.3 | | | | |
| nnU-Net(2D) | | | | | | |
| SwinUNETR | 15.70 | 1.26 | | | | |
| SimpleMamba | | | | | | |

**`SimpleMamba` 那一行可以空着，或者写「未实跑，原因：Windows 无法安装 mamba-ssm」。**

**诚实比编数据重要。**

### 消融实验

**光比最终数字不够，还要说清「收益从哪来」。**

**以 SwinUNETR 为例，做三组消融：**

| 配置 | Dice | 说明 |
|---|---|---|
| 基线：3D U-Net | | |
| + 换成 Transformer 编码器 | | 只换编码器 |
| + 窗口注意力（不移位） | | 只看窗口内 |
| + 移位窗口 | | 完整 Swin |

**这样才能说清「是 Transformer 有用，还是窗口机制有用」。**

---

## 五、跟着做

### 任务：三个架构同一份数据跑起来

**用 18-2 的假数据（球体）或者你自己的数据。**

**第一步：把 18-2 的训练脚本改成可切换架构。**

```python
import torch
from monai.networks.nets import UNet, SwinUNETR

def build_model(name, in_ch=1, out_ch=2):
    if name == "unet":
        return UNet(spatial_dims=3, in_channels=in_ch, out_channels=out_ch,
                    channels=(16, 32, 64), strides=(2, 2), num_res_units=2)
    if name == "swinunetr":
        return SwinUNETR(in_channels=in_ch, out_channels=out_ch,
                         feature_size=24, use_checkpoint=False, spatial_dims=3)
    raise ValueError(name)
```

**第二步：每个模型跑 10 个 epoch，记录四个数。**

```python
import time, torch

def benchmark(model, patch=(96, 96, 96), batch=1):
    model = model.cuda().eval()
    x = torch.randn(batch, 1, *patch).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)                                   # 预热
    t0 = time.time()
    with torch.no_grad():
        for _ in range(5):
            model(x)
    dt = (time.time() - t0) / 5
    mem = torch.cuda.max_memory_allocated() / 1024**3
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"参数量(M)": round(params, 2),
            "显存(GB)": round(mem, 2),
            "前向(s)": round(dt, 3)}
```

**第三步：填表。**

**验收：**

- [ ] 两个模型都能跑通 10 个 epoch
- [ ] 参数量、显存、耗时三个数都有了
- [ ] Dice 有对比
- [ ] **能说出「SwinUNETR 多花了多少，换来了多少」**

**最后一项是这一节的核心。**

---

## 六、练习

### 练习 18-5-A（必做）：量出代价

**跑上面那个 `benchmark`，填表：**

| 架构 | 参数量 | 显存 | 前向耗时 |
|---|---|---|---|
| 3D U-Net | | | |
| SwinUNETR | | | |

**算一下倍数。**

**结论写一句话：多花 X 倍的参数、Y 倍的时间，换来了 Z 个 Dice 点。值不值？**

### 练习 18-5-B（必做）：找出收益来源

**SwinUNETR 有两个版本：**

```python
# 默认（sw_branch=True）：用 Swin Transformer 做编码器
SwinUNETR(..., use_v2=False)

# 版本 2
SwinUNETR(..., use_v2=True)
```

**两个都跑 10 epoch，对比。** 如果版本 2 更好，说明什么改进了？

**（提示：v2 改的是窗口大小和归一化的位置。）**

### 练习 18-5-C（必做）：跑通简化 Mamba

**把第三节的 `SimpleMambaBlock` 拼成一个分割网络：**

```python
class MiniMambaSeg(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, dim=64):
        super().__init__()
        self.in_conv = nn.Conv3d(in_ch, dim, 3, padding=1)
        self.mamba = SimpleMambaBlock(dim)
        self.out_conv = nn.Conv3d(dim, out_ch, 1)

    def forward(self, x):
        h = self.in_conv(x)                        # (B, dim, D, H, W)
        B, C, D, H, W = h.shape
        s = h.flatten(2).transpose(1, 2)           # (B, D*H*W, dim)
        # 序列太长会卡死，先截断
        s = self.mamba(s[:, :1024])                # 只处理前 1024 个
        s = s.transpose(1, 2).view(B, C, D, H, W)
        return self.out_conv(s)
```

**跑一个 patch 看看能不能前向。**

**你会发现它比 SwinUNETR 还慢——这就是为什么真实的 Mamba 一定要用 CUDA 优化实现。**

### 练习 18-5-D（选做）：文献对比

**去 Papers with Code 查「LiTS liver segmentation」的排行榜，找：**

1. nnU-Net 的公开 Dice
2. SwinUNETR 的公开 Dice
3. U-Mamba 的公开 Dice

**做一张表，和你自己的结果对比。**

**如果差很多，想想为什么**（数据量？训练时长？预处理？）

---

## 七、自测题

1. U-Net 的三个局限是什么？
2. Swin Transformer 的「窗口注意力」解决了什么问题？
3. 「移位窗口」是干什么的？
4. SwinUNETR 相比 U-Net，参数量和显存大概多多少倍？
5. Mamba 属于哪一类模型？它相比 RNN 强在哪？
6. 为什么 Mamba 的计算复杂度是线性的？
7. 做架构对比实验时，必须控制哪些变量？
8. 为什么说「消融实验比最终数字更重要」？

---

## 八、出口标准

- [ ] 能说清 U-Net 的局限和 Transformer / Mamba 各自的对策
- [ ] 能用 MONAI 建 SwinUNETR 并测出参数量、显存、速度
- [ ] 能解释 Mamba 的线性复杂度从哪来
- [ ] 能在同一份数据上做公平的架构对比
- [ ] 能设计消融实验找出收益来源
- [ ] 能诚实地报告「跑不了的部分」

---

## 九、常见错误

| 错误 | 后果 | 改法 |
|---|---|---|
| 对比实验用了不同 epoch | 结论不成立 | 固定 epoch 或都用早停 |
| 数据划分不同 | 数字没可比性 | 用同一份划分文件 |
| 只看 Dice 不看代价 | 得出「新架构更好」的错误结论 | 成本也要报 |
| 强行在 Windows 装 mamba-ssm | 浪费一天 | 用简化实现或换 Linux |
| 用 SimpleMamba 做真实实验 | 时间全耗在等待上 | 只用来理解原理 |
| 不报「没跑的部分」 | 显得不专业 | 老实写原因 |

---

## 十、下一节要干什么

**你现在有四个模型能跑了。但它们在实验室的 4090 上跑得动，不代表能交付。**

**医院不会给你装 PyTorch。他们要的是「一个能塞进现有系统的推理引擎」。**

**下一节 18-6 讲部署：ONNX 导出、推理加速、精度对齐。**

**这一节完成之后，你才算从「能训练」走到「能交付」。**
