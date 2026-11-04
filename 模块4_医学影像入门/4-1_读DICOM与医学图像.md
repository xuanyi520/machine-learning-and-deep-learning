# 4-1 读 DICOM 与医学图像

**建议用时：6-8 小时**

前面三个模块你处理的都是普通图片（MNIST 的 28×28 灰度图）。**医学影像不是普通图片**，它有三层额外的东西：文件格式、物理单位、显示规则。

这一节的目标只有一个：**你拿到一张真实的 CT，能把它正确读出来、正确显示、正确解释每个数字代表什么。**

这一节是模块 4 的地基。地基歪了，后面做的模型全是错的。

***

## 〇、先把包装上

打开 Anaconda Prompt，激活 ai 环境：

```bash
conda activate ai
pip install pydicom medmnist seaborn
```

**三个都要装，别跳。**

| 包 | 干什么 | 不装的后果 |
|---|---|---|
| `pydicom` | 读 .dcm 文件 | 4-1 的第一节就跑不了 |
| `medmnist` | 医学图像练手数据集（自动下载，不需要 DICOM） | 4-1 第七节和 4-2 的练习没法跑 |
| `seaborn` | 画混淆矩阵热图 | 4-2 评估那节报 ImportError |

装完验一下：

```python
import pydicom, medmnist, seaborn
print(pydicom.__version__, medmnist.__version__, seaborn.__version__)
```

`medmnist` 第一次使用时才下载数据，装包本身很快。如果下载卡住，往后翻到"下载慢怎么办"。

***

## 一、医学图像到底特殊在哪

**先看一个最容易被忽略的事实：一次 CT 检查不是一个文件，是一堆文件。**

```
E:\某病人CT\
    ├── IM0001.dcm      ← 第 1 层
    ├── IM0002.dcm      ← 第 2 层
    ├── IM0003.dcm      ← 第 3 层
    ├── ...
    └── IM0247.dcm      ← 第 247 层
```

**每一个 .dcm 文件 = 一层横断面切片**（就像把面包切成 200 片，每片拍一张照片）。

247 个文件叠起来，才是一个完整的三维肝脏。

新手最容易犯的错觉是"一张 CT 就是一个文件"。不是。**这也是为什么医学 AI 的数据量按 GB 算——一个病人就几百 MB。**

### 和普通图片的三点区别

| | 普通图片 JPG/PNG | 医学影像 DICOM |
|---|---|---|
| 像素值的含义 | 亮度，没有物理意义 | **HU 值**，有物理意义（衡量组织密度） |
| 尺寸单位 | 像素 | **毫米**（靠 PixelSpacing 换算） |
| 一个文件 | 一张完整图 | **一层切片**（几百个文件才是一次检查） |
| 显示 | 直接显示 | 必须**开窗**才有意义 |
| 元数据 | 几乎没有 | **几十上百个标签**（设备、参数、病人信息） |

这五点里，最要命的是**第一点**。

普通图片里一个像素值是 200，你只知道"比较亮"。CT 里一个像素值是 200 HU，你能说出"这是造影剂强化的血管"。**像素值有物理意义，这是医学影像和 CV 的根本区别。**

***

## 二、DICOM 里到底存了什么

DICOM = **图像数据 + 一张巨大的信息表**。

信息表里的字段叫"标签（Tag）"，每个标签有编号、名字、值。比如：

| 标签名 | 中文名 | 编号 | 值举例 |
|---|---|---|---|
| `PatientName` | 患者姓名 | (0010,0010) | 张三 |
| `Modality` | 模态（设备类型） | (0008,0060) | CT |
| `Rows` / `Columns` | 行数 / 列数（图像尺寸） | (0028,0010/0011) | 512 / 512 |
| `PixelSpacing` | 像素间距 | (0028,0030) | [0.7, 0.7] |
| `SliceThickness` | 层厚 | (0018,0050) | 5.0 |
| `RescaleSlope` | HU 换算斜率 | (0028,1053) | 1 |
| `RescaleIntercept` | HU 换算截距 | (0028,1052) | -1024 |
| `WindowCenter` / `WindowWidth` | 窗位 / 窗宽 | (0028,1050/1051) | 60 / 160 |

**两个说明：**

- **值举例里的"张三"是 anonymize（匿名化）之后的示意**。真实 DICOM 里有完整姓名、生日、检查号——属于病人隐私，所以公开数据集（TCIA、LiTS）发布前都会把这一栏抹掉或改成假名。你以后拿临床数据，第一件事就是确认已经脱敏。
- **`WindowCenter` / `WindowWidth` 就是第五节要讲的窗宽窗位**——设备厂家出厂时把"医生最常用的窗"写在了文件里，放射科软件打开就自动套用。

**你以后写论文，Methods 部分必须报告的就是表里这几个：** 扫描设备、层厚、像素间距、管电压。审稿人会问。

### 必须记住的五个标签

| 标签 | 含义 | 为什么重要 |
|---|---|---|
| `PixelSpacing` | 每个像素多少毫米 `[行间距, 列间距]` | 没有它算不出病灶真实大小 |
| `SliceThickness` | 层厚（mm） | 三维重建必需 |
| `RescaleSlope` / `RescaleIntercept` | 换算成 HU 的两个数 | 不换算，看到的就是没意义的整数 |
| `Modality` | CT / MR / US / PT | 不同类型像素含义完全不同 |
| `WindowCenter` / `WindowWidth` | 默认窗宽窗位 | 医生屏幕上默认看到的效果 |

**下面每个都会单独讲。**

***

## 三、第一步：把文件读出来

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 中文显示（Windows 上不加这两行，标题会变方块）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 注意：E:\某病人CT\IM0001.dcm 只是示意路径，你电脑上没有这个文件，
# 直接照抄会 FileNotFoundError。pydicom 自带真实示例文件，先跑通它：
from pydicom.data import get_testdata_file
ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))

# 以后拿到自己的 .dcm 文件时，换成这行（路径必须真实存在）：
# ds = pydicom.dcmread(r"E:\某病人CT\IM0001.dcm")

print("模态:", ds.Modality)
print("图像尺寸:", ds.Rows, "x", ds.Columns)
print("像素间距:", ds.PixelSpacing)
print("层厚:", getattr(ds, 'SliceThickness', '该文件没有'))
print("位深:", ds.BitsStored)
```

### 为什么有的地方要 `getattr`

**因为 DICOM 标签不是每个文件都齐全。**

不同厂商、不同设备、不同扫描协议，缺失的标签不一样。直接写 `ds.SliceThickness`，遇到缺失的文件直接崩：

```
AttributeError: 'FileDataset' object has no attribute 'SliceThickness'
```

**所以固定写法是：**

```python
层厚 = getattr(ds, 'SliceThickness', '无')   # 有就取，没有返回'无'
```

这是医学影像代码的**第一铁律**。你以后写的每个读 DICOM 的函数，都要这样防。

### 读出像素

```python
pixels = ds.pixel_array

print("形状:", pixels.shape)        # (128, 128)；临床文件常见 (512, 512)
print("数据类型:", pixels.dtype)     # uint16
print("取值范围:", pixels.min(), "~", pixels.max())
```

`pixel_array` 是 pydicom 帮你把文件里的二进制**解压、按位深还原**出来的 numpy 数组。

**注意这里只有两个维度 `(128, 128)`——没有通道。** 因为 CT 是单通道灰度图。（自带示例是 128×128；临床文件常见 512×512，维度结构完全一样）

中间加一个通道就是 `(1, 512, 512)`，再加批就是 `(N, 1, 512, 512)`——跟你在 3-4 学的 NCHW 完全一致。

### 直接 imshow 会很难看

```python
plt.imshow(pixels, cmap='gray')
plt.title('直接显示原始像素值')
plt.show()
```

你会看到一张图。**但有两个问题：**

1. imshow 默认把"这张图自己的最小值→黑、最大值→白"：图里最黑的不一定是空气，黑白没有物理意义
2. **这个数字根本不是亮度，是 X 射线的衰减量**，得先换算成 HU 才有物理意义

所以下一步必须换算。

***

## 四、HU 值——CT 的灵魂

### 一句话

**HU（Hounsfield Unit）= 以水为基准，把组织的 X 射线吸收能力放大 1000 倍。**

公式：

```
HU = (μ组织 - μ水) / μ水 × 1000
```

- `μ` 是 X 射线衰减系数（组织挡住射线的能力）
- 除以 μ水 → 变成"相对于水"的倍数，水自己就是 0
- 乘 1000 → 因为原始比值只有 ±0.001 量级，放大 1000 倍才好记好用

### 不用背表，公式能推

**水的 HU 为什么是 0？**

```
HU = (μ水 - μ水) / μ水 × 1000 = 0
```

水自己跟自己比，当然是 0。**这就是"基准"的意思。**

**空气的 HU 为什么是 -1000？**

空气几乎不挡 X 射线，μ空气 ≈ 0：

```
HU = (0 - μ水) / μ水 × 1000 = -1000
```

**骨的 HU 为什么是 +1000？**

骨的衰减大约是水的 2 倍：

```
HU = (2μ水 - μ水) / μ水 × 1000 = +1000
```

**所以 -1000 到 +1000 这个范围不是随便定的，是"几乎没有 → 水 → 两倍水"的物理刻度。** 记住这个推导，表就忘不掉了。

### 常用组织

| 组织 | HU | 记忆方式 |
|---|---|---|
| 空气 | -1000 | 公式推出来的下限 |
| 肺 | -500 ~ -800 | 大部分是空气，但有点组织 |
| 脂肪 | -100 ~ -50 | 比水"透"一点 |
| **水** | **0** | **基准** |
| 软组织 | 20 ~ 60 | 比水略"挡" |
| **肝脏** | **40 ~ 70** | **你的方向，重点记** |
| 增强血管 / 增强肝 | 100 ~ 300 | 打了造影剂，明显变"挡" |
| 骨 | 400 ~ 1000 | 公式推出来的上限 |
| 金属 | > 1000 | 远超上限，会有伪影 |

**肝脏 40~70 这个数字要背下来。** 以后做肝脏分割、肝脏病灶检出，都靠这个窗口划范围。

> 🖼 **这张表已做成图片**：`E:\AI学习\常用组织HU值速查表.png`——上半是九种组织排在同一根数轴上（负值 = 比水透，正值 = 比水挡），下半是带记忆要点的表格。复习时直接看图。

### 怎么算出来

DICOM 文件里存的**不是 HU**，是 0~4095 的整数（省空间）。要换算：

```python
slope = float(getattr(ds, 'RescaleSlope', 1))
intercept = float(getattr(ds, 'RescaleIntercept', 0))

hu = pixels.astype(np.float32) * slope + intercept
```

**为什么先 `astype(np.float32)`？**

`pixels` 是整数类型（uint16）。整数乘 -1024 会溢出或者被截断。**换算前必须转成浮点。**

### 手算一遍

假设某个像素存的是 `1024`，`RescaleSlope = 1`，`RescaleIntercept = -1024`：

```
HU = 1024 × 1 + (-1024) = 0     ← 水
```

再试 `0`：

```
HU = 0 × 1 + (-1024) = -1024   ← 空气（实际一般是 -1000，这里 -1024 是扫描器下限）
```

再试 `1100`：

```
HU = 1100 × 1 + (-1024) = 76   ← 肝脏实质
```

**看出来了吧：换算前那串数字完全没意义，换算后每个数都能对应到组织。**

### 验证一下你读对了没

```python
print("换算前:", pixels.min(), "~", pixels.max())
print("换算后:", hu.min(), "~", hu.max())
```

**正确的结果应该是：最小值是负的（空气/肺的方向），最大值在 +1000 以上（骨）。用自带的 CT_small.dcm 实测是 -896 ~ 1167，对得上。**

如果换算后最小值还是 0，说明你漏了 intercept。这一步是最常见的错误。

***

## 五、窗宽窗位——为什么同一张 CT 能看出完全不同的东西

### 问题的起点

```
CT 的 HU 范围: -1024 到 +3000    ← 约 4000 个刻度
人眼能分辨:     约 256 级灰阶      ← 显示器上限
```

**4000 个刻度挤进 256 级，等于把书缩印到看不清。** 肉眼看这种图，肝和肌肉全是同一片灰。

解决办法：**只看其中一段，把这一段拉满 256 级。** 这就叫"开窗"。

### 两个参数，两个旋钮

| 参数 | 叫什么 | 通俗理解 | 作用 |
|---|---|---|---|
| **窗位 Window Center (C)** | 中心 | **亮度旋钮** | 你想重点看哪种组织，就以它的 HU 为中心 |
| **窗宽 Window Width (W)** | 宽度 | **对比度旋钮** | 这个范围内才有层次，范围外全黑或全白 |

**记忆口诀：窗位定看什么，窗宽定看得清不清。**

- **窗宽越小** → 对比越强，细节越突出，但看不到全貌
- **窗宽越大** → 对比越弱，整体都在，但细节糊在一起

就跟显示器上那两个旋钮一模一样。

### 公式只有三行

```python
def apply_window(img_hu, center, width):
    low  = center - width / 2
    high = center + width / 2
    img = np.clip(img_hu, low, high)          # 超出范围的裁掉
    img = (img - low) / (high - low) * 255    # 线性拉到 0~255
    return img.astype(np.uint8)
```

- `low = C - W/2` → 下限，比它低的全部变黑
- `high = C + W/2` → 上限，比它高的全部变白
- 中间的值线性映射到 0~255

### 手算一遍：肝窗 (C=60, W=160)

```
low  = 60 - 160/2 = 60 - 80 = -20
high = 60 + 160/2 = 60 + 80 = 140
```

| HU 值 | 计算式 | 显示灰度 | 是什么组织 |
|---|---|---|---|
| -100 | 低于下限 | 0（全黑） | 脂肪，被裁掉 |
| -20 | 下限本身 | 0（全黑） | 边界 |
| 0 | (0+20)/160×255 | 32（很暗） | 水 |
| **40** | (40+20)/160×255 | **96（暗灰）** | **肝脏下界** |
| **60** | (60+20)/160×255 | **128（中灰）** | **窗位中心** |
| **70** | (70+20)/160×255 | **143（中灰）** | **肝脏上界** |
| 100 | (100+20)/160×255 | 191（亮） | 增强血管 |
| 140 | 上限本身 | 255（全白） | 边界 |
| 400 | 高于上限 | 255（全白） | 骨，被裁掉 |

**肝脏的 40~70 落在这个区间里，占了 96~143 这一段灰阶，跨度 47 级——有层次，看得出病灶。**

对比一下纵隔窗 (C=40, W=400)：low=-160，high=240。

```
HU=40 → (40+160)/400×255 = 128
HU=70 → (70+160)/400×255 = 147
```

**跨度只有 19 级（128~147），是肝窗 47 级的三分之一。窗宽越大，肝脏占的灰阶越少，越糊。**

记住这个规律：**同样的 HU 差，窗宽越小，屏幕上的灰度差越大。**

### 反例：用肺窗看肝

肺窗 C=-600, W=1500：

```
low  = -600 - 750 = -1350
high = -600 + 750 = 150
```

肝脏 HU=40 映射到：

```
(40 - (-1350)) / (150 - (-1350)) × 255
= 1390 / 1500 × 255
= 236
```

肝脏下界 40 → 236，上界 70 → 241。

**肝脏的整个 HU 范围（40~70）在屏幕上只占 236~241 这 5 级灰阶。**

一片灰，什么都看不出来。**这就是"选错窗，肝脏就糊了"的数学原因。** 不是玄学，是算出来的。

### 放射科实际用的四套窗

```python
windows = {
    '肺窗':   (-600, 1500),   # 看肺结节、气胸
    '纵隔窗': (40,   400),    # 看血管、淋巴结、软组织
    '肝窗':   (60,   160),    # 看肝脏病灶 ← 你的方向
    '骨窗':   (400,  1800),   # 看骨折、骨转移
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (c, w)) in zip(axes, windows.items()):
    ax.imshow(apply_window(hu, c, w), cmap='gray')
    ax.set_title(f'{name}  (窗位{c}, 窗宽{w})')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**跑一遍，你会亲眼看到同一张图变成四张完全不同样的图。** 这是医学影像和普通图像处理最大的区别，也是面试常问的点。

### 对你的意义

做肝脏 AI，预处理时必须选肝窗（或直接不做窗、保留原始 HU 喂给网络）。

**如果预处理阶段用了肺窗，你等于把肝脏的信息压成了 5 级灰阶，再强的网络也学不出东西。** 这个坑很多人踩过。

> 补充：现在主流做法是**不手动开窗**，直接把 HU 值归一化（比如 clip 到 [-200, 300] 再缩放到 0~1）喂网络。开窗是给"人看"的，网络不需要。但你必须懂开窗，否则读不懂文献，也调不动预处理。

***

## 六、PixelSpacing 和层厚——给像素装上刻度尺

### PixelSpacing：每个像素多少毫米

一张 CT 是 512×512 像素。**但"像素"不是真实尺寸**，得知道每个像素代表多少毫米。

```python
print(ds.PixelSpacing)   # CT_small.dcm 实测 ['0.661468', '0.661468']
```

意思是：**每个像素 = 0.7 mm × 0.7 mm**。

这张图的实际大小：

```
512 × 0.7 = 358.4 mm   ← 约 36 厘米，接近一个成年人的腹部宽度
```

你现在跑的 CT_small.dcm 是 128×128、间距 0.661468 mm：`128 × 0.661468 ≈ 84.7 mm`。方法一样，数字不同而已。

### 顺序陷阱：是 [行, 列]，不是 [宽, 高]

`PixelSpacing` 的写法是 `[RowSpacing, ColumnSpacing]`：

- 第 1 个数 = 行间距 = **纵向**（上下方向，对应 height）
- 第 2 个数 = 列间距 = **横向**（左右方向，对应 width）

**如果两个数不一样（各向异性），搞反了尺寸就算错。** 这个坑很隐蔽，因为大部分 CT 两个数是相等的，你看不出来。

### 手算：肝区有多大

**题目**：图上肝区占 100×100 像素，PixelSpacing = [0.7, 0.7]。实际面积？

```
宽 = 100 × 0.7 = 70 mm
高 = 100 × 0.7 = 70 mm
面积 = 70 × 70 = 4900 mm² = 49 cm²
```

**如果不换算，你会说"肝区是 10000 像素"。这句话在临床上没有意义。** 医生问的是"肿瘤多大"，答案是厘米，不是像素。

### SliceThickness：层与层之间隔多远

```python
print(ds.SliceThickness)   # 5.0
```

意思：**这一层和下一层之间隔着 5 mm。**

### 各向异性——一个体素不是正方体

这是最容易被忽略、后果最严重的概念。

```
XY 方向：0.7 mm   ← 层内，很精细
Z  方向：5.0 mm   ← 层间，很粗糙
```

**一个体素是个"扁盒子"，不是立方体：**

```
0.7 × 0.7 × 5.0 mm
```

Z 方向比 XY 方向粗 **7 倍**（5.0 / 0.7 ≈ 7.1）。

### 后果一：三维卷积会变形

如果你直接把这堆层当成立方体做 3D 卷积，网络会把一个方向上 7 倍的差距当成真实形状。**学到的"病灶立体形状"是歪的。**

解决办法：**重采样（resample）**——把 Z 方向插值成和 XY 一样密，或者统一都变成 1mm × 1mm × 1mm 的各向同性体素。

医学影像预处理里，这一步叫 `spacing normalization`，几乎所有论文都会做。

### 后果二：小病灶可能被漏掉

层厚 5mm 意味着：**一个 3mm 的结节，可能在相邻两层之间被完全跳过。**

| 层厚 | 一个 300 层扫描覆盖 | 3mm 结节的检出 |
|---|---|---|
| 5 mm | 1500 mm = 150 cm | 大概率漏 |
| 1 mm | 300 mm = 30 cm | 能看见 |

**这就是为什么现在临床越来越推薄层 CT（1mm 甚至 0.625mm）。** 代价是文件大了 5 倍、辐射剂量要权衡。

**面试会问**："层厚对病灶检出有什么影响？" 答案就是上面这两条：**Z 分辨率不足 → 小病灶漏检 + 三维结构变形。**

***

## 七、没有 DICOM 文件怎么办

你手头大概率没有。四个办法，按推荐顺序：

### 办法 0：pydicom 自带的示例文件（零下载，本讲前六节用的就是它）

pydicom 安装时就自带十几张**真实的** DICOM 示例图，不用下载任何东西：

```python
from pydicom.data import get_testdata_file
ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))   # 一张真实 CT
```

- 优点：零下载、任何电脑能跑，学读文件、学 HU 换算、学开窗都够用
- 缺点：只是一张 128×128 的小图，**练流程可以，练项目不行**

### 办法 1：MedMNIST（推荐，先用这个）

**把医学影像做成和 MNIST 一样的格式**，28×28，下载快，流程完全一样。

优点：5 分钟就能跑通。
缺点：**分辨率太低，只能练流程，不能发论文。**

### 办法 2：真实公开数据集

| 数据集 | 内容 | 适合 |
|---|---|---|
| **LiTS** | 肝脏 + 肝肿瘤分割 CT | **直接对标你的方向** |
| **TCIA** | 各种癌种的影像库 | 找特定病种 |
| **MSD (Medical Segmentation Decathlon)** | 10 个器官分割任务 | 练分割 |
| **LUNA16** | 肺结节检测 CT | 练检测 |

**LiTS 是第二阶段要用的，先记住名字。**

### 办法 3：向导师或医院要

真实数据最值钱，但要走伦理审批（IRB），周期长。**现在不用急，第一阶段用公开数据足够。**

### 数据存哪儿 + 下载慢怎么办

**MedMNIST 默认把数据存在 `C:\Users\你的用户名\.medmnist\`。** 这是 C 盘，要改。

正确的是用 `root=` 参数（**没有 `MEDMNIST_CACHE` 这个环境变量，别被网上教程骗了**）：

```python
from pathlib import Path

ROOT = Path(r'E:\AI学习\练习代码\data\medmnist')
ROOT.mkdir(parents=True, exist_ok=True)      # ← 必须自己建目录，medmnist 不会帮你建

train_set = DataClass(split='train', transform=transform, download=True, root=str(ROOT))
test_set  = DataClass(split='test',  transform=transform, download=True, root=str(ROOT))
```

**两个坑：**

1. **`ROOT.mkdir(...)` 这行不能省。** medmnist 不会自己创建目录，目录不存在直接报：
   ```
   RuntimeError: Failed to setup the default `root` directory.
   Please specify and create the `root` directory manually.
   ```
2. **`root` 要传给每一个 split，三个集都写上。** 漏一个，那个集就下载到 C 盘。

如果下载还是卡住，挂代理再试。`pneumoniamnist` 只有十几 MB，正常百兆宽带几十秒就下完。

***

## 八、跟着做：MedMNIST 完整跑通

**这段代码是可以直接跑的，不用改。** 逐段解释在下面。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(r'E:\AI学习\练习代码\data\medmnist')   # 数据放 E 盘，不占 C 盘
ROOT.mkdir(parents=True, exist_ok=True)            # 目录要自己建

# ---------- 1. 选数据集 ----------
data_flag = 'pneumoniamnist'          # 肺炎胸片
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
n_classes = len(info['label'])

print("任务:", info['task'])
print("类别:", info['label'])
print("通道数:", info['n_channels'])

# ---------- 2. 加载 ----------
transform = transforms.Compose([transforms.ToTensor()])

train_set = DataClass(split='train', transform=transform, download=True, root=str(ROOT))
val_set   = DataClass(split='val',   transform=transform, download=True, root=str(ROOT))
test_set  = DataClass(split='test',  transform=transform, download=True, root=str(ROOT))

print(f"训练 {len(train_set)}  验证 {len(val_set)}  测试 {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

# 看一眼形状
x, y = train_set[0]
print("单张:", x.shape, "标签:", y.shape, "值:", y)

# ---------- 3. 网络 ----------
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

# ---------- 4. 训练准备 ----------
torch.manual_seed(0)                    # 固定随机种子，结果可复现
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

model = SmallCNN(info['n_channels'], n_classes).to(device)
print("参数量:", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- 5. 训练 ----------
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.squeeze().long().to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.squeeze().long().to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    print(f"Epoch {epoch+1}  验证准确率 = {correct/total:.4f}")
```

**本机实跑的输出**（代码里加了 `torch.manual_seed(0)`，你照抄能复现）：

```
样本数: 训练 4708  验证 524  测试 624
参数量: 105346
设备: cuda
Epoch 1  验证准确率 = 0.8836
Epoch 2  验证准确率 = 0.9447
Epoch 3  验证准确率 = 0.9447
Epoch 4  验证准确率 = 0.9523
Epoch 5  验证准确率 = 0.9504
```

**两个值得注意的现象：**

1. **Epoch 1 的 0.8836 已经远高于"全猜肺炎"的基线 0.7424**（训练集里肺炎占 74.2%）。说明这一轮就学到了真东西，不是碰运气。**拿到结果第一件事就是和基线比，别只看绝对数字。**
2. **第 4 轮 0.9523 之后，第 5 轮掉到 0.9504。** 不涨反跌，说明该停了——这就是 4-2 要讲的**早停**。

**`torch.manual_seed(0)` 是什么？** 固定随机种子，让每次运行的初始化、打乱顺序都一样，结果可复现。**写实验报告时必须固定种子**，否则你今天的 94.47% 明天重跑就变 93%。

上面完整代码已经加了这一行（在 `model = SmallCNN(...)` 之前）：

```python
torch.manual_seed(0)                    # 固定随机种子，结果可复现
model = SmallCNN(info['n_channels'], n_classes).to(device)
```

### 三个必须看懂的细节

**细节 1：`yb.squeeze().long()` 在干什么**

MedMNIST 的标签形状是 `(64, 1)`，多了一个维度。`CrossEntropyLoss` 要求标签是 `(64,)` 的一维整数。

```
yb              → tensor([[0],[1],[0],...])   shape (64,1)
yb.squeeze()    → tensor([0,1,0,...])         shape (64,)
.long()         → 转成整数类型（int64）
```

**这不是可有可无的装饰。** 少了 `.squeeze()` 会报"目标形状不匹配"。

**细节 2：`32 * 7 * 7` 怎么来的**

```
输入        (N, 1, 28, 28)
Conv(1→16)  (N, 16, 28, 28)   padding=1，尺寸不变
MaxPool(2)  (N, 16, 14, 14)   减半
Conv(16→32) (N, 32, 14, 14)   尺寸不变，通道变
MaxPool(2)  (N, 32,  7,  7)   再减半
Flatten     32 × 7 × 7 = 1568
```

**换数据集尺寸时，这个数必须重算。** 28×28 换成 64×64，最后就是 32×16×16。

**细节 3：`device` 那行**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**必须写 `"cuda"`，不能拼错。** 你之前写成了 `"cude"`，而你这台机器 `cuda.is_available()` 返回 `True`，所以会走到 `"cude"` 那一支，然后立刻报错：

```
RuntimeError: Expected one of cpu, cuda, ipu, ... device type at start of device string: cude
```

**这个错只有换到有 GPU 的机器才会暴露，所以在 CPU 机器上写错也没事——但你机器有 GPU，写了就崩。**

**怎么确认自己有没有 GPU：**

```python
print(torch.cuda.is_available())          # True 表示有
print(torch.cuda.get_device_name(0))      # NVIDIA GeForce RTX 5060 Laptop GPU
```

### 看看图像长什么样

```python
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    img, lab = train_set[i]
    ax.imshow(img.squeeze(), cmap='gray')   # (1,28,28) → (28,28)
    ax.set_title(info['label'][str(lab.squeeze().item())], fontsize=8)
    ax.axis('off')
plt.suptitle('肺炎胸片样本（28×28）')
plt.tight_layout()
plt.show()
```

**做完这一步你就有了第一个医学影像模型。** 流程和真实项目一字不差，只是分辨率低。

***

## 九、常见报错对照表

| 报错 | 原因 | 怎么修 |
|---|---|---|
| `ModuleNotFoundError: pydicom` | 没装包 | `pip install pydicom` |
| `AttributeError: 'FileDataset' object has no attribute 'SliceThickness'` | 标签缺失 | 用 `getattr(ds, 'SliceThickness', '无')` |
| `AttributeError: 'FileDataset' object has no attribute 'pixel_array'` | 压缩格式需要解码器 | `pip install pylibjpeg pylibjpeg-libjpeg` |
| HU 换算后最小值还是 0 | 漏了 intercept | 检查 `pixels * slope + intercept` |
| `RuntimeError: ... device string: cude` | 拼错了 | 改成 `"cuda"` |
| `RuntimeError: Expected target size` | 标签没 squeeze | `yb.squeeze().long()` |
| `ImportError: seaborn` | 没装 | `pip install seaborn` |
| 中文标题变方块 | 没设字体 | `plt.rcParams['font.sans-serif'] = ['SimHei']` |
| MedMNIST 下载卡住 | 网络 | 挂代理；目录用 `root=` 参数指定（见第七节。没有 `MEDMNIST_CACHE` 这个环境变量） |

***

## 十、练习

### 练习 4-1-A（读文件）

用 pydicom 自带的 CT_small.dcm 读（以后有自己的 DICOM，再换成自己的路径）。

用 pydicom 读出来，打印：模态、尺寸、像素间距、层厚、位深、像素值范围。

**验收**：六个信息都打印出来了，没有报错。

### 练习 4-1-B（HU 换算）

对读到的像素做 HU 换算，打印换算前后的范围。

**验收**：换算后最小值约 -896、最大值约 +1167（CT_small.dcm 实测）。对不上就是漏了 intercept。

### 练习 4-1-C（开窗对比）

同一张 CT，肝窗 / 肺窗 / 骨窗 / 原始各显示一次，2×2 布局。

**然后回答**：为什么肺窗下肝脏几乎看不出层次？**用 HU 数字解释**，别写"因为不合适"。

### 练习 4-1-D（MedMNIST 换数据集）

把 `data_flag` 换成 `bloodmnist`（血细胞，8 类，3 通道），重新训练。

1. 图像是几通道？几类？
2. 准确率多少？
3. 和 MNIST 的 98.66% 比如何？

**注意：医学图像通常更难，准确率低是正常的。** 别以为是自己写错了。

### 练习 4-1-E（计算题，面试会问）

一张 CT：PixelSpacing = [0.7, 0.7]，SliceThickness = 5.0。

1. 图上 100×100 像素的区域，实际面积是多少 mm²？
2. 一个体素的体积是多少 mm³？
3. Z 方向比 XY 方向粗几倍？
4. 层厚 5mm 对 3mm 结节的检出有什么影响？

### 练习 4-1-F（对比实验）

对 MedMNIST 的图，分别试：

```python
transform_A = transforms.ToTensor()                                     # 不做归一化
transform_B = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.5], [0.5])])  # 归一化
```

各训练 3 个 epoch，比较准确率。

**思考**：为什么医学图像通常要归一化？和 HU 换算是不是同一类操作？

***

## 十一、自测题（带答案）

**先自己答，再往下看答案。答不出来的地方就是你的漏洞。**

**Q1. DICOM 和普通 JPG 的三点核心区别是什么？**

<details>
<summary>看答案</summary>

1. **像素值有物理意义**：JPG 是亮度，DICOM 是 HU（组织密度）
2. **尺寸有物理单位**：JPG 是像素，DICOM 靠 PixelSpacing 换算成毫米
3. **一个文件不是一张图**：JPG 一个文件一张完整图，DICOM 一个文件是一层，几百个文件才是一次检查

第 4 点也算：DICOM 自带大量元数据，JPG 几乎没有。

</details>

**Q2. HU 值的定义是什么？水的 HU 是多少？为什么？**

<details>
<summary>看答案</summary>

**HU = (μ组织 - μ水) / μ水 × 1000**，是以水为基准、把 X 射线衰减能力放大 1000 倍的刻度。

**水 = 0**，因为水跟自己比，分子是 0。

顺带：**空气 ≈ -1000**（不挡射线），**骨 ≈ +1000**（约两倍水）。

</details>

**Q3. 窗宽窗位各控制什么？肝窗参数是多少？**

<details>
<summary>看答案</summary>

- **窗位（Center）= 亮度旋钮**，决定"重点看哪种组织"，以它的 HU 为中心
- **窗宽（Width）= 对比度旋钮**，决定"多大范围有层次"，窗宽越小对比越强

**肝窗：窗位 60，窗宽 160。**

对应显示范围 [-20, 140]，覆盖肝脏的 40~70 并留出余量。

</details>

**Q4. 为什么同一张 CT 不同窗看到的东西差别巨大？**

<details>
<summary>看答案</summary>

因为 CT 的 HU 跨度约 -1024 ~ +3000（4000 级），而显示器只有 256 级灰阶。**必须只截取其中一段拉满 256 级。**

窗宽 160 时，160 个 HU 值分到 256 级，每级约 0.6 HU，细节清楚。
窗宽 1500 时，1500 个 HU 值挤进 256 级，每级约 6 HU，肝脏的 40~70 只占 5 级灰阶，全糊在一起。

**不是显示问题，是信息量分配问题。**

</details>

**Q5. PixelSpacing 为什么重要？**

<details>
<summary>看答案</summary>

因为**像素不是真实单位**。没有 PixelSpacing，你只能说出"病灶占 100×100 像素"，这在临床上没有意义。

换算后才知道"病灶是 7cm × 7cm，49 cm²"——这是医生能用的信息。

另外，PixelSpacing 和 SliceThickness 不一致（各向异性）时，三维重建必须先重采样，否则形状是变形的。

</details>

**Q6（加分）. 为什么层厚 5mm 的 CT 容易漏掉小病灶？**

<details>
<summary>看答案</summary>

两个原因：

1. **Z 方向采样不够密**：3mm 的结节可能正好落在两层之间，两层都拍不到它
2. **部分容积效应**：就算拍到了，一个 3mm 结节混在 5mm 的层厚里，HU 值会被周围组织稀释，看起来"不像病灶"

这就是薄层 CT（1mm 或 0.625mm）成为主流的原因。

</details>

***

## 十二、出口标准

全部打勾才能进 4-2：

- [ ] 能用 pydicom 读出一张 DICOM，并用 `getattr` 防缺标签
- [ ] 会做 HU 换算，能推导出 -1000 / 0 / +1000 三个基准值
- [ ] 知道肝脏 HU 是 40~70
- [ ] 会写 `apply_window`，能手算肝窗的显示范围
- [ ] 能解释"为什么肺窗看肝是一片灰"
- [ ] 知道 PixelSpacing 是 `[行, 列]` 不是 `[宽, 高]`
- [ ] 能解释各向异性对三维重建的影响
- [ ] MedMNIST 跑通，验证准确率 > 90%（本机实测 95.24%）

**全打勾了，进 4-2。**
