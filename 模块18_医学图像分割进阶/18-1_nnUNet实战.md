# 18-1 nnU-Net 实战：不设计网络，设计流程

**建议用时：10 小时**

**出口标准：能在自己的数据上跑通 nnU-Net 的完整五步流程，并说清它自动配置了哪些参数。**

---

## 〇、环境说明

**这一节需要装 nnU-Net（本机尚未安装）：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install nnunetv2
```

**我还没在这台机器上装它，所以本节的命令是按官方文档写的，没有实跑。**

**装完请你跑一遍，报错就把完整错误发我。我不给你没验证过还说「已验证」的东西。**

**装完先验证：**

```
nnUNetv2_train --help
```

**能打印帮助信息就说明装好了。**

---

## 一、先说一个反直觉的事实

**nnU-Net 的论文标题是：《nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation》**

关键词是 **self-configuring（自配置）**。

**它最大的贡献不是提出了新网络，而是证明了：把预处理、训练策略、后处理的每一步都「配置对」，比设计新网络更重要。**

### 它干过什么

2018 年 Medical Segmentation Decathlon 比赛，任务涵盖 10 种不同的医学分割任务（脑、肝、脾、前列腺、肺……），模态包括 CT、MRI、PET。

**nnU-Net 用同一个流程、不做任何人工调整，拿了第一。**

而且它参赛的版本**没有用任何新奇的网络结构**——就是标准的 U-Net。

**这件事打击了很多人的研究热情，但也说清了真相：**

> **在医学分割里，90% 的收益来自「预处理做得对」，不是「网络设计得花哨」。**

---

## 二、它到底自动配置了什么

**四个步骤，环环相扣。**

```
① 提取指纹（fingerprint）
   扫描你的数据集：多少例、图像尺寸多大、体素间距是多少、
   强度分布如何、各类别的像素占比多少
        ↓
② 实验规划（plan）
   根据指纹 + 你的显存，算出：
   - patch size 该用多大
   - batch size 该是多少
   - 网络该有几层
   - 要不要用级联
   - 归一化方式选哪种
        ↓
③ 预处理（preprocess）
   按规划结果做：重采样、裁剪、归一化，存成 npy
        ↓
④ 训练 + 后处理
   训练时用增强、深度监督
   后处理时用「非最大连通域抑制」之类的方法
```

### 第 ② 步是精髓

**patch size 怎么算：**

```
从数据中位尺寸开始，逐次缩小：
512×512×200 → 256×256×200 → 256×256×120 → ...
直到整个 patch 能塞进你的显存

同时约束：每个维度的尺寸必须是 2^n 的倍数
（因为要下采样再上采样回来）
```

**为什么这一步重要：** 你自己调 patch size，可能试十几次才找到一个不失真又能跑的组合。**它一次算对。**

### 第 ③ 步的关键预处理

| 操作 | 为什么 |
|---|---|
| **裁剪到非零区域** | CT 图像大片是空气，裁掉能省一半显存 |
| **重采样到目标间距** | 原始层厚 5mm、层内 0.7mm，各向异性严重，必须统一 |
| **CT 用前景强度裁剪 + Z-score** | 只按器官区域的强度分布做归一化，排除骨头空气的干扰 |
| **MRI 用 Z-score** | 同上，但不用裁剪 |

**这四条你手动做，每一条都要调参数。它自动做。**

---

## 三、安装（Windows 上有坑）

### 基本安装

**⚠️ 装之前必须先做一件事：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip freeze > F:\torch环境备份.txt
```

**然后看一眼 torch 的版本，记下来。**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**应该是 `2.9.1+cu128 True`。**

### 为什么要先备份（这是实测踩过的坑）

**`nnunetv2` 的依赖里写死了一行：**

```
torch!=2.9.*,>=2.1.2
```

**它明确排除了 torch 2.9。**

**后果（2026-11-10 实测）：**

```
pip install nnunetv2
    ↓
pip 发现 torch 2.9.1 不满足要求
    ↓
去 PyPI 找「合规」的最高版本 → 2.15.0
    ↓
PyPI 的 torch 默认是 CPU 版
    ↓
你的 GPU 环境被覆盖，CUDA 不可用
    ↓
而且整个过程没有任何提示
```

**修复要重新下 2.66 GB 的 cu128 wheel，花了将近半小时。**

### 安全的安装方式

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install nnunetv2 --no-deps
```

**`--no-deps` 只装 nnunetv2 本体，不让它碰其他包。**

**然后补装它真正缺的依赖（这些不涉及 torch）：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install ^
  acvl-utils batchgenerators batchgeneratorsv2 blosc2 ^
  dynamic-network-architectures einops graphviz imagecodecs ^
  scikit-image tifffile yacs
```

### 装完立刻检查

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**如果变成 `2.15.0+cpu False`，说明环境被破坏了，要立刻按下面的方法修复。**

### 修复方法（如果已经被换掉）

**不要用 pip 直接装，它会再从 PyPI 拉 CPU 版。** 用 curl 下 wheel 再本地装：

```
curl -L -C - -o "F:\torch-2.9.1+cu128-cp311-cp311-win_amd64.whl" ^
  "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-win_amd64.whl"

curl -L -o "F:\torchvision-0.24.1+cu128-cp311-cp311-win_amd64.whl" ^
  "https://download.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp311-cp311-win_amd64.whl"
```

**然后本地安装，`--no-index` 保证不联网：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install --no-deps --no-index ^
  "F:\torch-2.9.1+cu128-cp311-cp311-win_amd64.whl" ^
  "F:\torchvision-0.24.1+cu128-cp311-cp311-win_amd64.whl"
```

**实测：curl 下载 4.86 MB/s，pip 自己下只有 1.87 MB/s，中途还会掉到 43 KB/s。**

### 一个好消息

**装完之后 `pip check` 会报：**

```
nnunetv2 2.8.1 has requirement torch!=2.9.*,>=2.1.2, but you have torch 2.9.1+cu128.
```

**这条可以忽略。实测 nnunetv2 2.8.1 在 torch 2.9.1 上能正常导入和运行。**

**那个约束是保守写法，不是真的不兼容。**

### 三个必须设的环境变量

**nnU-Net 靠这三个环境变量找文件，不设就跑不起来。**

| 变量 | 放什么 |
|---|---|
| `nnUNet_raw` | 原始数据（NIfTI + dataset.json） |
| `nnUNet_preprocessed` | 预处理后的 npy（这一步很占空间） |
| `nnUNet_results` | 训练好的模型权重 |

**Windows 命令行临时设置：**

```
set nnUNet_raw=F:\AI学习环境\nnunet\raw
set nnUNet_preprocessed=F:\AI学习环境\nnunet\preprocessed
set nnUNet_results=F:\AI学习环境\nnunet\results
```

**永久设置（推荐）：**

```
setx nnUNet_raw "F:\AI学习环境\nnunet\raw"
setx nnUNet_preprocessed "F:\AI学习环境\nnunet\preprocessed"
setx nnUNet_results "F:\AI学习环境\nnunet\results"
```

**`setx` 设完要重开终端才生效。**

### 三个 Windows 特有的坑

**坑一：路径不能有中文和空格**

```
F:\AI学习环境\nnunet\raw      ← 有中文，可能出问题
F:\nnunet\raw                 ← 安全
```

**nnU-Net 底层用了 `batchgenerators`，它对非 ASCII 路径处理不好。** 报错通常是 `FileNotFoundError` 或者乱码。

**建议就用纯英文路径。**

**坑二：长路径限制**

Windows 默认路径上限 260 字符。nnU-Net 会生成很深的嵌套目录：

```
nnUNet_results\Dataset002_Liver\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\validation\...
```

**很容易超。** 解决办法有两个：

1. 环境变量目录设短一点（`F:\nn\...`）
2. 开启 Windows 长路径支持：
   ```
   组策略 → 计算机配置 → 管理模板 → 系统 → 文件系统
   → 启用 Win32 长路径
   ```

**坑三：符号链接失败**

预处理的某些环节会创建软链接，Windows 普通用户权限下会失败。

**通常不影响主流程，看到 `OSError: symbolic link privilege not held` 可以忽略，如果中断了再处理。**

---

## 四、数据格式：DatasetXXX_NAME

### 目录结构

```
nnUNet_raw/
└── Dataset002_Liver/
    ├── imagesTr/           ← 训练图像
    │   ├── liver_001_0000.nii.gz
    │   ├── liver_002_0000.nii.gz
    │   └── ...
    ├── labelsTr/           ← 训练标签
    │   ├── liver_001.nii.gz
    │   └── ...
    ├── imagesTs/           ← 测试图像（可选）
    └── dataset.json        ← 元信息
```

### 三个命名规则

**① 数据集编号必须是三位数字**

```
Dataset002_Liver      ✓
Dataset2_Liver        ✗
Dataset_002_Liver     ✗
```

**后边的名字随便起，但要有意义**（训练命令里会用到编号）。

**② 模态后缀 `_0000`、`_0001`**

**这是最容易被忽略的一条。**

```
imagesTr/liver_001_0000.nii.gz     ← CT 只有一个模态，用 _0000
labelsTr/liver_001.nii.gz          ← 标签没有后缀
```

**多模态时：**

```
imagesTr/case_001_0000.nii.gz      ← T1
imagesTr/case_001_0001.nii.gz      ← T2
imagesTr/case_001_0002.nii.gz      ← FLAIR
labelsTr/case_001.nii.gz
```

**为什么：** nnU-Net 靠这个后缀区分不同的输入通道。**少了它，训练时会报「找不到对应的标签」或者把标签当图像读。**

**③ 标签必须是无符号整数**

```python
label = label.astype(np.uint8)     # 或 uint16
```

**浮点数的标签会被拒绝。**

### dataset.json

```python
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

generate_dataset_json(
    output_folder=r"F:\nnunet\raw\Dataset002_Liver",
    channel_names={0: "CT"},              # 模态编号 -> 名称
    labels={
        "background": 0,
        "liver": 1,
        "tumor": 2,
    },
    num_training_cases=131,
    file_ending=".nii.gz",
)
```

**注意 `labels` 是「名字: 数值」，v1 时代是反过来的。** 很多老教程还写着旧格式。

---

## 五、五步命令

### ① 规划 + 预处理

```
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```

| 参数 | 作用 |
|---|---|
| `-d 2` | 数据集编号 |
| `--verify_dataset_integrity` | **第一次跑必加**，检查数据完整性和标签 |
| `-c 2d 3d_fullres` | 只做指定配置（可选） |
| `-np 8` | 并行进程数 |

**这一步会跑几分钟到几十分钟，取决于数据量。**

**输出：**

```
nnUNet_preprocessed/Dataset002_Liver/
├── nnUNetPlans.json              ← 规划结果，打开看能知道它算出了什么
├── dataset_fingerprint.json      ← 指纹
├── gt_seg_maps.npz
└── nnUNetPlans_3d_fullres/       ← 预处理后的 npy
```

**跑完第一件事：打开 `nnUNetPlans.json` 看 `patch_size` 和 `batch_size` 是多少。**

**这是理解它「自配置」最直接的方式。**

### ② 训练

```
nnUNetv2_train 2 3d_fullres 0
```

| 位置参数 | 含义 |
|---|---|
| `2` | 数据集编号 |
| `3d_fullres` | 配置名（另有 `2d`、`3d_lowres`、`3d_cascade_fullres`） |
| `0` | 第几折（0-4），或者 `all` 表示全部数据 |

**默认 1000 个 epoch，5 折交叉验证。**

**8 GB 显存的现实：**

```
nnUNetv2_train 2 2d 0                    ← 2D 配置，能跑
nnUNetv2_train 2 3d_fullres 0            ← 3D 全分辨率，大概率 OOM
```

**3D 跑不动怎么办，见下一节。**

### ③ 找最佳配置

```
nnUNetv2_find_best_configuration 2 -c 2d 3d_fullres
```

**它会跑完所有折，算出每折的 Dice，然后告诉你：该用哪个配置、后处理要不要开。**

**输出写进 `inference_instructions.txt`。**

### ④ 推理

```
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 2 -c 2d -f 0
```

| 参数 | 含义 |
|---|---|
| `-i` / `-o` | 输入/输出目录，输入要带 `_0000` 后缀 |
| `-c` | 用哪个配置 |
| `-f` | 用哪一折的权重，`-f all` 表示用 5 折集成 |
| `-step_size` | 滑窗步长，默认 0.5（重叠 50%） |

**滑窗推理：** 整个 3D 体积放不进去，所以用固定大小的窗口一格格滑过去，重叠部分取平均。`step_size=0.5` 表示每次滑半个窗口。

### ⑤ 应用后处理

```
nnUNetv2_apply_postprocessing \
    -i pred_dir -o final_dir \
    -pp_pkl_file nnUNet_results/Dataset002_Liver/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/postprocessing.pkl \
    -np 8 -plans_json nnUNetPlans.json
```

**参数在 `inference_instructions.txt` 里都写好了，照着抄。**

**后处理通常是「移除小于某个体积的连通域」或者「只保留最大的一个」。** 对肝脏这类「只有一个」的器官特别有效，对多发病灶要慎用。

---

## 六、8 GB 显存跑不动 3D 怎么办

**四条路，从简单到复杂。**

### 路一：改用 2D 配置（最简单）

```
nnUNetv2_train 2 2d 0
```

**逐层切，用 2D U-Net。** Dice 通常比 3D 低 2-5 个点，但**能跑起来。**

**对肝脏这种大的、层间连续的器官，2D 效果不差。**

### 路二：限制 3D patch 大小

**改 `nnUNetPlans.json` 里的 `patch_size`：**

```json
"patch_size": [96, 96, 96]      ← 从默认的可能 128³ 降下来
"batch_size": 1                  ← 同时降到 1
```

**改完要重新跑预处理：**

```
nnUNetv2_preprocess -d 2 -c 3d_fullres -plans_name nnUNetPlans
```

**注意：改了 plans 后必须重新预处理，否则尺寸对不上。**

### 路三：用 `3d_lowres` 配置

```
nnUNetv2_train 2 3d_lowres 0
```

**低分辨率版，patch 小，显存够。** 精度差一些，但比 2D 保留了层间信息。

### 路四：开混合精度

**nnU-Net 默认已经开了 AMP。** 如果还是 OOM，说明真的不够。

---

## 七、跟着做

### 任务：用自己的数据跑通五步

**数据来源（挑一个）：**

- **MedMNIST 里的 2D 数据**（如 `pneumoniamnist`）——最简单，但要转成 NIfTI
- **LiTS 的部分数据**（20 例够验证流程）
- **模块 11 你跑过的分割数据**

**最小可行版本：**

```python
# 把 2D 的 npz 转成 NIfTI
import numpy as np
import nibabel as nib
from pathlib import Path

RAW = Path(r"F:\nnunet\raw\Dataset100_MNIST")
(RAW / "imagesTr").mkdir(parents=True, exist_ok=True)
(RAW / "labelsTr").mkdir(parents=True, exist_ok=True)

data = np.load("pneumoniamnist.npz")
for split in ["train"]:
    imgs = data[f"{split}_images"]
    for i in range(min(50, len(imgs))):        # 先用 50 例
        im = imgs[i].squeeze().astype(np.float32)
        nib.save(nib.Nifti1Image(im, np.eye(4)),
                 str(RAW / "imagesTr" / f"case_{i:03d}_0000.nii.gz"))
        # 标签随便造一个圆，只为跑通流程
        lab = (im > np.median(im)).astype(np.uint8)
        nib.save(nib.Nifti1Image(lab, np.eye(4)),
                 str(RAW / "labelsTr" / f"case_{i:03d}.nii.gz"))

print("转换完成")
```

**然后依次执行五步。**

**验收：**

- [ ] `nnUNetv2_plan_and_preprocess` 不报错
- [ ] 打开生成的 `nnUNetPlans.json`，找到 `patch_size` 和 `batch_size` 各是多少，**写下来**
- [ ] `nnUNetv2_train` 至少跑完 1 个 epoch（可以改 epoch 数）
- [ ] 推理能输出 NIfTI 文件

**第二项是重点。** 你要亲眼看到它算出来的数字，才算理解「自配置」是什么意思。

---

## 八、练习

### 练习 18-1-A（必做）：对比指纹

跑两个数据集（比如一个 28×28 的、一个 256×256 的），各看一次 `dataset_fingerprint.json`。

**填表：**

| 数据集 | 图像尺寸中位数 | 体素间距 | 类别占比 | 算出的 patch_size |
|---|---|---|---|---|
| 小的 | | | | |
| 大的 | | | | |

**结论写一句话：patch_size 主要受什么影响？**

### 练习 18-1-B（必做）：只跑 2 个 epoch

找到 nnU-Net 的 trainer 类，把 `num_epochs` 改小，验证流程能通。

**为什么要改：** 默认 1000 epoch 要跑几小时。验证流程时用 2 个就够。

**提示：** 可以继承 `nnUNetTrainer` 写一个自己的 trainer，然后用 `-tr` 参数指定。

### 练习 18-1-C（必做）：故意制造一个格式错误

把某一个 label 文件的后缀去掉 `_0000` 的对应关系（比如把 `case_003_0000.nii.gz` 改名成 `case_003.nii.gz`）。

**看 `--verify_dataset_integrity` 报什么错。**

**这个错误信息你以后会经常见到，记下来。**

### 练习 18-1-D（选做）：对比 2D 和 3D

同一个数据集，分别跑 `2d` 和 `3d_lowres` 两个配置各 5 个 epoch。

**记录：单 epoch 耗时、显存峰值、验证 Dice。**

---

## 九、自测题

1. nnU-Net 最大的贡献是提出了什么新网络吗？
2. 「自配置」具体配了哪四样东西？
3. patch size 是怎么算出来的？受什么限制？
4. 为什么图像文件名要有 `_0000` 后缀？
5. `dataset.json` 里的 `labels` 是「名字:数值」还是「数值:名字」？
6. Windows 上用 nnU-Net 有哪三个坑？
7. 8 GB 显存跑 3D 有哪四条路？
8. 滑窗推理的 `step_size=0.5` 是什么意思？

---

## 十、出口标准

- [ ] 能说清 nnU-Net 为什么「不设计网络」也能赢
- [ ] 能解释指纹 → 规划 → 预处理这三步各自在干什么
- [ ] 能把自己的数据转成 `DatasetXXX_Name` 格式
- [ ] 能跑通五步命令，不查文档
- [ ] 能根据显存选择 2D / 3d_lowres / 改 patch 三条路
- [ ] 能读懂 `nnUNetPlans.json` 里的关键字段

---

## 十一、常见错误

| 错误 | 现象 | 原因 | 改法 |
|---|---|---|---|
| 数据集命名不对 | 找不到数据集 | 编号不是三位数 | `Dataset002_Name` |
| 缺 `_0000` 后缀 | 训练报「通道不匹配」 | 模态识别失败 | 加后缀 |
| 标签是浮点 | 预处理报错 | 类型不对 | `astype(np.uint8)` |
| 路径有中文 | `FileNotFoundError` | batchgenerators 不吃非 ASCII | 用纯英文路径 |
| 路径过长 | 写入失败 | Windows 260 字符限制 | 缩短目录或开长路径 |
| 改了 plans 没重预处理 | 尺寸不匹配 | 预处理产物是旧的 | 重跑 `nnUNetv2_preprocess` |
| 忘了设环境变量 | 到处找不到文件 | 三个变量没配 | `setx` 配好后重开终端 |
| 直接上 3D 全分辨率 | OOM | 显存不够 | 先跑 2D |

---

## 十二、下一节要干什么

nnU-Net 是「拿来就能用」的黑盒。**会用它，简历上能写「熟练使用 nnU-Net」。**

**但面试官会追问：你知道它内部在干什么吗？如果换个数据集，它的默认配置不work，你怎么改？**

**下一节 18-2 要做的，就是用 MONAI 从零搭一遍 3D 流程**——重采样、patch 裁剪、滑窗推理、后处理，每一步自己写。

**跑通那一遍，nnU-Net 对你就不再是黑盒了。**
