# 18-6 ONNX 导出与 TensorRT 加速

**建议用时：8 小时**

**出口标准：能把 PyTorch 模型导出成 ONNX，验证精度一致，并测出真实的加速比。**

---

## 〇、环境说明

**这一节需要装：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install onnx onnxruntime
```

**约 30 MB，不大。**

**TensorRT 单独说：它在 Windows 上装起来麻烦，而且和 CUDA 版本强绑定。本节会给你一条不依赖 TensorRT 的路径。**

**本节代码我还没实跑（包没装），跑完请把耗时和报错发我。**

---

## 一、为什么要部署，PyTorch 有什么问题

**你在笔记本上训好一个模型，想给别人用。对方可能是：**

- 医院的 PACS 工作站（只有 CPU，装不了 PyTorch）
- 一个 C++ 写的影像后处理软件（不能调 Python）
- 边缘设备（Jetson，资源受限）

**PyTorch 在这三种场景下都不合适：**

| 问题 | 具体表现 |
|---|---|
| **依赖重** | 光 torch 就 4.3 GB（你自己环境实测过） |
| **启动慢** | `import torch` 要 3-5 秒 |
| **Python 生态** | C++ 项目集成困难 |
| **优化器不成熟** | 图优化、算子融合能力有限 |

**ONNX 的思路：**

> **把模型导出成一个「计算图文件」，任何支持 ONNX 的运行时都能执行它，不管底层是什么框架、什么语言。**

**类比：PyTorch 的模型是 Word 文档，ONNX 是 PDF。谁都能打开，但不能改。**

---

## 二、ONNX 导出

### 基本流程

```python
import torch
from monai.networks.nets import UNet

model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
             channels=(16, 32, 64), strides=(2, 2), num_res_units=2)
model.eval()                       # ← 必须，否则 BN/Dropout 行为不一致

dummy = torch.randn(1, 1, 96, 96, 96)

torch.onnx.export(
    model,
    dummy,
    "unet3d.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input":  {0: "batch"},     # batch 维度可变化
        "output": {0: "batch"},
    },
)
print("导出完成")
```

### 五个必须注意的地方

**① `model.eval()` 不能忘**

**训练模式下 BatchNorm 用的是当前 batch 的统计量，导出后推理时没有 batch 概念，结果会完全不对。**

**这个坑很隐蔽**——导出不报错，推理结果就是错的。

**② `opset_version` 选多少**

| 版本 | 说明 |
|---|---|
| 11 | 老，兼容性最好 |
| **17** | **推荐的默认值**，支持大部分算子 |
| 20+ | 新算子，但部分运行时还不支持 |

**3D 卷积用 17 基本没问题。如果导出报「不支持的算子」，先降到 13/11，再不行才考虑升级。**

**③ `dynamic_axes` 设不设**

**不设的话，导出的模型 batch 固定为 1。** 推理时传 batch=2 会报错。

**分割任务建议只把 batch 设成动态，空间维度保持固定。**

```
为什么：分割的输入尺寸变化会导致内部特征图尺寸变化，
某些算子（比如固定窗口的注意力）不支持动态尺寸。
```

**SwinUNETR 就属于这类，空间维度不能动态。**

**④ 别忘了 `torch.no_grad()`**

导出时用 `dummy` 跑一遍前向。不加 `no_grad` 会占额外显存。

```python
with torch.no_grad():
    torch.onnx.export(...)
```

**注意：新版 PyTorch 的 `export` 内部会处理好，但养成习惯。**

**⑤ 自定义算子会失败**

**如果你的模型里用了 `torch.nn.functional.grid_sample`、自定义的 CUDA 算子、动态控制流（if/while 依赖张量值），导出会失败。**

**报错关键字：**

```
RuntimeError: ONNX export failed: Couldn't export operator xxx
```

**解决办法：**

- 换成 ONNX 支持的等价实现
- 或者用 `torch.onnx.export` 的 `custom_opsets` 参数
- 或者放弃 ONNX，直接部署 PyTorch（配合 TorchScript）

---

## 三、验证导出是否正确

**导出不报错 ≠ 结果正确。** 必须做数值比对。

```python
import numpy as np
import onnxruntime as ort

# ---------- PyTorch 结果 ----------
x = torch.randn(1, 1, 96, 96, 96)
with torch.no_grad():
    torch_out = model(x).numpy()

# ---------- ONNX 结果 ----------
sess = ort.InferenceSession("unet3d.onnx", providers=["CUDAExecutionProvider"])
onnx_out = sess.run(None, {"input": x.numpy()})[0]

# ---------- 比对 ----------
diff = np.abs(torch_out - onnx_out)
print(f"最大绝对误差: {diff.max():.6f}")
print(f"平均绝对误差: {diff.mean():.8f}")
print(f"相对误差:     {(diff / (np.abs(torch_out) + 1e-8)).max():.2e}")
```

### 判定标准

| 误差量级 | 说明 |
|---|---|
| **1e-6 以下** | 正常，浮点误差 |
| 1e-4 左右 | 可以接受，通常是算子实现差异 |
| **1e-2 以上** | **有问题**，检查 eval 模式、opset、动态维度 |
| 结果完全不一样 | 大概率忘了 `model.eval()` |

**分割任务的特殊之处：**

**输出要经过 argmax 才变成掩码。** 阈值附近的小误差可能导致某些体素分类改变。

**所以要额外验证 Dice：**

```python
from monai.metrics import DiceMetric

dice = DiceMetric(include_background=False, reduction="mean")
dice(y_pred=torch.from_numpy(torch_out), y=torch.from_numpy(onnx_out))
print(f"Dice(torch vs onnx) = {dice.aggregate().item():.4f}")
```

**这个数应该在 0.999 以上。低于 0.99 就要查原因。**

---

## 四、onnxruntime 推理

```python
import onnxruntime as ort
import numpy as np

# ---------- 选 provider ----------
providers = ort.get_available_providers()
print("可用:", providers)          # 通常有 ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = ort.InferenceSession(
    "unet3d.onnx",
    providers=["CUDAExecutionProvider"],     # 想用 GPU
)

# ---------- 看输入输出 ----------
for i in sess.get_inputs():
    print(f"输入: {i.name}, shape={i.shape}, dtype={i.type}")
for o in sess.get_outputs():
    print(f"输出: {o.name}, shape={o.shape}, dtype={o.type}")

# ---------- 推理 ----------
x = np.random.randn(1, 1, 96, 96, 96).astype(np.float32)
out = sess.run(None, {"input": x})[0]
print(out.shape)
```

### 一个常见的坑：dtype 必须是 float32

```
onnxruntime 不接受 float64，也不接受 int64 当作图像输入
```

**报错：**

```
[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))
```

**改法：** `x.astype(np.float32)`

**这个坑在你从 numpy 读 DICOM 数据时特别容易踩——`pydicom` 读出来经常是 int16。**

---

## 五、TensorRT：更快的路

### 它比 ONNX Runtime 快在哪

| 优化 | 效果 |
|---|---|
| **层融合** | Conv+BN+ReLU 合成一个算子 |
| **精度校准** | FP16/INT8，显存减半、速度翻倍 |
| **kernel 自动调优** | 针对你的具体 GPU 选最快的实现 |
| **显存复用** | 提前规划，减少分配 |

**典型加速比：ONNX Runtime GPU 的 1.5-3 倍。**

### 但它在 Windows 上很难装

**三个原因：**

1. **版本强绑定**：TensorRT 8.x / 10.x 各自支持特定 CUDA 版本，和你的 torch 的 CUDA 版本不一定对得上
2. **RTX 5060 是 sm_120**（新架构），老版本 TensorRT 不支持
3. **pip 装的 `tensorrt` 包经常缺 DLL**

### 更实际的路径

**你的显卡是 RTX 5060 Laptop，8 GB。**

**我建议：**

```
第一步：ONNX Runtime（CUDA 版）  ← 先跑通
第二步：ONNX Runtime + FP16      ← 容易的加速
第三步：TensorRT                 ← 有余力再折腾
```

**前两步就能拿到大部分收益，而且不折腾。**

### ONNX Runtime 的 FP16 加速

```python
from onnxruntime.transformers import float16

# 把 fp32 的 onnx 转成 fp16
float16.convert_float_to_float16(
    "unet3d.onnx",
    "unet3d_fp16.onnx",
    keep_io_types=True,        # 输入输出还是 fp32，内部用 fp16
)
```

**需要 `pip install onnxruntime-tools`。**

**效果：显存减半，速度通常提升 1.3-2 倍。精度损失很小（分割任务通常 Dice 掉不到 0.001）。**

**注意 `keep_io_types=True`：** 保持输入输出是 fp32，这样你不需要改推理代码。**转换后的模型在内部用 fp16 算，但对外的接口不变。**

### 如果一定要 TensorRT

**在 Windows 上的大致步骤：**

```
1. 下载 TensorRT 的 Windows zip（要 NVIDIA 开发者账号）
2. 解压，把 lib 目录加到 PATH
3. pip install tensorrt
4. 用 trtexec 把 onnx 转成 engine
   trtexec --onnx=unet3d.onnx --saveEngine=unet3d.engine --fp16
5. Python 里用 tensorrt 加载 engine
```

**第 4 步是关键：** `trtexec` 会打印每一层的耗时，能看出瓶颈在哪。

**但你的 RTX 5060 是 sm_120，需要 TensorRT 10.x 以上。老教程里的 8.x 版本不支持。**

---

## 六、测出真实的加速比

**这是这一节唯一有价值的产出。别只看「导出成功」，要看快了多少。**

```python
import time
import numpy as np
import torch
import onnxruntime as ort

def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n

x_np = np.random.randn(1, 1, 96, 96, 96).astype(np.float32)
x_t = torch.from_numpy(x_np).cuda()

# PyTorch
model.eval().cuda()
t_torch = bench(lambda: model(x_t))

# ONNX Runtime
sess = ort.InferenceSession("unet3d.onnx", providers=["CUDAExecutionProvider"])
t_onnx = bench(lambda: sess.run(None, {"input": x_np}))

print(f"PyTorch:      {t_torch*1000:7.1f} ms")
print(f"ONNX Runtime: {t_onnx*1000:7.1f} ms")
print(f"加速比:       {t_torch/t_onnx:.2f}x")
```

### 关于 `torch.cuda.synchronize()`

**CUDA 是异步的。** 你调用 `model(x)` 之后，CPU 立刻返回，GPU 还在算。

**不 `synchronize` 就直接计时，测出来的是「提交任务的时间」，不是「算完的时间」——会快得离谱。**

**这一行不加，你的加速比全是假的。**

---

## 七、部署时还要考虑什么

**除了速度，真实部署还有四件事：**

### ① 显存峰值

```python
torch.cuda.reset_peak_memory_stats()
# ... 推理 ...
print(f"峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

**医院的工作站可能只有一块 4 GB 的老显卡。** 你的模型在 8 GB 上跑得动，不代表别人能用。

### ② 滑窗推理的开销

**整卷 CT 是 512×512×200，模型只吃 96³。**

**滑窗推理的次数：**

```
(512/48) × (512/48) × (200/48) ≈ 11 × 11 × 5 = 605 个窗口
```

**每个窗口 50 ms，就是 30 秒。** 这已经超出临床可接受范围（通常要求 10 秒内）。

**优化方向：**

- 降低 overlap（0.5 → 0.25），次数减半，精度略降
- 用更大的 patch（128³），窗口数减少
- 先下采样到 256×256 推理，再上采样回原尺寸

### ③ 后处理不能省

**模型输出是概率图，后处理（去小斑块、阈值、连通域）也是必须执行的步骤。**

**部署时要把后处理一起打包，或者至少提供参考实现。**

### ④ 版本锁定

**部署环境和训练环境不一致，是「我这能跑你那边报错」的第一大原因。**

```text
# 训练和部署都要固定版本
torch==2.9.1
onnx==1.17.0
onnxruntime-gpu==1.20.0
```

**用 `pip freeze > requirements.txt` 存下来。**

---

## 八、跟着做

### 任务：导出一个模型，测出加速比

**步骤：**

1. 用 18-2 训的 3D U-Net（或者随便建一个）
2. 导出成 ONNX，`opset_version=17`，batch 动态
3. 验证 PyTorch 和 ONNX 的最大绝对误差
4. 测两者的推理耗时
5. 算加速比

**验收：**

- [ ] 导出的 `.onnx` 文件能加载
- [ ] **最大绝对误差 < 1e-4**
- [ ] **两个耗时都测出来了（记得 synchronize）**
- [ ] 算出了加速比

**如果 ONNX 比 PyTorch 还慢，检查：**

- provider 是不是真的用了 CUDA（`sess.get_providers()`）
- 第一次推理包含初始化，不算进平均

---

## 九、练习

### 练习 18-6-A（必做）：忘掉 eval 会怎样

**导出时不调 `model.eval()`，直接用训练模式导出。**

**然后对比结果误差。**

**记录：误差有多大？为什么？**

**（提示：BatchNorm 在两种模式下用不同的统计量。）**

### 练习 18-6-B（必做）：动态维度的边界

**试三种动态设置：**

| 设置 | 导出能否成功 | batch=2 能否推理 |
|---|---|---|
| 不设 dynamic_axes | | |
| 只动态 batch | | |
| batch + 空间都动态 | | |

**把结果填进去。**

**对 SwinUNETR 再试一次，看是不是失败。**

### 练习 18-6-C（必做）：FP16 的收益和代价

**用 `convert_float_to_float16` 转一版。**

**对比三项：**

| | FP32 | FP16 |
|---|---|---|
| 模型大小 | | |
| 推理耗时 | | |
| Dice（和原始结果比） | | |

**结论写一句话：值不值得转？**

### 练习 18-6-D（选做）：滑窗推理的部署成本

**算一下你的模型处理一整卷 512×512×200 的 CT 需要多久。**

**填表：**

| overlap | 窗口数 | 预估耗时 |
|---|---|---|
| 0.25 | | |
| 0.5 | | |
| 0.75 | | |

**这个数字，是你面试时回答「你的系统能用吗」的依据。**

---

## 十、自测题

1. ONNX 解决了 PyTorch 的什么问题？
2. 导出时为什么必须 `model.eval()`？不调会怎样？
3. `opset_version` 怎么选？
4. `dynamic_axes` 设了和不设有什么区别？什么情况下不能动态？
5. 导出不报错，就等于结果正确吗？怎么验证？
6. 为什么测耗时必须调 `torch.cuda.synchronize()`？
7. TensorRT 比 ONNX Runtime 快在哪三点？
8. 部署时除了速度还要考虑哪四件事？

---

## 十一、出口标准

- [ ] 能导出 ONNX 并验证数值一致性
- [ ] 知道导出失败的常见原因和排查顺序
- [ ] 能用 onnxruntime 推理并选对 provider
- [ ] 能测出真实的加速比（会 synchronize）
- [ ] 能算出整卷 CT 的推理耗时
- [ ] 能说清 TensorRT 值不值得折腾

---

## 十二、常见错误

| 错误 | 现象 | 原因 | 改法 |
|---|---|---|---|
| 忘了 `model.eval()` | 结果全错但不报错 | BN 用的是 batch 统计量 | 导出前一定调 |
| 输入 dtype 是 float64 | INVALID_ARGUMENT | numpy 默认 float64 | `.astype(np.float32)` |
| 动态维度导致导出失败 | RuntimeError | 算子不支持动态尺寸 | 空间维度固定 |
| 测速忘了 synchronize | 加速比虚高十几倍 | CUDA 异步 | 加 `torch.cuda.synchronize()` |
| 用 CPU provider | 比 PyTorch 还慢 | 没指定 CUDA | `providers=["CUDAExecutionProvider"]` |
| 只看导出成功 | 上线才发现结果不对 | 没做数值验证 | 必须比 max abs diff |
| 部署环境版本不一致 | 别人那跑不通 | 没锁版本 | `pip freeze` |

---

## 十三、这个模块学完了

**你现在应该有的东西：**

```
一个 nnU-Net baseline         （18-1）
一个自己搭的 3D 分割流程       （18-2）
一个能用的交互式分割工具       （18-3）
一套标注质量评估方法           （18-4）
一张三个架构的对比表           （18-5）
一个能部署的推理引擎           （18-6）
```

**下一步：串成毕业项目。**

**见 `项目_肝肿瘤分割对比研究.md`。**
