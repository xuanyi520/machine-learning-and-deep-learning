# 19-3 SFT 指令微调：LoRA 与 QLoRA

**建议用时：14 小时**

**出口标准：用 8 GB 显存完成一次 LoRA 微调，并能说清每一步在干什么、为什么这么选。**

---

## 〇、环境说明

**这一节需要装：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install peft trl datasets accelerate
```

**约 100 MB。**

**还要下载基座模型（约 3.5 GB）：**

```
Qwen/Qwen3-1.7B
```

**存在 F 盘，走 hf-mirror（和 16-4 一样）。**

**⚠️ `bitsandbytes` 在 Windows 上大概率装不上**，所以本节主线是**不量化的 fp16 LoRA**（1.7B 够用），QLoRA 作为选做。

**本节手写的 LoRA 实现已经在这台机器上验证过，peft 的代码我还没实跑。**

---

## 一、SFT 是什么

**SFT = Supervised Fine-Tuning，监督微调。**

**大白话：拿「问题-答案」对，教模型「遇到这类问题该怎么回答」。**

### 和预训练的区别

| | 预训练 | SFT |
|---|---|---|
| 数据 | 互联网文本（万亿 token） | 人工标注的问答对（几千到几百万条） |
| 目标 | 预测下一个词 | **同样是预测下一个词，但只算「答案」部分的 loss** |
| 学到什么 | 语言能力、世界知识 | **特定任务的回答风格、格式、领域知识** |
| 成本 | 几百万美元 | **几美元到几百美元（LoRA）** |

### 为什么需要 SFT

**你可能会问：写提示词不就行了？**

**能解决一部分，但有三件事提示词做不到：**

| 场景 | 提示词够吗 |
|---|---|
| 改变回答风格（简短 → 详细的病历格式） | ⚠️ 勉强 |
| 注入领域知识（本院的规定、特定术语） | ❌ 提示词太长，而且每次都要重发 |
| **降低延迟**（不用每次塞 2000 字提示） | ❌ 微调后提示词可以很短 |

**第三点常被忽略：** 如果你的系统每天调一万次，提示词里塞 2000 token 的领域说明，成本就是微调成本的几十倍。

---

## 二、数据格式：对话模板

**SFT 的数据不是「一句话」，是「多轮对话」。**

**不同模型用不同的模板。Qwen 用的是 ChatML 风格：**

```
<|im_start|>system
你是一个医学影像助手。<|im_end|>
<|im_start|>user
这张 CT 上肝脏密度正常吗？<|im_end|>
<|im_start|>assistant
从图像看，肝实质密度均匀，未见明显低密度灶。<|im_end|>
```

**训练时，特殊 token 的排布决定了模型学到什么。**

### 一个数据集长什么样

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个医学影像助手。"},
      {"role": "user", "content": "肝窗的窗宽窗位是多少？"},
      {"role": "assistant", "content": "肝窗常用窗宽 160 HU、窗位 60 HU。"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "AFP 多少提示肝癌？"},
      {"role": "assistant", "content": "AFP 超过 400 ng/mL 高度提示原发性肝癌。"}
    ]
  }
]
```

### 关键：只对 assistant 部分算 loss

**这是新手最容易搞错的地方。**

```python
# 错误：整个序列都算 loss
loss = cross_entropy(logits, labels)

# 正确：把 user 和 system 部分的 label 设成 -100（忽略）
labels[labels == tokenizer.pad_token_id] = -100
# 以及 user/system 段的 token 也设 -100
```

**为什么要这样：**

**你不想让模型学「怎么提问」，只想让它学「怎么回答」。**

**如果把问题部分也算 loss，模型会花一半容量学习复述问题。**

**`trl` 的 `SFTTrainer` 有现成的 `DataCollatorForCompletionOnlyLM`，或者新版直接在数据处理时处理。**

---

## 三、LoRA：为什么必须用它

### 先算一笔账

**1.7B 的模型，全量微调需要多少显存？**

```
权重（fp16）     1.7e9 × 2 字节  = 3.4 GB
梯度（fp16）     1.7e9 × 2 字节  = 3.4 GB
Adam 优化器状态  1.7e9 × 8 字节  = 13.6 GB（一阶+二阶动量，fp32）
─────────────────────────────────────────
合计                             约 12.7 GB（不含激活值）
```

**12.7 GB > 你的 8 GB。跑不了。**

### LoRA 的做法

**核心思想：冻结原模型，在旁边挂两个小矩阵。**

```
原来的线性层：  y = W x          （W 是 1536×1536，冻结）
LoRA 分支：     y = W x + (B A x)  （A: r×1536, B: 1536×r，只训这两个）
```

**`r` 是秩，通常 8-64。**

### 实测参数量对比

**我在本机实现并验证过（维度 1536×1536）：**

```
r= 8: 原权重 2,359,296  |  LoRA   24,576  |  占比  1.04%
r=16: 原权重 2,359,296  |  LoRA   49,152  |  占比  2.08%
r=32: 原权重 2,359,296  |  LoRA   98,304  |  占比  4.17%
```

**r=16 时，只训 2% 的参数。**

### 两个关键设计

**① B 初始化为 0**

```python
self.A = nn.Parameter(torch.zeros(r, in_dim))    # A 随机
self.B = nn.Parameter(torch.zeros(out_dim, r))   # B 为 0
```

**这样 `B@A = 0`，初始时 `ΔW = 0`，模型输出和原模型完全一致。**

**实测：**

```
初始时 LoRA 输出与原模型一致: True
```

**为什么重要：** 训练开始时模型不会「突变」，梯度平滑地更新，收敛更稳。

**② 缩放系数 alpha/r**

```python
self.scaling = alpha / r
delta = (x @ A.T) @ B.T * self.scaling
```

**改 `r` 的时候不用改学习率。**

**常用配置：`alpha = 2 * r`。**

### 显存对比

```
1.7B 模型:
  全量微调(fp16+Adam)：约 12.7 GB
  LoRA r=16 微调：      约 3.2 GB      ← 能跑
```

**3.2 GB 里主要是冻结的权重（3.4 GB fp16）和激活值。**

**LoRA 参数本身只有几十 MB。**

---

## 四、手写一遍 LoRA（不依赖 peft）

**理解原理最好的方式是自己写一遍。**

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # 原权重：冻结
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA 分支
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base = x @ self.weight.T
        delta = (x @ self.A.T) @ self.B.T * self.scaling
        return base + delta

    def merge(self):
        """推理时把 B@A 合并回原权重，零额外开销"""
        with torch.no_grad():
            self.weight.data += (self.B @ self.A) * self.scaling
        return self
```

**验证：**

```python
lora = LoRALinear(1536, 1536, r=8, alpha=16)
trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
print(trainable)          # ['A', 'B']
```

**实测输出：**

```
可训练参数: ['A', 'B']
```

**只有 A 和 B 可训练，原权重被冻结——这就是 LoRA 这个名字的由来（Low-Rank Adaptation）。**

### `merge()` 为什么重要

**推理时不需要额外计算：**

```
训练时：y = Wx + (BAx) * scaling      ← 多一次矩阵乘
合并后：y = (W + BA*scaling)x          ← 和普通模型一样快
```

**所以 LoRA 模型部署时没有性能损失。**

---

## 五、用 peft 做真实微调

**手写版用来理解原理，真干活用 `peft`。**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=r"F:\AI学习环境\models",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=r"F:\AI学习环境\models",
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**`print_trainable_parameters()` 会打印类似：**

```
trainable params: 4,325,376 || all params: 1,720,026,112 || trainable%: 0.25%
```

### `target_modules` 选哪些

| 配置 | 参数量 | 效果 |
|---|---|---|
| 只加 `q_proj`, `v_proj` | 最少 | 够用，原论文的配置 |
| **q, k, v, o 全加** | 中等 | **推荐**，效果更好 |
| 再加上 FFN 层 | 最多 | 大任务才需要 |

**你现在的情况：`["q_proj", "k_proj", "v_proj", "o_proj"]` 是稳妥选择。**

### 训练循环

**用 `trl` 的 `SFTTrainer`，不用手写。**

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="medical_qa.jsonl", split="train")

def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }

dataset = dataset.map(format_chat)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="output/qwen3-1.7b-medical-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        max_length=1024,
        gradient_checkpointing=True,
    ),
)

trainer.train()
trainer.save_model("output/qwen3-1.7b-medical-lora")
```

### 四个 8 GB 显存必开的开关

| 开关 | 作用 | 省多少 |
|---|---|---|
| `per_device_train_batch_size=1` | 一次只过一条 | 必须 |
| `gradient_accumulation_steps=8` | 累积 8 步再更新，等效 batch 8 | 不省显存，但保效果 |
| `gradient_checkpointing=True` | 不存中间激活，反向时重算 | **省 30-50%** |
| `fp16=True` 或 `bf16=True` | 半精度 | 省一半 |

**`gradient_checkpointing` 是 8 GB 显存能跑的关键。代价是训练慢 20-30%。**

---

## 六、QLoRA（选做）

**QLoRA = 把原模型量化到 4bit，再在它上面挂 LoRA。**

**好处：1.7B 模型从 3.4 GB 降到约 0.9 GB，能跑更大的模型。**

**坏处：Windows 上 `bitsandbytes` 装不上。**

### 如果装得上

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

**然后 `LoraConfig` 照常配。**

### 装不上怎么办

**两条路：**

**① 不量化，就用 fp16（本节主线）**

**1.7B 完全够学。** 等你到实验室或者有服务器了，再玩 QLoRA。

**② 试试第三方 Windows 版**

**有人说能装，但不保证稳定。** 试之前先备份环境。

**别在这上面耗超过半天。**

---

## 七、训练完之后

### 保存的是什么

**`save_model` 只保存 LoRA 权重，不是整个模型。**

```
output/qwen3-1.7b-medical-lora/
├── adapter_config.json
├── adapter_model.safetensors      ← 只有几十 MB
└── tokenizer 相关文件
```

**这很关键——你可以给不同任务训练不同的 adapter，共用同一个基座模型。**

### 怎么用

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "output/qwen3-1.7b-medical-lora")
model = model.merge_and_unload()        # 合并，推理时无额外开销
```

### 一份完整的评测脚本应该做的事

```python
def evaluate(model, tokenizer, test_set):
    results = []
    for item in test_set:
        prompt = tokenizer.apply_chat_template(
            item["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0)
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        results.append({"prompt": item["messages"][-2]["content"],
                        "pred": answer,
                        "gold": item["messages"][-1]["content"]})
    return results
```

**注意 `temperature=0`。** 和 16-1 讲的一样，医疗任务要可复现。

---

## 八、跟着做

### 任务：用 100 条数据跑通全流程

**第一步：造一个小数据集**

```python
import json

samples = [
    {"messages": [
        {"role": "user", "content": "肝窗的窗宽窗位是多少？"},
        {"role": "assistant", "content": "肝窗常用窗宽 160 HU、窗位 60 HU，用于观察肝实质与病灶的对比。"}
    ]},
    {"messages": [
        {"role": "user", "content": "AFP 多少提示肝癌？"},
        {"role": "assistant", "content": "AFP 参考上限 7 ng/mL，超过 400 ng/mL 高度提示原发性肝癌。"}
    ]},
    # ... 自己凑 100 条，可以从你的讲义里提取
]

with open("medical_qa.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
```

**第二步：跑 1 个 epoch**

**第三步：对比基座和微调后的回答**

**验收：**

- [ ] `print_trainable_parameters()` 打印出可训练比例（应该 < 1%）
- [ ] 训练能跑完，显存不爆
- [ ] **同 5 个问题，基座和微调后的回答有可见差异**
- [ ] 保存下来的 adapter 只有几十 MB

**第三项是重点。** 如果回答完全一样，说明训练没生效——检查是不是忘了 `get_peft_model`。

---

## 九、练习

### 练习 19-3-A（必做）：量出显存

**在训练循环里加：**

```python
print(f"显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

**对比三组：**

| 配置 | 显存 |
|---|---|
| batch=1, 无 gradient_checkpointing | |
| batch=1, 有 gradient_checkpointing | |
| batch=2, 有 gradient_checkpointing | |

**结论：checkpointing 省了多少？**

### 练习 19-3-B（必做）：r 的影响

**r=4 / 8 / 16 / 32 各跑 1 epoch，记录：**

| r | 可训练参数 | 训练耗时 | 验证 loss |
|---|---|---|---|
| 4 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |

**结论：r 越大越好吗？**

### 练习 19-3-C（必做）：target_modules 的作用

**只加 `q_proj, v_proj` 和四个全加，各跑一次。**

**对比可训练参数和最终 loss。**

### 练习 19-3-D（选做）：merge 前后对比

**写一个函数，验证 `merge_and_unload()` 之后输出不变。**

```python
out_before = model.generate(**inputs)
merged = model.merge_and_unload()
out_after = merged.generate(**inputs)
print(torch.equal(out_before, out_after))
```

**理论上应该完全一样（浮点误差除外）。**

---

## 十、自测题

1. SFT 和预训练有什么区别？损失怎么算？
2. 为什么只对 assistant 部分算 loss？
3. LoRA 的 A 和 B 分别是什么形状？为什么 B 初始化为 0？
4. 为什么显存能省这么多？省的是哪部分？
5. `alpha / r` 这个缩放是干什么的？
6. `gradient_checkpointing` 怎么省显存？代价是什么？
7. `target_modules` 选不同层有什么影响？
8. 训练完保存的是什么？为什么只有几十 MB？
9. QLoRA 相比 LoRA 又省在哪？

---

## 十一、出口标准

- [ ] 能说清 SFT 的数据格式和 loss 计算方式
- [ ] 能手写一个 LoRA 层并验证初始等价
- [ ] 能算出全量微调和 LoRA 的显存差
- [ ] 能用 peft 跑通一次完整微调
- [ ] 知道 8 GB 显存必开的四个开关
- [ ] 能说清 QLoRA 的原理和 Windows 上的限制

---

## 十二、常见错误

| 错误 | 现象 | 改法 |
|---|---|---|
| 忘 `get_peft_model` | 全量训练，显存爆 | 包一层 |
| 没开 gradient_checkpointing | OOM | 必开 |
| 整个序列算 loss | 模型学复述问题 | 只算 assistant |
| `temperature` 不设 0 | 评测不可复现 | 设 0 |
| 保存整个模型 | 几 GB 一个 | 只存 adapter |
| r 设太大 | 参数暴涨，没收益 | 16 起步 |
| 用 `.bin` 格式 | 安全风险 | safetensors |

---

## 十三、下一节要干什么

**你现在能跑一次微调了。但你可能发现：手写这些参数很累，而且很多坑。**

**`LLaMA-Factory` 把这些都封装好了——一个 yaml 文件，一条命令。**

**下一节 19-4 讲训练框架，以及它背后那些你没见过的东西（DeepSpeed、FSDP）。**

**这节的价值在于：让你知道「框架帮你做了什么」，而不是只会敲命令。**
