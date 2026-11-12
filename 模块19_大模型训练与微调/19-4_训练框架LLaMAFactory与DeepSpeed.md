# 19-4 训练框架：LLaMA-Factory 与 DeepSpeed

**建议用时：10 小时**

**出口标准：能用 LLaMA-Factory 一条命令完成微调，能说清 DeepSpeed 的 ZeRO 三个阶段各在优化什么。**

---

## 〇、环境说明

**这一节需要装 LLaMA-Factory：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install llamafactory
```

**或从源码装（功能更全）：**

```
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**约 500 MB。**

**本节代码我还没实跑。**

---

## 一、为什么需要框架

**19-3 你手写了训练脚本。它能跑，但有一堆重复的事：**

| 重复的事 | 每次都要写 |
|---|---|
| 数据格式转换 | ✅ |
| 对话模板套用 | ✅ |
| LoRA 配置 | ✅ |
| 训练参数 | ✅ |
| 显存优化开关 | ✅ |
| checkpoint 保存与恢复 | ✅ |
| 日志 | ✅ |
| 评估 | ✅ |

**换个模型、换个数据集，这些全要重来一遍。**

**框架做的事：把这些都配置化。**

---

## 二、LLaMA-Factory 是什么

**一个开源的中文友好的微调框架，支持：**

- 100+ 个模型（LLaMA、Qwen、ChatGLM、Baichuan、DeepSeek……）
- SFT、DPO、PPO、KTO、ORPO 全流程
- LoRA、QLoRA、全量微调
- 有 WebUI（图形界面）

**最大优点：一条命令 + 一个 yaml 文件，就能跑。**

### 安装

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**或者简单点：**

```
pip install llamafactory
```

**装完验证：**

```
llamafactory-cli version
```

### 数据准备

**LLaMA-Factory 用 `dataset_info.json` 注册数据集。**

**第一步：把你的数据转成它的格式。**

```json
[
  {
    "instruction": "肝窗的窗宽窗位是多少？",
    "input": "",
    "output": "肝窗常用窗宽 160 HU、窗位 60 HU。"
  },
  {
    "instruction": "判断这例有没有肺炎。",
    "input": "右肺下叶见斑片状高密度影。",
    "output": "提示右肺下叶炎症可能。"
  }
]
```

**支持多种格式：**

| 格式 | 字段 |
|---|---|
| **alpaca** | `instruction` / `input` / `output` |
| **sharegpt** | `conversations`（多轮） |
| **openai** | `messages`（同 19-3） |

**建议用 alpaca 格式——最简单。**

**第二步：在 `data/dataset_info.json` 里注册。**

```json
{
  "medical_qa": {
    "file_name": "medical_qa.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 配置文件

**新建 `train_lora.yaml`：**

```yaml
### 模型
model_name_or_path: Qwen/Qwen3-1.7B
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj
lora_dropout: 0.05

### 数据
dataset: medical_qa
template: qwen3                # ← 对话模板，必须和模型匹配
cutoff_len: 1024
preprocessing_num_workers: 4
max_samples: 1000              # 先用 1000 条试试

### 训练
output_dir: saves/qwen3-1.7b/lora/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
gradient_checkpointing: true

### 日志与保存
logging_steps: 10
save_steps: 200
plot_loss: true
```

**`template` 这一行特别重要。** 填错了，模型学不会对话格式。

**Qwen 系用 `qwen` 或 `qwen3`，LLaMA 用 `llama3`。** 具体查文档。

### 启动

```
llamafactory-cli train train_lora.yaml
```

**就这一条命令。**

### 出一张损失曲线

```
llamafactory-cli train train_lora.yaml
```

**训练完会自动在 `output_dir` 里生成 `training_loss.png`。**

### 合并并导出

```yaml
# export.yaml
model_name_or_path: Qwen/Qwen3-1.7B
adapter_name_or_path: saves/qwen3-1.7b/lora/sft
template: qwen3
finetuning_type: lora
export_dir: models/qwen3-1.7b-medical
export_size: 2
```

```
llamafactory-cli export export.yaml
```

### 还有个 WebUI

```
llamafactory-cli webui
```

**打开浏览器就能配参数、看训练曲线。适合不想写 yaml 的时候。**

---

## 三、DeepSpeed：为什么能让 7B 在单卡上训

**你的 8 GB 显存训不了 7B。但实验室的服务器能。**

**DeepSpeed 是微软的分布式训练库，核心是 ZeRO（Zero Redundancy Optimizer）。**

### 显存都花在哪了

**全量微调一个模型，显存分四块：**

| 项 | 7B 模型要多少 |
|---|---|
| 权重（fp16） | 14 GB |
| 梯度（fp16） | 14 GB |
| **优化器状态（fp32）** | **56 GB**（一阶+二阶动量 + fp32 主权重） |
| 激活值 | 取决于 batch 和序列长度 |

**总计轻松超过 80 GB。**

### ZeRO 的三个阶段

**核心思想：这些数据，不同 GPU 之间**可以只存一部分**。**

| 阶段 | 分片什么 | 7B 模型每卡显存（8 卡） |
|---|---|---|
| **ZeRO-1** | 优化器状态 | 约 32 GB |
| **ZeRO-2** | 优化器状态 + 梯度 | 约 18 GB |
| **ZeRO-3** | **全部（含权重）** | **约 9 GB** |

**ZeRO-3 才能在 8 张 24 GB 卡上跑 7B 全量微调。**

### 代价是什么

**分片要有通信开销。**

```
ZeRO-1：通信量 = 2 × 参数量
ZeRO-2：通信量 ≈ 2 × 参数量 + 梯度
ZeRO-3：通信量 ≈ 3 × 参数量       ← 通信最多
```

**所以：**

```
ZeRO-1/2 → 能放得下 → 快
ZeRO-3   → 放不下才用 → 慢但能跑
```

**经验法则：能用 ZeRO-2 就别用 ZeRO-3。**

### 配置示例

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 4,
  "fp16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "overlap_comm": true
  }
}
```

**`offload_optimizer` 是关键：把优化器状态放到 CPU 内存里。**

**你的机器有 32 GB 内存，这就是「用内存换显存」。**

**代价是慢 2-3 倍（PCIe 带宽限制）。**

---

## 四、FSDP：PyTorch 官方的方案

**Meta 的 FSDP（Fully Sharded Data Parallel）做的是和 ZeRO-3 类似的事。**

| | DeepSpeed | FSDP |
|---|---|---|
| 出品 | 微软 | PyTorch 官方 |
| 成熟度 | 高 | 高 |
| 配置 | 复杂（JSON） | 简单（代码里包装） |
| 生态 | HuggingFace Trainer 集成好 | 原生支持 |
| 推荐 | 大集群 | **单机多卡** |

**你的场景（单卡）用不上，知道有这个东西就行。**

---

## 五、你的情况该用哪个

**8 GB 单卡，1.7B 模型微调。**

| 方案 | 需要吗 |
|---|---|
| LLaMA-Factory | ✅ **推荐**，省事 |
| 手写 peft 脚本 | ✅ **必须会**，理解原理 |
| DeepSpeed | ❌ 用不上（单卡） |
| FSDP | ❌ 用不上（单卡） |
| 多卡分布式 | ❌ 没条件 |

**结论：**

```
19-3 学原理（手写） → 19-4 学效率（框架）
```

**两个都要会。只会框架的人，出问题不知道从哪查；只会手写的人，换个模型重写一遍。**

---

## 六、跟着做

### 任务：用 LLaMA-Factory 重做 19-3 的微调

**步骤：**

1. 把 19-3 的 100 条数据转成 alpaca 格式
2. 注册到 `dataset_info.json`
3. 写 `train_lora.yaml`
4. 一条命令启动
5. 对比手写版和框架版的训练 loss 曲线

**验收：**

- [ ] `llamafactory-cli train` 能跑起来
- [ ] 生成了 `training_loss.png`
- [ ] 合并导出的模型能正常回答那 5 个问题
- [ ] **能说出框架帮你做了哪三件你手写时也要做的事**

**最后一项是重点。** 用了框架不等于不懂原理。

---

## 七、练习

### 练习 19-4-A（必做）：template 填错会怎样

**把 `template: qwen3` 改成 `template: llama3`，重新跑。**

**看训练 loss 和最终输出有什么变化。**

**记录：模型回答变成什么样了？**

**这一题让你理解「对话模板」不是装饰，它决定了模型看到的 token 序列。**

### 练习 19-4-B（必做）：对比手写和框架

**同一个数据集、同一组超参，分别用手写 peft 和 LLaMA-Factory。**

**填表：**

| | 手写 | 框架 |
|---|---|---|
| 代码行数 | | |
| 训练耗时 | | |
| 最终 loss | | |
| 显存峰值 | | |

**结论写一句话。**

### 练习 19-4-C（必做）：算 ZeRO 的显存

**假设一个 7B 模型，8 张 24 GB 卡，全量微调（fp16 权重+梯度，fp32 优化器）。**

**算三种阶段的每卡显存：**

```
权重 14 GB，梯度 14 GB，优化器 56 GB
ZeRO-1：每卡 = 14 + 14 + 56/8 = ?
ZeRO-2：每卡 = 14 + 14/8 + 56/8 = ?
ZeRO-3：每卡 = 14/8 + 14/8 + 56/8 = ?
```

**填数，并回答：哪种能跑在 24 GB 卡上？**

### 练习 19-4-D（选做）：WebUI 探索

```
llamafactory-cli webui
```

**打开界面，找一找：它暴露了多少个训练参数？其中哪些是你在 19-3 里手写过的？**

---

## 八、自测题

1. 框架帮你做了哪几件重复的事？
2. LLaMA-Factory 的 `template` 参数是干什么的？填错会怎样？
3. 全量微调的显存花在哪四块？
4. ZeRO 三个阶段分别分片什么？每卡显存怎么算？
5. 为什么 ZeRO-3 比 ZeRO-2 慢？
6. `offload_optimizer` 是拿什么换什么？
7. FSDP 和 DeepSpeed 的关系？
8. 你的单卡场景需要 DeepSpeed 吗？

---

## 九、出口标准

- [ ] 能用 LLaMA-Factory 一条命令跑完微调
- [ ] 能写对 yaml 配置（特别是 template 和 target_modules）
- [ ] 能算出 ZeRO 各阶段的显存
- [ ] 能说清 DeepSpeed 和 FSDP 的定位
- [ ] 知道自己的场景该用哪些工具

---

## 十、常见错误

| 错误 | 现象 | 改法 |
|---|---|---|
| `template` 和模型不匹配 | 学不会对话格式 | 查文档对应关系 |
| 数据集没注册 | 找不到数据 | 改 `dataset_info.json` |
| 单卡也配 DeepSpeed | 报错或变慢 | 不需要 |
| 忘记 `cutoff_len` | 长文本被截断 | 按数据实际长度设 |
| 一次 `max_samples` 全用 | 慢且难调 | 先 1000 条验证 |

---

## 十一、下一节要干什么

**SFT 让模型「学会回答问题」。**

**但它不保证「回答得好」——可能啰嗦、可能不礼貌、可能学到数据里的坏习惯。**

**让模型「符合人类偏好」，需要另一套方法：RLHF、DPO、GRPO。**

**下一节 19-5 讲这一块。** 主要是原理，因为你的硬件做不了完整实验。

**但面试会问，而且这是「懂大模型」和「只会调 API」的分水岭。**
