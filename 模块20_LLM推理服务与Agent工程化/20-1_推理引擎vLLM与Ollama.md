# 20-1 推理引擎：vLLM 与 Ollama

**建议用时：8 小时**

**出口标准：能解释推理引擎比原生 PyTorch 快在哪，能在本地跑起服务，能测出 TTFT 和吞吐。**

---

## 〇、环境说明

**不需要装任何东西。**

**你机器上已经有：**

```
Ollama 0.40.5                          E:\Dowmload\Ollama_Dowmload\ollama
OLLAMA_MODELS = E:\Dowmload\OLLAMA_MODELS   （模型在 E 盘，不占 C 盘）

qwen3-vl:8b-instruct-q4_K_M          5.8 GB   多模态，能看图
deepseek-coder-v2:16b-Q4_K_M         9.7 GB   代码
qwen2.5:7b-instruct-q4_K_M           4.4 GB   通用
```

**本节所有数据都是在这台机器上实测的。**

---

## 一、先说结论：为什么需要推理引擎

**假设你用 19-3 的方式加载一个 7B 模型，然后写一个 Flask 接口：**

```python
model = AutoModelForCausalLM.from_pretrained(...)
model.to("cuda")

@app.post("/chat")
def chat(prompt):
    return model.generate(prompt)
```

**能跑。但如果同时来 10 个用户，会发生三件糟糕的事：**

| 问题 | 后果 |
|---|---|
| **一个请求占满 GPU** | 其他 9 个排队，等到超时 |
| **每个请求都重新分配显存** | 显存碎片，最终 OOM |
| **没法复用已算过的前缀** | 相同系统提示词算了 10 次 |

**推理引擎（vLLM、SGLang、TensorRT-LLM、Ollama）就是解决这三件事的。**

---

## 二、vLLM 的三个核心优化

### 优化一：PagedAttention（分页式 KV Cache）

**问题：**

**19-1 你算过，KV Cache 要占大量显存。** 朴素的实现是「每个请求预先分配一块连续显存，大小按最大长度算」。

```
请求 A 要 4096 token → 先占 4096 的空间
实际只用了 500 token   → 剩下 3596 的空间浪费着
```

**如果有 10 个请求，浪费就非常可观。这种浪费叫「内部碎片」。**

**PagedAttention 的做法：**

**像操作系统管理内存一样，把 KV Cache 切成固定大小的「页」（通常 16 个 token 一页），用多少分多少。**

```
请求 A 需要 500 token → 分配 32 页（512 个位置）
请求 B 需要 120 token → 分配 8 页
```

**效果：显存利用率从 20-40% 提高到 90% 以上。**

**这一条让同样显存能跑更多并发。**

### 优化二：连续批处理（Continuous Batching）

**问题：**

**传统批处理是「一批一起开始、一起结束」。**

```
批次 [A(要生成 500 token), B(要生成 10 token)]
必须等 A 生成完，B 才能释放
→ GPU 在 B 完成后就空转，等 A
```

**连续批处理：**

```
A 还在生成，B 生成完了 → B 立刻退出，C 马上能进来
```

**效果：吞吐量能提升几倍到十几倍。**

**这是所有现代推理引擎的标配。**

### 优化三：前缀缓存（Prefix Caching）

**问题：**

**你的 Agent 每次请求都带着同样的系统提示词 + 工具定义，可能有 2000 token。**

**每次都重新算一遍，纯浪费。**

**前缀缓存：**

```
第一次请求：算完系统提示词，把 KV Cache 存起来
后续请求：检测到前缀相同，直接复用
```

**效果：TTFT 能降到原来的十分之一。**

**这一条对你的 Agent 场景特别有用——因为系统提示词是固定的。**

### 一个补充：Chunked Prefill

**长输入（比如 32K 的文档）在 prefill 阶段会占满 GPU，把正在生成的请求卡住。**

**Chunked Prefill 把长输入切成小块，和生成请求交替调度，避免「一个长请求卡死所有人」。**

---

## 三、vLLM 在 Windows 上装不了（实测）

**这是必须说清楚的事实。**

```
E:\...\python.exe -m pip install --dry-run vllm
  Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
  Collecting vllm
    Downloading vllm-0.33.0.tar.gz (47.1 MB)
```

**注意：是 `.tar.gz` 源码包，不是 `.whl` 预编译包。**

**这意味着 pip 要在你机器上编译 CUDA 扩展，需要：**

```
CMake 3.26+
CUDA Toolkit（版本要和 torch 的 cu128 匹配）
MSVC（Visual Studio Build Tools）
```

**三者版本必须完全匹配，否则编译到一半失败。**

**即使编译成功，vLLM 的一些调度逻辑依赖 Linux 特性，在 Windows 上可能行为异常。**

### 那为什么还要学它

**因为面试会问。**

**而且它在 Linux 服务器上是事实标准——**
**等你到实验室用服务器，或者公司给你一台 Linux 开发机，第一件事就是装 vLLM。**

### 三条可行路径

| 路径 | 说明 |
|---|---|
| **① 用 Ollama（现在）** | Windows 能跑，本机已有模型 |
| **② WSL2 里装 vLLM** | 需要 WSL2 + CUDA 直通，配置复杂 |
| **③ 租一台 Linux GPU 云服务器** | 按小时计费，最接近真实生产环境 |

**本节走路径 ①。** 原理一样，指标一样，只是不能演示 vLLM 特有的参数。

---

## 四、Ollama 是什么

**一句话：Ollama = 模型管理器 + 推理服务。**

**它做的事：**

```
ollama pull qwen2.5:7b     ← 下载模型（自动选量化版本）
ollama run qwen2.5:7b      ← 命令行对话
ollama serve               ← 启动 HTTP 服务（默认 11434 端口）
```

**它底层用的推理实现，同样有连续批处理和 KV Cache 管理。**

**对你来说最重要的是：它提供了一个标准的 HTTP API，你的 Agent 可以直接调。**

---

## 五、启动服务

```
cd /e/Dowmload/Ollama_Dowmload
./ollama serve
```

**会看到类似输出：**

```
msg="starting ollama server"
msg="Listening on 127.0.0.1:11434"
```

**验证：**

```
curl http://127.0.0.1:11434/api/tags
```

**实际返回（本机）：**

```
qwen3-vl:8b-instruct-q4_K_M                  5.8 GB
second_constantine/deepseek-coder-v2:16b-Q4_K_M   9.7 GB
qwen2.5:7b-instruct-q4_K_M                   4.4 GB
```

**三个模型都能用。**

---

## 六、测出真实的推理指标

**这一节的核心。四个指标，决定了一个服务「好不好用」。**

| 指标 | 全称 | 含义 | 用户能感知吗 |
|---|---|---|---|
| **TTFT** | Time To First Token | 从发请求到收到第一个字 | **能，这是「卡不卡」** |
| **TPOT** | Time Per Output Token | 每个后续 token 的间隔 | 能，这是「快不快」 |
| **吞吐** | tokens/s | 每秒生成多少 token | 间接 |
| **总耗时** | —— | 整个请求完成 | 能 |

**为什么 TTFT 最重要：**

```
用户按下回车，屏幕上 0.05 秒就开始出字 → 感觉「秒回」
用户按下回车，3 秒后才开始出字，然后哗哗出完 → 感觉「卡」
```

**同样的总耗时，TTFT 小的体验好得多。所以流式输出不是为了炫技，是为了这个。**

### 实测代码

```python
import requests, json, time

URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
Q = "用一句话说明CT窗宽窗位是什么。"

# ---------- 非流式 ----------
t0 = time.time()
r = requests.post(URL, json={
    "model": MODEL,
    "messages": [{"role": "user", "content": Q}],
    "stream": False,
    "options": {"temperature": 0},
}, timeout=300)
total = time.time() - t0
d = r.json()
print(f"总耗时: {total:.2f}s")
print(f"prompt tokens: {d['prompt_eval_count']}, 生成: {d['eval_count']}")
print(f"生成速度: {d['eval_count']/(d['eval_duration']/1e9):.1f} tokens/s")

# ---------- 流式，测 TTFT ----------
t0 = time.time()
ttft = None
r = requests.post(URL, json={
    "model": MODEL,
    "messages": [{"role": "user", "content": Q}],
    "stream": True,
    "options": {"temperature": 0},
}, stream=True, timeout=300)

for line in r.iter_lines():
    if not line:
        continue
    chunk = json.loads(line)
    if ttft is None and chunk.get("message", {}).get("content"):
        ttft = time.time() - t0
    if chunk.get("done"):
        gen = chunk.get("eval_duration", 1) / 1e9
        print(f"TTFT: {ttft:.3f}s")
        print(f"总耗时: {time.time()-t0:.2f}s")
        print(f"生成速度: {chunk.get('eval_count',0)/gen:.1f} tokens/s")
        break
```

### 实测结果（本机）

**非流式（首次，含模型加载）：**

```
总耗时: 3.90s
prompt tokens: 39, 生成 tokens: 24
生成速度: 69.9 tokens/s
回答: CT窗宽窗位是一组参数，用于在CT图像中调整显示的组织范围和对比度。
```

**流式（模型已加载）：**

```
TTFT: 0.03s
总耗时: 0.36s
生成速度: 74.6 tokens/s
```

### 怎么读这两个数

**第一次 3.9 秒，第二次 0.36 秒。**

**差的主要是模型加载时间。这就是为什么生产环境必须「模型常驻内存」。**

**另一个关键：TTFT 只有 0.03 秒。**

**7B 的 Q4 模型在 8 GB 显卡上，首字延迟 30 毫秒——这是很不错的水平。**

**作为对比，如果走云端 API，网络往返通常就要 200-800 毫秒。**

**这就是本地部署的价值：延迟低一个数量级。**

---

## 七、调 Ollama 的三种方式

### 方式一：`/api/chat`（对话，推荐）

```python
r = requests.post("http://127.0.0.1:11434/api/chat", json={
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "messages": [
        {"role": "system", "content": "你是医学影像助手。"},
        {"role": "user", "content": "肝窗的窗宽窗位是多少？"},
    ],
    "stream": False,
    "options": {"temperature": 0},
})
print(r.json()["message"]["content"])
```

**和 16-1 学的 Anthropic 格式几乎一样。** 转换只需要改 `role` 和 URL。

### 方式二：`/api/generate`（纯文本补全）

```python
r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "prompt": "肝窗的窗宽窗位是",
    "stream": False,
})
```

**不给对话结构，直接续写。适合做基础测试。**

### 方式三：OpenAI 兼容接口

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="qwen2.5:7b-instruct-q4_K_M",
    messages=[{"role": "user", "content": "你好"}],
)
```

**这一条最有价值：很多框架（LangChain、RAGAS、OpenAI SDK）默认就支持 OpenAI 格式。**

**接口一兼容，这些框架不用改代码就能接本地模型。**

**这也是为什么很多人用 Ollama 做开发和评测——省钱。**

---

## 八、什么时候该本地跑，什么时候该调 API

| 场景 | 本地（Ollama） | 云端 API |
|---|---|---|
| **开发调试** | ✅ **免费、无限次** | 每次都要钱 |
| **评测跑 1000 条** | ✅ **免费** | 几十上百元 |
| 需要最强模型（GPT-4 级） | ❌ 跑不动 | ✅ |
| 生产高并发 | ❌ 单卡撑不住 | ✅ |
| 数据敏感（不出去） | ✅ **数据不出本机** | ❌ |
| 低延迟要求（<100ms） | ✅ **没有网络往返** | ⚠️ 看网络 |

**你的策略应该是：**

```
开发 + 评测  → 本地 Ollama（省钱）
关键生成     → 云端 API（要质量）
```

**这个组合能省下 90% 的钱。**

---

## 九、跟着做

### 任务一：启动服务并测出三个指标

1. 启动 `ollama serve`
2. 跑上面那段实测代码
3. **记录：TTFT、总耗时、生成速度**

**验收：**

- [ ] 三个数都有了具体值
- [ ] 能解释「第一次 3.9 秒、第二次 0.36 秒」差在哪
- [ ] **能说出 TTFT 和总耗时的区别**

### 任务二：换个模型再测一遍

**换成 `qwen3-vl:8b-instruct-q4_K_M` 跑同样的问题。**

**填表：**

| 模型 | 参数量 | TTFT | 生成速度 |
|---|---|---|---|
| qwen2.5:7b | 7B | | |
| qwen3-vl:8b | 8B | | |

**结论：大一点是不是一定慢？**

### 任务三：用 OpenAI 兼容接口调

**装 `openai` 包（约 300 KB），用 `/v1` 接口调一次。**

**验收：**

- [ ] 能调通
- [ ] **能说出为什么这个接口重要**（提示：想想 LangChain）

---

## 十、练习

### 练习 20-1-A（必做）：前缀缓存的效果

**同一个系统提示词（写长一点，比如 500 字），连发 5 次请求。**

**记录每次的 TTFT。**

**填表：**

| 次数 | TTFT |
|---|---|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |

**结论：第 1 次和之后几次比，差多少？这说明什么？**

### 练习 20-1-B（必做）：上下文长度对 TTFT 的影响

**同一句话，前面加不同长度的前缀（0 / 500 / 2000 / 8000 字）。**

**记录 TTFT。**

**结论：** 为什么 prefill 阶段的时间随输入长度线性增长？

### 练习 20-1-C（必做）：并发测试

**用 `concurrent.futures` 同时发 4 个请求，记录总耗时。**

```python
from concurrent.futures import ThreadPoolExecutor

def ask(q):
    ...

with ThreadPoolExecutor(4) as ex:
    t0 = time.time()
    list(ex.map(ask, questions))
    print(f"4 并发总耗时: {time.time()-t0:.2f}s")
```

**对比「串行发 4 个」的总耗时。**

**结论：并发的加速比是多少？为什么不是 4 倍？**

### 练习 20-1-D（选做）：算一笔账

**你的 Agent 每天跑 1000 次，每次输入 1500 token、输出 300 token。**

| 方案 | 成本 |
|---|---|
| 云端 API（按你现在的价格） | |
| 本地 Ollama | 电费 |

**算出来，看看本地能省多少。**

---

## 十一、自测题

1. 推理引擎解决了原生 PyTorch 的三个什么问题？
2. PagedAttention 解决的是什么浪费？类比成什么？
3. 连续批处理相比传统批处理，提升在哪？
4. 前缀缓存对 Agent 场景为什么特别有用？
5. vLLM 为什么在 Windows 上装不了？
6. TTFT 和 TPOT 分别是什么？哪个更影响用户体验？
7. Ollama 的 OpenAI 兼容接口为什么重要？
8. 什么情况下应该本地跑，什么情况下该调 API？

---

## 十二、出口标准

- [ ] 能说清推理引擎的三个核心优化
- [ ] 能解释 vLLM 在 Windows 的限制和替代方案
- [ ] 能在本地起服务并调通 API
- [ ] 能测出 TTFT、TPOT、吞吐
- [ ] 知道什么时候用本地、什么时候用云
- [ ] 能用 OpenAI 兼容接口接本地模型

---

## 十三、常见错误

| 错误 | 现象 | 改法 |
|---|---|---|
| 每次请求都重新加载模型 | 第一次 4 秒、之后 0.3 秒 | serve 常驻，不要每次 new |
| 不设 `temperature` | 结果不可复现 | 设 0 |
| 用非流式测 TTFT | 测不出来 | 必须 stream=True |
| 以为并发数越高越好 | 显存爆 | 按显存算并发上限 |
| 硬要在 Windows 装 vLLM | 浪费一天 | 用 Ollama |
| 混淆 TTFT 和总耗时 | 优化错方向 | 分开测 |

---

## 十四、下一节要干什么

**你知道了 PagedAttention 和连续批处理是干嘛的。**

**但它们具体怎么工作？为什么分页能省显存？连续批处理为什么能把吞吐提十倍？**

**下一节 20-2 会把这两个机制拆开算一遍。**

**算完你会真正理解「为什么 8 GB 显卡能同时服务多个用户」——这是面试的高频问题。**
