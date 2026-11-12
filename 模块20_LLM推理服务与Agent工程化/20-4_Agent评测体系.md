# 20-4 Agent 评测体系

**建议用时：10 小时**

**出口标准：能设计一套 Agent 评测方案，用本地模型当裁判跑完，并给出可汇报的数字。**

---

## 〇、环境说明

**不需要装任何东西。**

**评测用的裁判模型，就用你本地的 Ollama。**

**实测：一次裁判调用 0.6-0.7 秒，跑 100 条不到两分钟，成本 0 元。**

---

## 一、为什么 Agent 评测比模型评测难

**19-6 讲的是「单轮问答」的评测：一个问题、一个答案、一个分数。**

**Agent 不一样：**

| | 单轮问答 | Agent |
|---|---|---|
| 输出 | 一段文字 | **一串动作 + 最终结果** |
| 对错 | 答案对不对 | **每一步都对吗？最终达没达成目标？** |
| 路径 | 唯一 | **可以有多种正确路径** |
| 失败 | 答错 | 答错 / 死循环 / 工具乱调 / 中途放弃 |

### 一个例子

**任务：「这个病人 AFP 45，帮我判断风险并给建议」**

**Agent A：**
```
查 AFP 标准 → 查分期 → 查治疗 → 给出完整建议     ✅ 成功
```

**Agent B：**
```
查 AFP 标准 → 又查了一次 AFP → 又查了一次 → 超步数退出   ❌ 死循环
```

**Agent C：**
```
没查任何工具，直接凭训练知识回答               ⚠️ 答案是「对」的，但方式错了
```

**C 最麻烦：结果看起来对，但过程不可靠。换个病人可能就错。**

**所以 Agent 评测必须评「过程」和「结果」两件事。**

---

## 二、四层评测

```
① 组件层  —— 每个工具单独测
② 轨迹层  —— 一次任务里调了哪些工具、顺序对不对
③ 结果层  —— 最终答案对不对
④ 端到端  —— 用户能不能用（延迟、成本、稳定性）
```

### ① 组件层：工具能不能用

**先测最基础的。工具都错了，后面全白搭。**

```python
def test_tools():
    cases = [
        ("get_hu_value", {"organ": "肝"}, lambda r: "error" not in r),
        ("get_hu_value", {"organ": "不存在的器官"}, lambda r: "error" in r),
        ("calc_fib4", {"age": 52, "ast": 45, "plt": 120, "alt": 38},
         lambda r: 0 < r["fib4"] < 100),
    ]
    passed = 0
    for name, args, check in cases:
        r = TOOL_IMPL[name](**args)
        if check(r):
            passed += 1
    print(f"工具测试: {passed}/{len(cases)}")
```

**这一层应该是 100% 通过。不到就不往下测。**

### ② 轨迹层：过程对不对

**定义「理想轨迹」，然后比对。**

```python
TRAJECTORY_TESTS = [
    {
        "task": "AFP 45 的风险判断",
        "must_call": ["get_hu_value", "search_guideline"],
        "must_not_call": [],
        "max_steps": 6,
    },
    {
        "task": "随便聊聊",
        "must_call": [],
        "must_not_call": ["search_guideline"],
        "max_steps": 2,
    },
]

def check_trajectory(trace, expected):
    """trace 是 16-5 里记录的工具调用列表"""
    called = [t["tool"] for t in trace]
    issues = []

    for t in expected["must_call"]:
        if t not in called:
            issues.append(f"应该调用 {t} 但没调用")

    for t in expected["must_not_call"]:
        if t in called:
            issues.append(f"不应该调用 {t} 但调用了")

    if len(trace) > expected["max_steps"]:
        issues.append(f"步数超限: {len(trace)} > {expected['max_steps']}")

    return issues
```

**这一层能抓住「答案对但过程错」的情况。**

### ③ 结果层：答案对不对

**三种方法，成本从低到高：**

| 方法 | 成本 | 准确度 | 什么时候用 |
|---|---|---|---|
| **精确匹配** | 0 | 低 | 有唯一答案（如数字） |
| **关键词检查** | 0 | 中 | 关键信息必须出现 |
| **LLM 裁判** | 低（本地免费） | **高** | 开放式回答 |

### ④ 端到端：能不能用

```python
def measure_e2e(agent, test_set):
    latencies = []
    for t in test_set:
        t0 = time.time()
        agent.run(t["task"])
        latencies.append(time.time() - t0)
    latencies.sort()
    return {
        "P50": latencies[len(latencies)//2],
        "P95": latencies[int(len(latencies)*0.95)],
        "最大": latencies[-1],
    }
```

**注意：Agent 的延迟是「多轮之和」，通常比单次调用大 5-10 倍。**

---

## 三、任务成功率怎么定义

**这是最容易被问倒的问题：「你的 Agent 成功率多少？」**

**「成功率」必须定义清楚，否则毫无意义。**

### 三种定义，严格程度递增

| 定义 | 标准 | 严格度 |
|---|---|---|
| **宽松** | 最终答案包含关键信息 | 低 |
| **中等** | 答案正确 + 没有死循环 | 中 |
| **严格** | 答案正确 + 轨迹合理 + 无多余调用 | 高 |

**建议：三个都报。**

```
任务成功率（宽松）：85%
任务成功率（中等）：78%
任务成功率（严格）：62%
```

**只报一个数会被追问「你怎么定义的」，报三个反而显得专业。**

### 代码

```python
def evaluate_agent(agent, test_cases):
    results = {"宽松": 0, "中等": 0, "严格": 0}
    n = len(test_cases)

    for case in test_cases:
        answer, trace = agent.run(case["task"])

        ans_ok = case["check"](answer)                     # 答案对不对
        no_loop = len(trace) <= case.get("max_steps", 8)   # 没死循环
        traj_ok = not check_trajectory(trace, case)        # 轨迹合理

        if ans_ok:
            results["宽松"] += 1
        if ans_ok and no_loop:
            results["中等"] += 1
        if ans_ok and no_loop and traj_ok:
            results["严格"] += 1

    return {k: f"{v}/{n} ({v/n*100:.0f}%)" for k, v in results.items()}
```

---

## 四、LLM-as-Judge：用本地模型当裁判

**这是本节最实用的部分。**

### 实测：这条路走得通吗

**我在这台机器上跑了三个案例，裁判用 `qwen2.5:7b-instruct-q4_K_M`：**

**案例 1（正确答案）：**

```
回答: 肝窗常用窗宽 160 HU、窗位 60 HU。
→ 正确性 5/5  完整性 5/5  耗时 0.7s
→ 理由: 回答准确且完整覆盖了参考答案的要点
```

**案例 2（数值错误）：**

```
回答: 肝窗的窗宽是 1600，窗位是 600。
→ 正确性 2/5  完整性 3/5  耗时 0.6s
→ 理由: 数值错误，但提及了窗宽和窗位
```

**案例 3（回避问题）：**

```
回答: 窗宽窗位是一组用于调整CT显示对比度的参数。
→ 正确性 2/5  完整性 1/5  耗时 0.7s
→ 理由: 回答缺少具体数值，未能准确描述肝窗的窗宽和窗位
```

**结论：**

- **评分合理**，三个案例的区分度明显
- **理由准确**，能指出「数值错误」「缺少具体数值」
- **速度快**，0.6-0.7 秒一次
- **零成本**，本地跑

**100 条评测不到两分钟，一分钱不花。**

### 代码

```python
import re, json, requests

JUDGE_PROMPT = """你是医学问答质量评估员。根据参考答案给模型回答打分。

问题：{q}
参考答案：{ref}
模型回答：{ans}

从两个维度打分（1-5）：
- 正确性：医学事实是否准确
- 完整性：是否覆盖了参考答案的要点

只输出 JSON：{{"correctness": 数字, "completeness": 数字, "reason": "一句话"}}
"""

def judge(q, ref, ans, model="qwen2.5:7b-instruct-q4_K_M"):
    r = requests.post("http://127.0.0.1:11434/api/chat", json={
        "model": model,
        "messages": [{"role": "user",
                      "content": JUDGE_PROMPT.format(q=q, ref=ref, ans=ans)}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }, timeout=300)

    txt = r.json()["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None
```

### 三个必须注意的坑

**坑一：裁判也会被长度骗**

**长答案容易得高分。** 所以：

```
① 提示词里明确「简洁不是缺点，啰嗦不是优点」
② 或者做长度归一化
```

**坑二：裁判有位置偏好**

**判断「A 和 B 哪个好」时，倾向选第一个。**

**对策：交换位置跑两次，只认两次一致的结论。**

```python
def judge_pairwise(q, a, b):
    r1 = judge_compare(q, a, b)      # A 在前
    r2 = judge_compare(q, b, a)      # B 在前
    if r1 == "A" and r2 == "B":
        return "A"
    if r1 == "B" and r2 == "A":
        return "B"
    return "平局"                     # 两次不一致 = 有位置偏好，不采信
```

**坑三：同源模型会偏袒自己**

**如果你用的裁判模型和被评测的是同一个，它会给自己高分。**

**对策：尽量用不同模型做裁判。** 比如被测是 qwen2.5:7b，裁判用 deepseek-coder-v2。

---

## 五、RAGAS 的四个指标（了解）

**RAGAS 是一个专门评测 RAG 系统的框架，它的四个指标值得知道。**

| 指标 | 衡量什么 | 怎么算 |
|---|---|---|
| **Faithfulness（忠实度）** | 答案里的每条断言，能不能在检索到的资料里找到 | 拆成断言 → 逐条核对 |
| **Answer Relevancy（相关性）** | 答案和问题的相关程度 | 用答案反推问题，看像不像 |
| **Context Precision（上下文精确度）** | 检索到的资料里，有用的排前面吗 | 相关性加权 |
| **Context Recall（上下文召回）** | 该找到的资料找到了吗 | 和标准答案比对 |

**这四个指标覆盖了 RAG 的两个环节：**

```
检索 → Context Precision / Context Recall
生成 → Faithfulness / Answer Relevancy
```

**为什么值得知道：面试问「你怎么评估 RAG」，答这四个指标+它们的含义，比说「看准确率」专业得多。**

**不用真装 RAGAS**，它依赖较重，而且和现有环境可能冲突。**理解指标定义，自己实现简化版就行。**

---

## 六、回归测试：防止改坏了

**这是工程上最重要、学生最容易忽略的一环。**

### 问题

**你为了修一个问题改了提示词，结果另外五个问题坏了。**

**没有回归测试，你根本不知道。**

### 做法

**把评测集固化成测试，每次改动都跑一遍。**

```python
# test_agent_regression.py
import pytest

GOLDEN_SET = [
    {"task": "...", "check": lambda a: "160" in a and "60" in a},
    # ... 30 条
]

@pytest.mark.parametrize("case", GOLDEN_SET)
def test_golden(case):
    answer, trace = agent.run(case["task"])
    assert case["check"](answer), f"回归失败: {case['task']}\n回答: {answer}"
```

**跑：**

```
pytest test_agent_regression.py -v
```

### 门禁（CI/CD Gating）

**JD 里出现的词：「CI/CD gating」「回归门禁」。**

**意思很简单：**

```
代码提交 → 自动跑评测 → 成功率低于阈值就阻止合并
```

```yaml
# .github/workflows/eval.yml 的示意
- name: 跑 Agent 评测
  run: pytest test_agent_regression.py

- name: 检查成功率
  run: |
    python check_success_rate.py --min 0.85
```

**为什么重要：**

```
没有门禁 → 提示词改一个字，线上效果掉 10%，没人知道
有门禁   → 提交时就被拦住
```

---

## 七、跟着做

### 任务一：建一个 20 条的评测集

**从你的医学问答数据里挑 20 条，每条包含：**

```python
{
    "task": "问题",
    "reference": "标准答案",
    "must_call": ["工具名"],        # 应该调什么
    "max_steps": 5,
}
```

**要求覆盖三类：**

| 类型 | 条数 | 目的 |
|---|---|---|
| 正常问题 | 12 | 测基本能力 |
| **应该拒绝的** | 4 | 知识库没有的，看会不会编 |
| **边界情况** | 4 | 多工具、长上下文 |

### 任务二：用本地裁判跑一遍

**用第四节的 `judge` 函数，把 20 条跑一遍。**

**统计：**

| 指标 | 值 |
|---|---|
| 平均正确性 | |
| 平均完整性 | |
| 正确性 ≥4 的比例 | |
| 总耗时 | |

**验收：**

- [ ] 20 条都跑完
- [ ] 四个数都有
- [ ] **跑完的总耗时不超过 1 分钟**
- [ ] 能说出「哪几条最差，为什么」

### 任务三：算三个成功率的版本

**按第三节的定义，算出宽松/中等/严格三个数。**

**验收：** 三个数都有，并且能说清它们的区别。

---

## 八、练习

### 练习 20-4-A（必做）：验证裁判的可靠性

**人工给 20 条里挑 5 条打分（1-5），然后和裁判的分数对比。**

**填表：**

| 案例 | 人工分 | 裁判分 | 差多少 |
|---|---|---|---|
| 1 | | | |
| ... | | | |

**结论：裁判和人工的相关系数大概多少？差在哪类案例上？**

**这个实验能让你在面试时说：「我用本地模型做裁判，人工核对后一致性是 XX%」——这是很硬的话。**

### 练习 20-4-B（必做）：位置偏好的验证

**拿两段质量相近的回答，做两次成对比较（交换顺序）。**

**记录两次结果。**

**如果两次选的是不同位置（都是第一个），说明有位置偏好。**

### 练习 20-4-C（必做）：设计一个「必须拒绝」的测试

**准备 5 个知识库里没有答案的问题。**

**测你的 Agent：**

```
正确行为：明确说「资料中未找到」
错误行为：编一个答案
```

**算拒绝率。这个数应该接近 100%。**

### 练习 20-4-D（选做）：做成 pytest

**把 20 条评测集写成 pytest 测试。**

**跑一次，看通过率。**

**然后故意把系统提示改坏一点（比如删掉「不知道就说不知道」那一条），再跑，看通过率掉多少。**

**这一题让你体会「门禁」的价值。**

---

## 九、自测题

1. Agent 评测和单轮问答评测的三个区别？
2. 四层评测分别测什么？
3. 「答案对但过程错」为什么更危险？
4. 任务成功率有哪三种定义？为什么要都报？
5. LLM 当裁判有哪三个已知问题？
6. 为什么同源模型当裁判会偏袒自己？
7. RAGAS 的四个指标分别衡量什么？
8. 回归门禁解决什么问题？

---

## 十、出口标准

- [ ] 能设计带标准答案的评测集
- [ ] 能测出任务成功率的三个版本
- [ ] 能用本地模型当裁判跑完评测
- [ ] 知道裁判的三个坑和对策
- [ ] 能说出 RAGAS 四个指标的含义
- [ ] 能把评测固化成 pytest 做回归

---

## 十一、常见错误

| 错误 | 现象 | 改法 |
|---|---|---|
| 只测答案不测轨迹 | 掩盖了死循环 | 加轨迹检查 |
| 只报一个成功率 | 被追问「怎么定义」 | 报三个版本 |
| 裁判不给 temperature=0 | 结果不可复现 | 设 0 |
| 用同一个模型做裁判 | 自己给自己打高分 | 换模型 |
| 不做人工核对 | 不知道裁判准不准 | 抽 20% 对比 |
| 没有回归测试 | 改一处坏五处 | 做成 pytest |
| 评测集太少 | 数字不可信 | 至少 20 条 |

---

## 十二、模块 20 学完了

**你现在应该有的能力：**

```
解释推理引擎快在哪              （20-1）
算出显存和并发的上限             （20-2）
管理 Agent 的上下文预算          （20-3）
设计并跑通一套评测               （20-4）
```

**下一步：串成毕业项目。**

**见 `项目_生产级医学Agent服务.md`。**
