# 16-3 工具调用 Function Calling

**建议用时：8 小时**

**出口标准：能不用任何框架，手写一个完整的工具调用循环，让模型调用你自己的 Python 函数。**

---

## 〇、这一节在解决什么

上一节你能让模型**按格式说话**了。

但它还是只会说话。你问「今天天气怎么样」，它只能根据训练数据猜，或者编。

**这一节要让它「动手」——调你的函数，查你的数据，算你的数字。**

原理很简单，一句话：

> **模型不会调用任何函数。它只会输出一段 JSON，说「我想调 get_weather，参数是 {'city': '上海'}」。真正去调用函数的，是你的代码。**

**想清楚这一点，工具调用就没有神秘感了。**

---

## 一、拆开看：一次工具调用发生了什么

### 完整链路

```
你的代码                        模型                        你的代码
   │                             │                             │
   │ 1. 发消息 + 工具清单  ──────▶│                             │
   │                             │                             │
   │                             │ 2. 判断需要工具              │
   │                             │    输出 tool_use 块           │
   │ 3. 收到 tool_use      ◀──────│                             │
   │                             │                             │
   │ 4. 执行 get_hu_window()  ────┼────────────────────────────▶│
   │                             │                             │
   │ 5. 把结果发回去        ──────▶│                             │
   │                             │                             │
   │                             │ 6. 看懂了，组织答案           │
   │ 7. 收到 text 回答     ◀──────│                             │
```

**模型参与的是第 1、2、6 步。第 4 步是你的代码干的。**

### 我实跑的返回长什么样

**第一轮请求（带工具清单）：**

```python
d = requests.post(..., json={
    "model": "deepseek-flash",
    "max_tokens": 1500,
    "tools": TOOLS,
    "messages": [{"role": "user", "content": "我想看肺部的CT，应该用什么窗宽窗位？"}],
}).json()
```

**实际返回：**

```
状态码: 200   stop_reason: tool_use
块类型: ['thinking', 'tool_use']
>>> 工具调用: get_hu_window  参数: {'organ': '肺'}  id: call_00_hzp5q9Q0lzdSdfFMy6HN1306
用量: input_tokens=324, output_tokens=58
```

**三个关键点：**

1. **`stop_reason` 是 `tool_use`，不是 `end_turn`** —— 这说明它没说完，在等你干活
2. **`content` 里有一个 `tool_use` 块**，带 `name`、`input`、`id`
3. **`message` 里**——它把工具调用的参数填对了，`{'organ': '肺'}`

**注意：此刻什么都没发生。没有函数被调用。** 只是模型告诉你它想调什么。

---

## 二、工具的 JSON Schema

**你要告诉模型「我有哪些工具」，格式是 JSON Schema。**

### 最小可用模板

```python
TOOLS = [
    {
        "name": "get_hu_window",                    # 函数名，英文，蛇形
        "description": "查询某个器官的CT窗宽窗位",   # ← 最重要的一行
        "input_schema": {
            "type": "object",
            "properties": {
                "organ": {
                    "type": "string",
                    "description": "器官名，如 肺、肝、骨",
                }
            },
            "required": ["organ"],
        },
    }
]
```

### 三个字段的作用

| 字段 | 作用 | 写错的后果 |
|---|---|---|
| `name` | 模型用什么名字点你的函数 | 和你的 Python 函数名对不上，你的分发逻辑找不到 |
| `description` | **模型唯一能看到的说明书** | 写得太含糊，模型该调的时候不调，不该调的时候乱调 |
| `input_schema` | 参数的类型和含义 | 类型写错，模型传过来的参数类型也不对 |

### `description` 怎么写

**这是整节最值得花时间的地方。** 模型全靠这一句话决定「要不要用这个工具」。

**差的写法：**

```
"获取窗宽窗位"
```

**好的写法：**

```
"查询某个器官或组织的标准CT窗宽窗位设置。
适用于用户问「看X应该用多大窗宽」「X的窗宽窗位是多少」。
输入器官的中文名，如：肺、肝、骨、脑、纵隔。
不适用于查询Hounsfield单位系数本身。"
```

**好的 description 包含三件事：**

1. **干什么**
2. **什么时候用**（正面清单）
3. **什么时候不用**（负面清单）

**负面清单尤其容易漏。** 你有两个相似工具时，不写「什么时候别用」，模型会随机挑一个。

---

## 三、完整循环：手写一遍

**这一节的核心，不看框架，纯手写。**

```python
import os, json, requests

URL = "https://api.deepseek.com/anthropic/v1/messages"
HEADERS = {
    "x-api-key": os.environ["DEEPSEEK_API_KEY"],
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# ---------- 工具实现 ----------
WINDOWS = {
    "肺":   (1500, -600),
    "肝":   (160,   60),
    "骨":   (2000,  400),
    "脑":   (80,    40),
}

def get_hu_window(organ: str) -> dict:
    if organ in WINDOWS:
        w, l = WINDOWS[organ]
        return {"organ": organ, "window_width": w, "window_level": l}
    return {"error": f"未知器官 {organ}"}

# ---------- 工具声明 ----------
TOOLS = [{
    "name": "get_hu_window",
    "description": "查询器官的CT窗宽窗位。用户问看某器官该用什么窗时使用。",
    "input_schema": {
        "type": "object",
        "properties": {"organ": {"type": "string", "description": "器官中文名"}},
        "required": ["organ"],
    },
}]

TOOL_IMPL = {"get_hu_window": get_hu_window}     # 名字 -> 真函数

# ---------- 主循环 ----------
def chat_with_tools(question, max_steps=5):
    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-flash",
            "max_tokens": 1500,
            "tools": TOOLS,
            "messages": messages,
        }, timeout=180).json()

        messages.append({"role": "assistant", "content": resp["content"]})

        # 不是要工具，就是最终答案
        if resp.get("stop_reason") != "tool_use":
            return "".join(b["text"] for b in resp["content"]
                           if b["type"] == "text")

        # 执行模型要的所有工具
        results = []
        for block in resp["content"]:
            if block["type"] != "tool_use":
                continue
            fn = TOOL_IMPL.get(block["name"])
            if fn is None:
                out = {"error": f"工具 {block['name']} 不存在"}
            else:
                try:
                    out = fn(**block["input"])       # ← 真正的执行在这里
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}

            results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],          # ← 必须回传同一个 id
                "content": json.dumps(out, ensure_ascii=False),
            })

        messages.append({"role": "user", "content": results})

    return "达到最大步数，未完成。"
```

### 我刚才实跑的结果

```
--- 第1轮 stop_reason=tool_use ---
  调用: get_hu_window {'organ': '肺'} -> {'organ': '肺', 'window_width': 1500, 'window_level': -600}
--- 第2轮 stop_reason=end_turn ---
最终回答: 观察肺部通常使用**肺窗（Lung Window）**，推荐参数为：
- **窗宽（WW，Window Width）：1500 HU**
- **窗位（WL，Window Level）：-600 HU**
...
```

**两轮就结束了。** 第一轮它要工具，第二轮它拿着结果说话。

---

## 四、四个必须理解的设计细节

### ① 为什么必须回传 `tool_use_id`

```python
"tool_use_id": block["id"]
```

**模型可能一次要调多个工具。** 没有 id，它不知道哪个结果对应哪个请求。

**如果 id 对不上，会报错或者结果错乱。** 这是最常见的实现 bug。

### ② 为什么 `messages` 要追加 assistant 那一条

```python
messages.append({"role": "assistant", "content": resp["content"]})
```

**这是把「我说过什么」记进历史。** 不追加的话，第二轮模型不知道它刚才要求过工具，会重新问一遍。

**注意：`content` 直接用返回的列表**（里面含 thinking 和 tool_use 块），不用自己拼。

### ③ 为什么 tool_result 的 `role` 是 `user`

**这是很多人第一次看会愣住的地方。**

Anthropic 的协议里，工具结果是用 `role: "user"` 发回去的。

原因：**从模型视角看，这就是「外部世界给了它一个输入」。** 谁给的抽象成「用户」。

**别被这个绊住，照写就对了。**

### ④ 为什么要有 `max_steps`

```python
for step in range(max_steps):
```

**因为模型会陷入循环。**

真实会发生的场景：

```
模型：我要调 search("肝癌")
结果：没找到
模型：我要调 search("肝癌")      ← 一模一样
结果：没找到
...
```

**没有步数上限，这个循环会跑到你把钱烧光。**

**业界叫这个「死循环」，是 Agent 三大杀手之一**（另外两个是上下文爆炸和工具失败级联）。

---

## 五、多工具与选择

### 加第二个工具

```python
TOOLS = [
    {
        "name": "get_hu_window",
        "description": "查询器官的CT窗宽窗位。用户问'看X用什么窗'时使用。"
                       "不适用于查询组织的HU值本身。",
        ...
    },
    {
        "name": "get_hu_value",
        "description": "查询某组织的典型HU值范围。用户问'X的CT值是多少'时使用。"
                       "不适用于查询窗宽窗位设置。",
        ...
    },
]
```

**两条 description 里各有一句「不适用于」。这不是啰嗦，是必需的区分。**

### 模型在一次返回里要多个工具

```python
for block in resp["content"]:
    if block["type"] == "tool_use":
        ...
```

**返回的 `content` 里可能有好几个 `tool_use` 块。** 上面那个 `for` 循环会全部执行，然后把所有结果一起发回去。

**不要假设一次只调一个。**

---

## 六、错误处理：工具一定会失败

**这是新手最容易忽略的一节。**

### 三种错误，三种处理

**① 工具名不存在**

```python
fn = TOOL_IMPL.get(block["name"])
if fn is None:
    out = {"error": f"工具 {block['name']} 不存在"}
```

**不要直接 KeyError 崩掉。** 把错误当成 tool_result 发回去，模型有机会改。

**② 参数不对**

```python
try:
    out = fn(**block["input"])
except TypeError as e:
    out = {"error": f"参数错误: {e}"}
```

**模型可能传 `{"organ_name": "肺"}` 而你的函数要 `organ`。** 把错误发回去，它下一轮往往就改对了。

**③ 函数内部抛异常**

```python
except Exception as e:
    out = {"error": f"{type(e).__name__}: {e}"}
```

**永远不要在工具里抛异常给外面。** 捕获它，序列化，发回去。

### 为什么这样有效

**模型看到 `{"error": "未知器官 心"}`，下一轮会自己换个说法或者告诉你查不到。**

**它不会因为一次失败就放弃——只要你把错误原样告诉它。**

---

## 七、安全边界（这一节对你尤其重要）

**工具调用意味着：模型说的话，会被你的代码执行。**

### 三条铁律

**① 永远不要给模型一个「执行任意代码」的工具**

```python
# 绝对不要
@tool
def run_shell(command: str) -> str:
    return subprocess.check_output(command, shell=True)
```

**理由**：模型被提示词注入，或者自己编一个命令，你的机器就完了。

**这一条不是理论风险。** 你之前让我写的 `study_mcp.py` 里专门有一个 `FORBIDDEN` 正则拦截删除操作，就是这个道理。

**② 工具的参数必须校验**

```python
def get_patient_data(patient_id: str) -> dict:
    if not re.fullmatch(r"P\d{6}", patient_id):
        return {"error": "patient_id 格式必须是 P + 6 位数字"}
    ...
```

**模型可能传 `"1; DROP TABLE patients"` 这种东西。**

**③ 有副作用的工具，要人工确认**

| 工具类型 | 例子 | 要确认吗 |
|---|---|---|
| 只读查询 | 查窗宽、查指南 | 不用 |
| 有副作用 | 写文件、发邮件、删数据 | **必须** |

**16-6 会讲 LangGraph 怎么在中间插入「人工确认」节点。**

---

## 八、跟着做

### 任务：给你自己的专业加工具

**要求：**

1. 复制第三节的完整代码
2. 加**两个**你自己领域的工具，建议从下面挑：
   - `get_hu_value(organ)`：查组织的典型 HU 值
   - `calc_fib4(age, ast, plt, alt)`：算 FIB-4
   - `lookup_guideline(keyword)`：在本地一个 txt 里查关键词
3. 每条 description 都要有「什么时候用」和「什么时候别用」
4. 测试三个问题：
   - 明确该用工具 1 的
   - 明确该用工具 2 的
   - **两个都不该用的**（看它会不会乱调）

**验收：**

- [ ] 用 `print(messages)` 打印完整对话，能看懂每一轮在传什么
- [ ] 第三个问题（不该用工具）时，`stop_reason` 是 `end_turn` 而不是 `tool_use`
- [ ] 故意传一个不存在的器官，看模型拿到 error 后怎么处理

**第三项是重点。** 如果它在本该直接回答时乱调工具，说明你的 description 写得不够明确。

---

## 九、练习

### 练习 16-3-A（必做）：打印并读懂完整对话

在循环里加一行：

```python
print(f"--- 第{step+1}轮 ---")
print(json.dumps(messages, ensure_ascii=False, indent=2)[:2000])
```

**跑一遍工具调用，把输出的对话历史截图保存。**

**然后回答：第二轮发给模型的消息里，一共有几条？分别是什么角色？**

### 练习 16-3-B（必做）：制造一次工具失败

让工具故意抛异常：

```python
def get_hu_window(organ):
    raise ValueError("数据库连接失败")
```

**观察：模型收到 error 之后，是重试、换说法、还是直接告诉你查不到？**

**把它的完整回答记下来。** 这个行为决定了你的 Agent 在真实环境里的健壮程度。

### 练习 16-3-C（必做）：故意写坏 description

把工具的 description 改成：

```
"查询"
```

**然后问一个明确该用工具的问题，看它还会不会调。**

**再问一个不该用的，看它会不会乱调。**

**结论写一句话：description 的质量，直接决定了什么？**

### 练习 16-3-D（选做）：多工具并行

让两个工具在一次返回里同时被调用（比如问「肺和肝分别用什么窗」）。

**观察 `content` 里有几个 `tool_use` 块，你的循环能不能都处理。**

---

## 十、自测题

1. 模型真的会调用你的函数吗？如果不是，谁在调用？
2. `stop_reason` 是 `tool_use` 和 `end_turn` 分别意味着什么？
3. `tool_use_id` 为什么必须原样回传？
4. 为什么工具结果要用 `role: "user"` 发回去？
5. `max_steps` 是干什么的？不设会怎样？
6. 工具的 `description` 里必须写哪三件事？
7. 工具抛异常了该怎么处理？为什么不能让它抛到外面？
8. 为什么绝对不能给模型一个「执行任意 shell 命令」的工具？

---

## 十一、出口标准

- [ ] 能手写完整的两轮工具调用循环，不查资料
- [ ] 能写出一条合格的 description（含正面和负面清单）
- [ ] 知道 `tool_use_id`、`role: user` 这两个反直觉的点并解释为什么
- [ ] 能处理三种工具失败，并把错误发回去
- [ ] 知道死循环怎么防
- [ ] 能说出工具调用的三条安全铁律

---

## 十二、常见错误

| 错误 | 表现 | 原因 | 改法 |
|---|---|---|---|
| 以为模型在执行函数 | 等着它自动调 | 理解错了 | 执行是你代码的事 |
| `tool_use_id` 乱填 | 报错或结果错位 | 没理解它的作用 | 原样回传 `block["id"]` |
| tool_result 用 `role: "assistant"` | 协议错误 | 直觉反了 | 必须用 `user` |
| 不追加 assistant 消息 | 模型重复要工具 | 历史缺了 | 每轮都 append |
| 不设 max_steps | 烧钱死循环 | 忘了 | 加步数上限 |
| description 写「查询」 | 该调不调、乱调 | 信息量太低 | 写清干什么、何时用、何时别用 |
| 工具异常没捕获 | 整个程序崩 | 没写 try | 捕获并序列化发回 |
| 给模型 shell 工具 | 机器被搞 | 图方便 | 永远不要 |

---

## 十三、下一节要干什么

你现在能让模型**动手**了。但工具只能查你**手动整理好**的数据。

**真实场景里，答案在一堆 PDF、指南、病历里，你不可能每条都写成函数。**

下一节 16-4 要解决：**怎么让模型在你没有整理过的文档里找答案。**

这就是 RAG。
