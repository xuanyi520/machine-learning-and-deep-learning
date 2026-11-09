# -*- coding: utf-8 -*-
"""
学习助手 MCP 服务

给 Claude Code 提供直接操作学习环境的能力：
- 在 ai 环境里跑 Python 代码（不用反复贴代码给你复制）
- 查进度欠账、查学习目录状态
- 检索 d2l-zh 中文教材、查 MedMNIST 官方基线
- 每日反馈骨架生成

启动方式由 Claude Code 配置，手动测试：
    E:\\Dowmload\\Anaconda3_Dowmload\\python.exe E:\\AI学习\\mcp\\study_mcp.py
"""
import os
import re
import sys
import json
import subprocess
import tempfile
from pathlib import Path

from mcp.server.mcpserver import MCPServer

# ---------------- 路径常量 ----------------
STUDY_ROOT = Path(r"E:\AI学习")
AI_PYTHON = Path(r"E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe")
NOTEBOOK = STUDY_ROOT / "练习代码" / "Learn.ipynb"
PROGRESS = STUDY_ROOT / "进度追踪.md"
FEEDBACK_DIR = STUDY_ROOT / "每日反馈"
D2L = STUDY_ROOT / "参考仓库" / "d2l-zh-master"
MEDMNIST_REPO = STUDY_ROOT / "参考仓库" / "MedMNIST-main"

mcp = MCPServer(
    "study-assistant",
    instructions=(
        "医学影像 AI 学习助手。所有输出用中文。"
        "回答涉及本机环境的问题前，先调用工具核实，不要凭记忆说。"
    ),
)

# 禁止在本服务中执行的危险操作（防止误删学习材料）
FORBIDDEN = re.compile(
    r"(rm\s+-rf?\s+/|rmtree\s*\(\s*['\"]?[A-Za-z]:|shutil\.rmtree|os\.remove\s*\(|"
    r"del\s+/[fsq]|format\s+[a-z]:)",
    re.IGNORECASE,
)


# ============ 1. 代码执行 ============
@mcp.tool()
def run_python(code: str, timeout: int = 300) -> str:
    """在 ai 环境（torch 2.9.1 + CUDA）里执行 Python 代码，返回标准输出。

    用途：验证讲义里的代码到底能不能跑、算参数量、查形状、跑小实验。
    参数 code 必须是完整可执行的 Python 片段；timeout 单位秒。
    注意：这是同步执行，跑训练循环要自己控制 epoch 数，别超时。
    """
    if FORBIDDEN.search(code):
        return "拒绝执行：代码里含删除类操作，学习环境不允许。"

    # 用 -I 隔离，避免当前目录的杂散模块被导入
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8", dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(
            [str(AI_PYTHON), "-I", tmp],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(STUDY_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        parts = []
        if out:
            parts.append(f"【标准输出】\n{out[-4000:]}")
        if err:
            parts.append(f"【错误输出】\n{err[-2500:]}")
        if not parts:
            parts.append("执行完成，无输出。")
        parts.append(f"【退出码】{r.returncode}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"超时（{timeout} 秒）：训练循环请减少 epoch 或 batch 数。"
    except Exception as e:
        return f"执行失败：{type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ============ 2. 环境体检 ============
@mcp.tool()
def check_env() -> str:
    """检查 ai 环境的 GPU、PyTorch、关键库版本，确认能否直接跑深度学习代码。"""
    code = (
        "import importlib.metadata as md, torch\n"
        "print('torch', torch.__version__)\n"
        "print('cuda_available', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print('gpu', torch.cuda.get_device_name(0))\n"
        "    print('capability', torch.cuda.get_device_capability(0))\n"
        "for p in ['torchvision','monai','medmnist','scikit-learn','timm',\n"
        "          'albumentations','opencv-python-headless','pydicom','SimpleITK',\n"
        "          'numpy','pandas','matplotlib','nibabel']:\n"
        "    try: print(p, md.version(p))\n"
        "    except Exception: print(p, 'NOT_INSTALLED')\n"
    )
    return run_python(code, timeout=90)


# ============ 3. 欠账清单 ============
@mcp.tool()
def list_debts() -> str:
    """读取 进度追踪.md 的「欠账清单」表格，返回尚未完成的条目。"""
    if not PROGRESS.exists():
        return f"找不到 {PROGRESS}"
    text = PROGRESS.read_text(encoding="utf-8")
    m = re.search(r"##\s*⚠?\s*欠账清单.*?\n(.*?)(?=\n\*\*\*|\n##\s)", text, re.S)
    if not m:
        return "进度追踪.md 里没找到「欠账清单」小节。"
    rows = [l for l in m.group(1).splitlines() if l.strip().startswith("|")]
    debts = []
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        if len(cells) >= 5 and cells[0].isdigit():
            status = cells[-1]
            if "☐" in status or (not status.startswith("☑") and "x" not in status.lower()):
                debts.append(f"[{cells[0]}] {cells[1]}　起始 {cells[2]}　挂了 {cells[3]}")
    if not debts:
        return "欠账清单已全部清零。"
    return "未完成欠账：\n" + "\n".join(debts)


# ============ 4. 教材检索 ============
@mcp.tool()
def search_d2l(keyword: str, limit: int = 8) -> str:
    """在 d2l-zh 中文教材（82k 星的《动手学深度学习》）里检索关键词，返回章节与命中片段。

    用途：交叉核对讲义内容、找标准讲法、补我没讲到的部分。
    """
    if not D2L.exists():
        return f"教材目录不存在：{D2L}"
    hits = []
    for md in D2L.rglob("*.md"):
        if ".git" in str(md):
            continue
        try:
            txt = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if keyword not in txt:
            continue
        idx = txt.find(keyword)
        # 定位所在小标题
        head = ""
        for hm in re.finditer(r"^#{1,3}\s+(.+)$", txt[:idx], re.M):
            head = hm.group(1).strip()
        snippet = txt[max(0, idx - 120): idx + 260].replace("\n", " ")
        hits.append((md.parent.name, head, snippet))
        if len(hits) >= limit:
            break
    if not hits:
        return f"「{keyword}」在 d2l-zh 里没检索到。"
    out = [f"「{keyword}」命中 {len(hits)} 处："]
    for i, (chap, head, sn) in enumerate(hits, 1):
        out.append(f"\n{i}. {chap} / {head}\n   {sn[:300]}")
    return "\n".join(out)


# ============ 5. MedMNIST 官方评测 ============
@mcp.tool()
def medmnist_baseline(dataset: str = "pneumoniamnist") -> str:
    """查 MedMNIST 官方基线成绩与数据集档案（类别数、样本数、任务类型）。

    用途：给用户的成绩找外部参照。例如用户训练出 0.9504 准确率，
    要知道官方 ResNet 是多少，才知道这个成绩处在什么水平。
    """
    code = (
        "from medmnist import INFO\n"
        f"info = INFO['{dataset}']\n"
        "for k in ['python_class','task','label','n_channels','n_samples']:\n"
        "    print(k, '=', info.get(k))\n"
    )
    base = run_python(code, timeout=90)
    # 官方 baseline 表在仓库 README 或论文里，先看本地的
    extra = []
    if MEDMNIST_REPO.exists():
        for f in MEDMNIST_REPO.rglob("*.md"):
            t = f.read_text(encoding="utf-8", errors="ignore")
            if dataset.rstrip("nist").lower() in t.lower() and "ACC" in t:
                extra.append(f"本地文档含基线线索：{f.name}")
                break
    if extra:
        base += "\n\n" + "\n".join(extra)
    base += (
        "\n\n提示：官方基线数（ResNet-18/50）见论文 MedMNIST v2 表 3，"
        "或 https://medmnist.com 的 Results 页。本机离线，需联网确认。"
    )
    return base


# ============ 6. 每日反馈骨架 ============
@mcp.tool()
def feedback_scaffold(date: str) -> str:
    """生成某天的每日反馈骨架（只返回内容，不写文件）。

    参数 date 格式 YYYY-MM-DD。写文件要用户点头，工具不擅自落盘。
    """
    tpl = (
        f"# {date} 学习反馈\n\n"
        "## 今天完成的事\n\n1. \n\n---\n\n"
        "## 学到的知识点\n\n### 一、\n\n\n---\n\n"
        "## 心得\n\n**1.**\n\n---\n\n"
        "## 卡住的地方\n\n\n---\n\n"
        "## 明天计划\n\n1. \n\n---\n\n"
        "## 一句话总结\n\n"
    )
    target = FEEDBACK_DIR / f"{date}.md"
    exist = "已存在，写入会覆盖。" if target.exists() else "尚不存在。"
    return f"目标路径：{target}（{exist}）\n\n```markdown\n{tpl}```"


# ============ 7. Notebook 结构 ============
@mcp.tool()
def notebook_outline(with_output: bool = False) -> str:
    """列出 Learn.ipynb 每个单元的首行与执行状态，找出没跑过的单元。

    用途：定位哪些代码写了但从未执行（这类单元最容易藏拼写错误）。
    """
    if not NOTEBOOK.exists():
        return f"找不到 {NOTEBOOK}"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    lines = [f"共 {len(nb['cells'])} 个单元"]
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]).strip().splitlines()
        first = src[0][:60] if src else "(空)"
        n_out = len(c.get("outputs", []))
        ec = c.get("execution_count")
        flag = "未执行" if ec is None and c["cell_type"] == "code" else f"#{ec}"
        lines.append(f"{i:>3} {c['cell_type'][:4]} {flag:>8} 输出{n_out}  {first}")
    if with_output:
        lines.append("\n提示：需要正文请直接读文件。")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run("stdio")
