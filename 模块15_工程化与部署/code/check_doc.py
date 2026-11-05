"""校验讲义：行数、代码块数、Python 语法、LaTeX/emoji 残留。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MD = Path(r"E:\AI学习\模块15_工程化与部署\15_工程化与部署.md")
text = MD.read_text(encoding="utf-8")
lines = text.splitlines()

blocks = re.findall(r"^```(\w*)\n(.*?)^```$", text, flags=re.S | re.M)
py_blocks = [(lang, code) for lang, code in blocks if lang == "python"]

errors = []
for i, (_, code) in enumerate(py_blocks, 1):
    try:
        compile(code, f"block_{i}", "exec")
    except SyntaxError as exc:
        errors.append((i, exc))

# 只抓 "$...$" 数学公式，不误伤 shell 提示符 "$ git ..."
latex = [
    (n, ln) for n, ln in enumerate(lines, 1)
    if re.search(r"\$[^$\n]+\$", ln) or "\\frac" in ln or "\\sum" in ln or "\\(" in ln
]
emoji = [
    (n, ln) for n, ln in enumerate(lines, 1)
    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", ln)
]

print(f"文件: {MD}")
print(f"总行数: {len(lines)}")
print(f"代码块总数: {len(blocks)}  其中 python: {len(py_blocks)}")
print(f"其他语言: {sorted({lang for lang, _ in blocks if lang != 'python'})}")
print(f"Python 语法错误: {len(errors)}")
for i, exc in errors:
    print(f"  block {i}: {exc}")
print(f"疑似 LaTeX 行: {len(latex)}")
for n, ln in latex[:5]:
    print(f"  {n}: {ln[:80]}")
print(f"疑似 emoji 行: {len(emoji)}")
for n, ln in emoji[:5]:
    print(f"  {n}: {ln[:80]}")

sys.exit(1 if errors else 0)
