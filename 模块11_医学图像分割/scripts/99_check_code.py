"""提取 Markdown 里的 python 代码块并做 compile 检查。"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
errors = []
for i, code in enumerate(blocks, 1):
    try:
        compile(code, f"block{i}", "exec")
    except SyntaxError as e:
        errors.append((i, e.lineno, e.msg, code.splitlines()[max(0, (e.lineno or 1) - 3):(e.lineno or 1) + 2]))
print(f"文件: {path}")
print(f"总行数: {len(text.splitlines())}")
print(f"python 代码块: {len(blocks)}")
print(f"语法错误: {len(errors)}")
for i, ln, msg, snippet in errors:
    print(f"  block {i} line {ln}: {msg}")
    for s in snippet:
        print("    ", s)
