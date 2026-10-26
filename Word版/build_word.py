# -*- coding: utf-8 -*-
r"""
把 E:\AI学习\ 下的 markdown 讲稿合并导出为 Word。

用法：
    set PYTHONUTF8=1
    E:\Dowmload\Anaconda3_Dowmload\python.exe "E:\AI学习\Word版\build_word.py"

产出：同目录下的 5 个 .docx，按阶段分组。
改内容只改 md 源文件，然后重跑本脚本即可。
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(r"E:\AI学习")
OUT = ROOT / "Word版"
PANDOC = r"E:\Dowmload\Anaconda3_Dowmload\Library\bin\pandoc.exe"
REF = r"E:\AI学习\Word版\ref_ai.docx"


# 分页符：pandoc 的原生 openxml 块，让每节另起一页
PAGEBREAK = (
    '\n\n```{=openxml}\n'
    '<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'
    '```\n\n'
)

GROUPS = {
    "00_开始之前": [
        "README.md",
        "00_环境准备.md",
        "进度追踪.md",
    ],
    "01_数学地基": [
        "模块1_数学地基/1-1_向量与矩阵.md",
        "模块1_数学地基/1-2_微积分与梯度.md",
        "模块1_数学地基/1-3_概率与统计.md",
        "参考答案/模块1_答案.md",
    ],
    "02_机器学习基础": [
        "模块2_机器学习基础/2-1_机器学习是什么.md",
        "模块2_机器学习基础/2-2_线性回归与梯度下降.md",
        "模块2_机器学习基础/2-3_分类与评估指标.md",
        "参考答案/模块2_答案.md",
    ],
    "03_PyTorch与神经网络": [
        "模块3_PyTorch与神经网络/3-1_Tensor与自动求导.md",
        "模块3_PyTorch与神经网络/3-2_搭建神经网络.md",
        "模块3_PyTorch与神经网络/3-3_第一个训练循环.md",
        "模块3_PyTorch与神经网络/3-4_卷积网络CNN.md",
        "参考答案/模块3_答案.md",
    ],
    "04_医学影像入门": [
        "模块4_医学影像入门/4-1_读DICOM与医学图像.md",
        "模块4_医学影像入门/4-2_第一个医学影像任务.md",
        "参考答案/模块4_答案.md",
    ],
}


def main():
    if not Path(PANDOC).exists():
        print("找不到 pandoc:", PANDOC)
        return 1
    if not Path(REF).exists():
        print("找不到中文模板:", REF)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0

    for name, files in GROUPS.items():
        parts = []
        for rel in files:
            p = ROOT / rel
            if not p.exists():
                print(f"  !! 缺少 {rel}")
                continue
            parts.append(p.read_text(encoding="utf-8"))

        combined = PAGEBREAK.join(parts)

        # 临时 md 放同目录，转完即删
        tmp_md = OUT / f"_tmp_{name}.md"
        tmp_md.write_text(combined, encoding="utf-8")

        docx = OUT / f"{name}.docx"
        cmd = [
            PANDOC, str(tmp_md),
            "-o", str(docx),
            "--reference-doc", REF,
            "--resource-path", str(ROOT),
            "--standalone",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")

        if r.returncode != 0:
            print(f"  !! 失败 {name}: {r.stderr[:500]}")
        else:
            print(f"  OK  {name}.docx  ({docx.stat().st_size/1024:.0f} KB)")
            ok += 1

        tmp_md.unlink(missing_ok=True)

    print(f"\n完成 {ok}/{len(GROUPS)} 个文件，位置：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
