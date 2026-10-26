# -*- coding: utf-8 -*-
r"""
基于 top1 项目的中文模板，生成一份更适合技术讲义与代码的模板 ref_ai.docx。

改动：
  - Source Code 段落：段前后间距 0、单倍行距、Consolas + 等线，9pt
  - Verbatim Char（行内 code）：Consolas
  - Heading 1/2/3：保留原样式，调整段前段后，让层级更清楚

用法：
    set PYTHONUTF8=1
    python make_ref.py
"""
from pathlib import Path
import shutil

from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

SRC = Path(r"E:\NewChallenge—文件\top1挑战\_build\ref_cn.docx")
DST = Path(r"E:\AI学习\Word版\ref_ai.docx")


def set_fonts(style, ascii_font, east_asia_font, size=None):
    """设置样式的西文/中文/东亚字体。python-docx 只管 ascii，eastAsia 要手写 XML。"""
    rpr = style.element.get_or_add_rPr()

    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, rfonts)

    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)
    rfonts.set(qn("w:cs"), ascii_font)
    rfonts.set(qn("w:eastAsia"), east_asia_font)

    # 去掉主题字体，否则设置会被主题覆盖
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]

    if size is not None:
        style.font.size = Pt(size)


def main():
    if not SRC.exists():
        print("找不到源模板:", SRC)
        return 1

    shutil.copy(SRC, DST)
    doc = Document(DST)

    names = {s.name for s in doc.styles}
    print("模板中的样式（部分）:", sorted(n for n in names if n)[:20])

    # ---- 代码块 ----
    # 这份模板里没有 Source Code 样式，pandoc 转换时会自己临时造一个。
    # 这里显式建出来（styleId 会自动成为 SourceCode，正是 pandoc 要找的那个），
    # 让它带上紧凑的段落格式和等宽字体。
    if "Source Code" not in names:
        from docx.enum.style import WD_STYLE_TYPE
        s = doc.styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = doc.styles["Normal"]
        s.quick_style = True
        print("已新建 Source Code 样式")

    s = doc.styles["Source Code"]
    pf = s.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    pf.left_indent = Pt(12)
    set_fonts(s, "Consolas", "等线", size=9)
    print("已调整 Source Code")

    # ---- 行内 code ----
    if "Verbatim Char" in names:
        s = doc.styles["Verbatim Char"]
        set_fonts(s, "Consolas", "等线", size=10)
        print("已调整 Verbatim Char")

    # ---- 标题：段前留白，段后收紧 ----
    for hname, before, after, size in [
        ("Heading 1", 18, 10, 18),
        ("Heading 2", 14, 6, 14),
        ("Heading 3", 10, 4, 12),
    ]:
        if hname in names:
            s = doc.styles[hname]
            s.paragraph_format.space_before = Pt(before)
            s.paragraph_format.space_after = Pt(after)
            s.paragraph_format.keep_with_next = True
            s.font.size = Pt(size)
            print(f"已调整 {hname}")

    doc.save(DST)
    print("\n生成:", DST, f"({DST.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
