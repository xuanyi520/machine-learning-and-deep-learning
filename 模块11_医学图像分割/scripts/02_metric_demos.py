"""手算 Dice/IoU、类别不平衡、patch 采样的真实验证。"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"
res = {}


def dice_iou(area_a, area_b, inter):
    dice = 2 * inter / (area_a + area_b)
    iou = inter / (area_a + area_b - inter)
    return dice, iou


print("=== 1. 三组手算 ===")
cases = [(100, 120, 90), (500, 400, 350), (100, 100, 60)]
res["dice_cases"] = []
for a, b, i in cases:
    d, j = dice_iou(a, b, i)
    d2 = 2 * j / (1 + j)
    j2 = d / (2 - d)
    print(f"A={a} B={b} inter={i} -> Dice={d:.4f} IoU={j:.4f} "
          f"(反算 Dice={d2:.4f}, IoU={j2:.4f})")
    res["dice_cases"].append({"a": a, "b": b, "inter": i, "dice": d, "iou": j})

print("\n=== 2. Dice-IoU 换算表 ===")
table = []
for d in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    table.append((d, d / (2 - d)))
    print(f"Dice {d:.2f} -> IoU {d/(2-d):.4f}")
res["dice_iou_table"] = table

print("\n=== 3. 空掩码与 smooth ===")
for smooth in [0.0, 1e-6, 1.0]:
    for a, b, i in [(0, 0, 0), (0, 100, 0), (100, 0, 0)]:
        try:
            val = (2 * i + smooth) / (a + b + smooth)
        except ZeroDivisionError:
            val = float("nan")
        print(f"smooth={smooth:<6} |A|={a:<3} |B|={b:<3} -> Dice={val}")
        res.setdefault("empty_cases", []).append(
            {"smooth": smooth, "a": a, "b": b, "dice": val})

print("\n=== 4. 小目标对位移的敏感性（二值圆盘） ===")
shift_rows = []
for r in [5, 10, 20, 30]:
    for s in [1, 2, 3]:
        yy, xx = np.mgrid[-40:41, -40:41]
        A = (yy ** 2 + xx ** 2) <= r ** 2
        B = ((yy) ** 2 + (xx - s) ** 2) <= r ** 2
        d = 2 * (A & B).sum() / (A.sum() + B.sum())
        shift_rows.append((r, s, float(d)))
        print(f"半径 {r:2d}px 平移 {s}px -> Dice {d:.4f}")
res["shift"] = shift_rows

print("\n=== 5. 全猜背景的真实代价 ===")
d = np.load(OUT / "phantom.npz")
for split, pre in [("val", "va"), ("test", "te")]:
    y = d[f"{pre}_y"]
    n = y.size
    fg = (y > 0).mean()
    liver = (y == 1).mean()
    tumor = (y == 2).mean()
    # 全猜背景
    acc_bg = 1 - fg
    # 全猜肝脏（类别 1）
    acc_liver = liver + (y == 1).sum() / n * 0  # 准确率就是类别 1 的占比
    dice_liver_all = 2 * liver / (liver + 1.0)
    # 全猜肿瘤
    dice_tumor_all = 2 * tumor / (tumor + 1.0)
    print(f"{split}: 前景={fg:.4f} 肝脏={liver:.4f} 肿瘤={tumor:.4f}")
    print(f"  全猜背景: 像素准确率={acc_bg:.4f}, 肝脏 Dice=0, 肿瘤 Dice=0")
    print(f"  全猜肝脏: 像素准确率={liver:.4f}, 肝脏 Dice={dice_liver_all:.4f}, 肿瘤 Dice=0")
    print(f"  全猜肿瘤: 像素准确率={tumor:.4f}, 肿瘤 Dice={dice_tumor_all:.4f}")
    res[f"imbalance_{split}"] = {
        "foreground": float(fg), "liver": float(liver), "tumor": float(tumor),
        "acc_all_bg": float(acc_bg), "dice_all_liver": float(dice_liver_all),
        "dice_all_tumor": float(dice_tumor_all)}

print("\n=== 6. 随机 patch 里有多少含肿瘤 ===")
rng = np.random.default_rng(0)
y = d["va_y"]
pos_idx = np.where((y == 2).any(axis=(1, 2)))[0]
for ps in [64, 96]:
    hits, total, pure_bg = 0, 0, 0
    for _ in range(2000):
        k = rng.choice(pos_idx)
        y0 = rng.integers(0, 256 - ps + 1)
        x0 = rng.integers(0, 256 - ps + 1)
        patch = y[k, y0:y0 + ps, x0:x0 + ps]
        hits += int((patch == 2).any())
        pure_bg += int((patch == 0).all())
        total += 1
    print(f"{ps}x{ps}: 随机采样含肿瘤比例={hits/total:.3f}, 纯背景比例={pure_bg/total:.3f}")
    print(f"        pos:neg=1:1 后含肿瘤比例=0.500（构造保证）")
    res[f"patch_{ps}"] = {"tumor_patch_rate": hits / total, "pure_bg_rate": pure_bg / total}

print("\n=== 7. 各向异性：不重采样的体积误差 ===")
spacing = (0.7, 0.7, 5.0)
voxel = spacing[0] * spacing[1] * spacing[2]
print(f"体素体积 {voxel:.3f} mm^3；重采样到 1mm 后 1.000 mm^3，差 {voxel:.1f} 倍")
print(f"直径 10mm 球体: 原始体素计数约 {4/3*np.pi*5**3/voxel:.0f}，1mm 下 {4/3*np.pi*5**3:.0f}")
res["anisotropy"] = {"voxel_mm3": voxel, "ratio": voxel, "sphere10_voxels_orig": 4/3*np.pi*125/voxel,
                     "sphere10_voxels_iso": 4/3*np.pi*125}

json.dump(res, open(OUT / "demo_numbers.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved", OUT / "demo_numbers.json")
