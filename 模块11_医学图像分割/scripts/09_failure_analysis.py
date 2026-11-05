"""失败案例定量分析：肿瘤 Dice 与病灶大小、对比度的关系。"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"
d = np.load(OUT / "phantom.npz")
x, y = d["te_x"], d["te_y"]
pred = np.load(OUT / "test_pred.npz")["pred"]
summary = json.load(open(OUT / "eval_summary.json"))
rows = {r["i"]: r for r in summary["per_case"]}

recs = []
for i in range(len(y)):
    if not (y[i] == 2).any():
        continue
    t = y[i] == 2
    liver = (y[i] == 1) | (y[i] == 2)
    p = pred[i] == 2
    area = int(t.sum())
    contrast = float(abs(x[i][t].mean() - x[i][liver].mean()))
    lab, n = ndimage.label(t)
    # 到肝边界的最近距离（衡量病灶是否贴边）
    lb = liver ^ ndimage.binary_erosion(liver)
    dt_out = ndimage.distance_transform_edt(~lb)
    margin = float(dt_out[t].min()) if t.any() else 0.0
    recs.append({"i": i, "dice": rows[i]["tumor_dice"], "area": area,
                 "contrast": contrast, "lesions": int(n), "margin": margin,
                 "pred_area": int(p.sum())})

dices = np.array([r["dice"] for r in recs])
areas = np.array([r["area"] for r in recs])
cons = np.array([r["contrast"] for r in recs])
print("n =", len(recs))
print("Dice vs 面积   rho=%.3f p=%.4f" % spearmanr(dices, areas))
print("Dice vs |对比度| rho=%.3f p=%.4f" % spearmanr(dices, cons))
print("面积 <100px 的平均 Dice:", np.mean([r["dice"] for r in recs if r["area"] < 100]))
print("面积 >=300px 的平均 Dice:", np.mean([r["dice"] for r in recs if r["area"] >= 300]))
print("|对比度| <10HU 的平均 Dice:", np.mean([r["dice"] for r in recs if r["contrast"] < 10]))
print("|对比度| >=15HU 的平均 Dice:", np.mean([r["dice"] for r in recs if r["contrast"] >= 15]))
print("完全漏检（预测面积=0）:", sum(r["pred_area"] == 0 for r in recs), "/", len(recs))
print("预测了但零重叠:", sum((r["pred_area"] > 0) and (r["dice"] == 0) for r in recs), "/", len(recs))
print("贴边（margin<2px）比例:", np.mean([r["margin"] < 2 for r in recs]))

print("\n最差 8 例:")
for r in sorted(recs, key=lambda z: z["dice"])[:8]:
    print(f"#{r['i']:2d} Dice={r['dice']:.3f} 面积={r['area']:4d} "
          f"|对比度|={r['contrast']:.1f}HU 病灶数={r['lesions']} 贴边={r['margin']:.1f}px "
          f"预测面积={r['pred_area']:4d}")

json.dump(recs, open(OUT / "failure_analysis.json", "w"), indent=2, ensure_ascii=False)
