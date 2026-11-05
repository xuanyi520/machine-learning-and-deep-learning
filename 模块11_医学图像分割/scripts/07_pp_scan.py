"""后处理参数扫描：最小肿瘤连通域阈值对 Dice / HD95 的影响。"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"


def pp(mask, min_tumor, keep_largest_liver=True):
    out = np.zeros_like(mask)
    liver = mask == 1
    if liver.any():
        lab, n = ndimage.label(liver)
        sizes = ndimage.sum(liver, lab, range(1, n + 1))
        if keep_largest_liver:
            liver = lab == (int(np.argmax(sizes)) + 1)
        liver = ndimage.binary_fill_holes(ndimage.binary_closing(liver, np.ones((3, 3))))
        out[liver] = 1
    tumor = mask == 2
    if tumor.any():
        lab, n = ndimage.label(tumor)
        sizes = ndimage.sum(tumor, lab, range(1, n + 1))
        keep = np.zeros_like(tumor)
        for k, s in enumerate(sizes, 1):
            if s >= min_tumor:
                keep |= lab == k
        keep = ndimage.binary_closing(keep, np.ones((3, 3)))
        out[keep] = 2
    return out


def surface(pred, gt):
    if not pred.any() or not gt.any():
        return None
    st = ndimage.generate_binary_structure(2, 1)
    ps, gs = pred ^ ndimage.binary_erosion(pred, st), gt ^ ndimage.binary_erosion(gt, st)
    dg = ndimage.distance_transform_edt(~gs)
    dp = ndimage.distance_transform_edt(~ps)
    return np.concatenate([dg[ps], dp[gs]])


def dice(p, t):
    if p.sum() + t.sum() == 0:
        return 1.0
    return 2 * (p & t).sum() / (p.sum() + t.sum())


def run(preds, gts, min_tumor):
    ds, hs, comps = [], [], []
    for p, t in zip(preds, gts):
        q = pp(p, min_tumor)
        if (t == 2).any():
            ds.append(dice(q == 2, t == 2))
            d = surface(q == 2, t == 2)
            if d is not None:
                hs.append(np.percentile(d, 95))
        lab, n = ndimage.label(q == 2)
        comps.append(n)
    return {"dice": float(np.mean(ds)), "hd95": float(np.mean(hs)),
            "components": float(np.mean(comps))}


d = np.load(OUT / "phantom.npz")
gts = d["te_y"]
preds = np.load(OUT / "test_pred.npz")
res = {}
for name in ["pred", "weak_pred"]:
    print("==", name)
    rows = []
    for mt in [0, 10, 20, 40, 80, 150, 300]:
        r = run(preds[name], gts, mt)
        rows.append({"min_tumor": mt, **r})
        print(f"  min_tumor={mt:3d}  Dice={r['dice']:.4f}  HD95={r['hd95']:.2f}  平均连通域={r['components']:.2f}")
    res[name] = rows

# 原始预测：每例肿瘤连通域数量与假阳性碎块
for name in ["pred", "weak_pred"]:
    counts, tiny = [], []
    for p in preds[name]:
        lab, n = ndimage.label(p == 2)
        counts.append(n)
        sizes = ndimage.sum(p == 2, lab, range(1, n + 1)) if n else []
        tiny.append(int(sum(s < 20 for s in sizes)))
    print(f"{name}: 平均连通域 {np.mean(counts):.2f}, 平均 <20px 碎块 {np.mean(tiny):.2f}")
    res[name + "_components_raw"] = {"mean_components": float(np.mean(counts)),
                                     "mean_tiny": float(np.mean(tiny))}

json.dump(res, open(OUT / "pp_scan.json", "w"), indent=2, ensure_ascii=False)
print("saved")
