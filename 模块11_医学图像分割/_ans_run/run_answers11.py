"""模块 11 参考答案：练习 1-8 的实测脚本。

复用 outputs/ 下已训练的 unet_best.pt、phantom.npz、test_pred.npz、
train_log.json、eval_summary.json、pp_scan.json、failure_analysis.json。
练习 6 重新跑一遍滑窗（6 例 512x512，秒级）。
"""
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

ROOT = r"E:\AI学习\模块11_医学图像分割"
OUT = os.path.join(ROOT, "outputs")
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
res = {}


def dump():
    with open(os.path.join(HERE, "results_mod11.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tm = load_mod("tm", os.path.join(ROOT, "scripts", "01_train_unet.py"))
gen = load_mod("gen", os.path.join(ROOT, "scripts", "00_make_phantom.py"))


def dice_iou(a, b, inter):
    return 2 * inter / (a + b), inter / (a + b - inter)


# ===================================================== 练习 1
def ex1():
    cases = [(100, 120, 90), (500, 400, 350), (100, 100, 60)]
    rows = []
    for a, b, i in cases:
        d, j = dice_iou(a, b, i)
        d_back = 2 * j / (1 + j)
        j_back = d / (2 - d)
        rows.append({"a": a, "b": b, "inter": i, "dice": d, "iou": j,
                     "dice_from_iou": d_back, "iou_from_dice": j_back,
                     "err_d": abs(d - d_back), "err_j": abs(j - j_back)})
        print(f"A={a} B={b} inter={i}: Dice={d:.6f} IoU={j:.6f} "
              f"反算 Dice={d_back:.6f} 反算 IoU={j_back:.6f}")
    # Dice=0.5 -> IoU
    d = 0.5
    rows.append({"special": "Dice=0.5", "iou": d / (2 - d),
                 "iou_frac": f"{int(round(d/(2-d)*3))}/3"})
    res["ex1"] = rows
    dump()


# ===================================================== 练习 2
def ex2():
    d = np.load(os.path.join(OUT, "phantom.npz"))
    y = d["va_y"]
    H, W = y.shape[1:]
    n = len(y)
    stats = {}
    for name, cls in [("all_bg", None), ("all_liver", 1), ("all_tumor", 2)]:
        pred = np.zeros_like(y) if cls is None else np.full_like(y, cls)
        acc = float((pred == y).mean())
        ld = np.mean([tm.case_dice(pred[i], y[i], 1) for i in range(n)])
        td = np.mean([tm.case_dice(pred[i], y[i], 2) for i in range(n) if (y[i] == 2).any()])
        stats[name] = {"acc": acc, "liver_dice": float(ld), "tumor_dice": float(td)}
        print(name, stats[name])
    stats["fg_ratio"] = float((y > 0).mean())
    stats["liver_ratio"] = float((y == 1).mean())
    stats["tumor_ratio"] = float((y == 2).mean())
    stats["n_tumor_cases"] = int(sum((y[i] == 2).any() for i in range(n)))
    # 全猜肝脏的 Dice 手算核对：2*|liver|/(|liver|+1)
    lr = stats["liver_ratio"]
    stats["all_liver_dice_formula"] = 2 * lr / (lr + 1)
    res["ex2"] = stats
    dump()


# ===================================================== 练习 3
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, base=32, upsample="transpose"):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], base * 16)
        if upsample == "transpose":
            self.up4 = nn.ConvTranspose2d(base * 16, c[3], 2, stride=2)
            self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
            self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
            self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        else:
            self.up4 = nn.Conv2d(base * 16, c[3], 3, padding=1)
            self.up3 = nn.Conv2d(c[3], c[2], 3, padding=1)
            self.up2 = nn.Conv2d(c[2], c[1], 3, padding=1)
            self.up1 = nn.Conv2d(c[1], c[0], 3, padding=1)
        self.upsample = upsample
        self.dec4 = DoubleConv(c[3] * 2, c[3])
        self.dec3 = DoubleConv(c[2] * 2, c[2])
        self.dec2 = DoubleConv(c[1] * 2, c[1])
        self.dec1 = DoubleConv(c[0] * 2, c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def _up(self, x, block):
        if self.upsample == "transpose":
            return block(x)
        return block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._up(b, self.up4), e4], 1))
        d3 = self.dec3(torch.cat([self._up(d4, self.up3), e3], 1))
        d2 = self.dec2(torch.cat([self._up(d3, self.up2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, self.up1), e1], 1))
        return self.head(d1)


def ex3():
    m = UNet(1, 3)
    x = torch.randn(1, 1, 256, 256)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print("256 输出", tuple(y.shape), "参数量", params)
    # 250 报错
    err = None
    try:
        m(torch.randn(1, 1, 250, 250))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    print("250 报错:", err)
    # pad
    x250 = torch.randn(1, 1, 250, 250)
    xp = F.pad(x250, (3, 3, 3, 3))
    yp = m(xp)
    yc = yp[:, :, 3:253, 3:253]
    # 插值上采样版
    mi = UNet(1, 3, upsample="interp")
    yi = mi(torch.randn(1, 1, 256, 256))
    res["ex3"] = {
        "out256": list(y.shape), "params": params,
        "error250": err, "pad_out": list(yp.shape), "crop_out": list(yc.shape),
        "interp_out": list(yi.shape),
        "interp_params": sum(p.numel() for p in mi.parameters()),
    }
    print("pad", tuple(yp.shape), "crop", tuple(yc.shape), "interp", tuple(yi.shape))
    dump()


# ===================================================== 练习 4
def ex4():
    d = np.load(os.path.join(OUT, "phantom.npz"))
    va_x, va_y = d["va_x"], d["va_y"]
    pos_idx = [i for i in range(len(va_y)) if (va_y[i] == 2).any()]
    rng = np.random.default_rng(0)
    out = {"n_pos_cases": len(pos_idx)}
    for patch in [64, 96]:
        tum = bg = 0
        N = 2000
        for _ in range(N):
            i = pos_idx[rng.integers(0, len(pos_idx))]
            y0 = rng.integers(0, 256 - patch + 1)
            x0 = rng.integers(0, 256 - patch + 1)
            p = va_y[i, y0:y0 + patch, x0:x0 + patch]
            if (p == 2).any():
                tum += 1
            if (p == 0).all():
                bg += 1
        out[f"rand_{patch}"] = {"tumor_rate": tum / N, "pure_bg_rate": bg / N}

        # pos:neg = 1:1
        T = 2000
        half = T // 2
        hit = 0
        for k in range(T):
            want_pos = (k % 2 == 0)
            if want_pos:
                i = pos_idx[rng.integers(0, len(pos_idx))]
                ys, xs = np.where(va_y[i] == 2)
                j = rng.integers(0, len(ys))
                y0 = int(np.clip(ys[j] - rng.integers(0, patch), 0, 256 - patch))
                x0 = int(np.clip(xs[j] - rng.integers(0, patch), 0, 256 - patch))
            else:
                i = pos_idx[rng.integers(0, len(pos_idx))]
                y0 = int(rng.integers(0, 256 - patch + 1))
                x0 = int(rng.integers(0, 256 - patch + 1))
            p = va_y[i, y0:y0 + patch, x0:x0 + patch]
            hit += int((p == 2).any())
        out[f"posneg_{patch}"] = {"tumor_rate": hit / T}
        print(patch, out[f"rand_{patch}"], out[f"posneg_{patch}"])
    res["ex4"] = out
    dump()


# ===================================================== 练习 5
def surface_dists(pred, gt, spacing=(1.0, 1.0)):
    if not pred.any() or not gt.any():
        return None
    st = ndimage.generate_binary_structure(2, 1)
    ps = pred ^ ndimage.binary_erosion(pred, st)
    gs = gt ^ ndimage.binary_erosion(gt, st)
    dg = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dp = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return np.concatenate([dg[ps], dp[gs]])


def hd95_assd(pred, gt):
    d = surface_dists(pred, gt)
    if d is None:
        return None, None
    return float(np.percentile(d, 95)), float(d.mean())


def ex5():
    model = tm.UNet(1, 3).to(DEV)
    model.load_state_dict(torch.load(os.path.join(OUT, "unet_best.pt"), map_location=DEV))
    model.eval()
    d = np.load(os.path.join(OUT, "phantom.npz"))
    x, y = d["te_x"], d["te_y"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            xb = torch.from_numpy(x[i:i + 8]).float()
            xb = torch.clamp(xb, tm.HU_MIN, tm.HU_MAX)
            xb = ((xb - tm.HU_MIN) / (tm.HU_MAX - tm.HU_MIN))[:, None].to(DEV)
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds).astype(np.uint8)

    liver = [tm.case_dice(pred[i], y[i], 1) for i in range(len(y))]
    td, hd, assd, acc, areas, pred_areas = [], [], [], [], [], []
    for i in range(len(y)):
        acc.append(float((pred[i] == y[i]).mean()))
        if (y[i] == 2).any():
            td.append(tm.case_dice(pred[i], y[i], 2))
            h, a = hd95_assd(pred[i] == 2, y[i] == 2)
            if h is not None:
                hd.append(h)
                assd.append(a)
            areas.append(int((y[i] == 2).sum()))
            pred_areas.append(int((pred[i] == 2).sum()))

    log = json.load(open(os.path.join(OUT, "train_log.json"), encoding="utf-8"))
    out = {
        "liver_dice_mean": float(np.mean(liver)),
        "liver_dice_median": float(np.median(liver)),
        "liver_dice_min": float(np.min(liver)),
        "tumor_dice_mean": float(np.mean(td)),
        "tumor_dice_median": float(np.median(td)),
        "tumor_zero_rate": float(np.mean([t == 0 for t in td])),
        "tumor_below_03_rate": float(np.mean([t < 0.3 for t in td])),
        "hd95_mean": float(np.mean(hd)),
        "hd95_median": float(np.median(hd)),
        "assd_mean": float(np.mean(assd)),
        "pixel_acc": float(np.mean(acc)),
        "n": len(y), "n_tumor": len(td),
        "epoch10": log[9],
        "epoch11": log[10],
        "last_epoch": log[-1],
        "log": log,
    }
    print({k: out[k] for k in out if k != "log"})
    res["ex5"] = out
    np.savez(os.path.join(HERE, "pred_check.npz"), pred=pred)
    dump()


# ===================================================== 练习 6
def norm_t(x):
    x = np.clip(x, tm.HU_MIN, tm.HU_MAX)
    x = (x - tm.HU_MIN) / (tm.HU_MAX - tm.HU_MIN)
    return torch.from_numpy(x[None, None].astype(np.float32)).to(DEV)


def gaussian_window(size, sigma=None):
    sigma = sigma or size / 8
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32)


@torch.no_grad()
def infer_direct(model, x):
    return model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def infer_resize(model, x, size=256):
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    logits = model(small)
    logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def infer_sliding(model, x, window=128, stride=64, num_classes=3):
    _, _, H, W = x.shape
    prob = torch.zeros((num_classes, H, W), device=x.device)
    weight = torch.zeros((H, W), device=x.device)
    g = torch.from_numpy(gaussian_window(window)).to(x.device)
    ys = list(range(0, H - window + 1, stride))
    xs = list(range(0, W - window + 1, stride))
    if ys[-1] != H - window:
        ys.append(H - window)
    if xs[-1] != W - window:
        xs.append(W - window)
    for yy in ys:
        for xx in xs:
            patch = x[:, :, yy:yy + window, xx:xx + window]
            p = torch.softmax(model(patch), 1)[0]
            prob[:, yy:yy + window, xx:xx + window] += p * g
            weight[yy:yy + window, xx:xx + window] += g
    return (prob / weight).argmax(0).cpu().numpy().astype(np.uint8)


def peak(fn, *a):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    o = fn(*a)
    torch.cuda.synchronize()
    return o, torch.cuda.max_memory_allocated() / 1e6, time.time() - t


def ex6():
    model = tm.UNet(1, 3).to(DEV)
    model.load_state_dict(torch.load(os.path.join(OUT, "unet_best.pt"), map_location=DEV))
    model.eval()
    rng = np.random.default_rng(2026)
    cases = [gen.make_case(rng, size=512, anatomy_scale=1.0, hard=True, has_tumor=True)
             for _ in range(6)]
    rows = []
    for i, (img, gt) in enumerate(cases):
        x = norm_t(img)
        rec = {"i": i}
        for name, fn in [("direct", lambda: infer_direct(model, x)),
                         ("resize256", lambda: infer_resize(model, x)),
                         ("sw128", lambda: infer_sliding(model, x, 128, 64)),
                         ("sw256", lambda: infer_sliding(model, x, 256, 128))]:
            p, mem, sec = peak(fn)
            rec[name] = {"liver": tm.case_dice(p, gt, 1), "tumor": tm.case_dice(p, gt, 2),
                         "mem_MB": mem, "sec": sec}
        rows.append(rec)
        print(i, {k: round(v["tumor"], 4) for k, v in rec.items() if k != "i"})
    summ = {}
    for k in ["direct", "resize256", "sw128", "sw256"]:
        summ[k] = {m: float(np.mean([r[k][m] for r in rows]))
                   for m in ["liver", "tumor", "mem_MB", "sec"]}
        print(k, summ[k])
    res["ex6"] = {"summary": summ, "per_case": rows}
    dump()


# ===================================================== 练习 7
def ex7():
    p = json.load(open(os.path.join(OUT, "pp_scan.json"), encoding="utf-8"))
    res["ex7"] = p
    print(json.dumps(p["pred"], indent=1))
    dump()


# ===================================================== 练习 8
def ex8():
    f = json.load(open(os.path.join(OUT, "failure_analysis.json"), encoding="utf-8"))
    pos = [r for r in f if r["dice"] is not None]
    pos.sort(key=lambda r: r["dice"])
    worst = pos[:8]
    from scipy.stats import spearmanr
    d = [r["dice"] for r in pos]
    area = [r["area"] for r in pos]
    contrast = [r["contrast"] for r in pos]
    margin = [r["margin"] for r in pos]
    out = {
        "n": len(pos),
        "worst8": worst,
        "spearman_area": [float(v) for v in spearmanr(d, area)],
        "spearman_contrast": [float(v) for v in spearmanr(d, contrast)],
        "spearman_margin": [float(v) for v in spearmanr(d, margin)],
        "zero": int(sum(1 for r in pos if r["dice"] == 0)),
        "mean_area_all": float(np.mean(area)),
        "mean_contrast_all": float(np.mean(contrast)),
    }
    print("worst8:")
    for r in worst:
        print(r)
    print("spearman", out["spearman_area"], out["spearman_contrast"], out["spearman_margin"])
    res["ex8"] = out
    dump()


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"1": ex1, "2": ex2, "3": ex3, "4": ex4, "5": ex5, "6": ex6, "7": ex7, "8": ex8}
    if which == "all":
        for k in "12345678":
            print("=====", k)
            fns[k]()
    else:
        for k in which:
            fns[k]()
    print("DONE")
