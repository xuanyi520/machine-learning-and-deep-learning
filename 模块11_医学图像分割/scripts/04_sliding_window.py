"""滑窗推理：512x512 大图上的三种推理方式对比 + 显存实测。"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"
device = "cuda" if torch.cuda.is_available() else "cpu"

spec = importlib.util.spec_from_file_location(
    "tm", r"E:\AI学习\模块11_医学图像分割\scripts\01_train_unet.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
gspec = importlib.util.spec_from_file_location(
    "gen", r"E:\AI学习\模块11_医学图像分割\scripts\00_make_phantom.py")
gen = importlib.util.module_from_spec(gspec)
gspec.loader.exec_module(gen)


def norm(img):
    img = np.clip(img, tm.HU_MIN, tm.HU_MAX)
    img = (img - tm.HU_MIN) / (tm.HU_MAX - tm.HU_MIN)
    return torch.from_numpy(img[None, None].astype(np.float32)).to(device)


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
    """手写滑窗：每个窗口输出 softmax 概率，按高斯权重累加。"""
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
    for y in ys:
        for xx in xs:
            patch = x[:, :, y:y + window, xx:xx + window]
            p = torch.softmax(model(patch), 1)[0]
            prob[:, y:y + window, xx:xx + window] += p * g
            weight[y:y + window, xx:xx + window] += g
    return (prob / weight).argmax(0).cpu().numpy().astype(np.uint8)


def dice(pred, gt, cls):
    p, t = pred == cls, gt == cls
    if p.sum() + t.sum() == 0:
        return 1.0
    return float(2 * (p & t).sum() / (p.sum() + t.sum()))


def peak_mem(fn, *args, **kw):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn(*args, **kw)
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() / 1e6, time.time() - t


def main():
    model = tm.UNet(1, 3).to(device)
    model.load_state_dict(torch.load(OUT / "unet_best.pt", map_location=device))
    model.eval()

    rng = np.random.default_rng(2026)
    cases = [gen.make_case(rng, size=512, anatomy_scale=1.0, hard=True, has_tumor=True) for _ in range(6)]

    rows = []
    for i, (img, gt) in enumerate(cases):
        x = norm(img)
        p_direct, m_direct, t_direct = peak_mem(infer_direct, model, x)
        p_resize, m_resize, t_resize = peak_mem(infer_resize, model, x)
        p_sw, m_sw, t_sw = peak_mem(infer_sliding, model, x, 128, 64)
        p_sw256, m_sw256, t_sw256 = peak_mem(infer_sliding, model, x, 256, 128)
        rows.append({
            "i": i,
            "direct": {"liver": dice(p_direct, gt, 1), "tumor": dice(p_direct, gt, 2),
                       "mem_MB": m_direct, "sec": t_direct},
            "resize": {"liver": dice(p_resize, gt, 1), "tumor": dice(p_resize, gt, 2),
                       "mem_MB": m_resize, "sec": t_resize},
            "sw128": {"liver": dice(p_sw, gt, 1), "tumor": dice(p_sw, gt, 2),
                      "mem_MB": m_sw, "sec": t_sw},
            "sw256": {"liver": dice(p_sw256, gt, 1), "tumor": dice(p_sw256, gt, 2),
                      "mem_MB": m_sw256, "sec": t_sw256},
        })
        print(f"case {i}: direct liver {rows[-1]['direct']['liver']:.4f} tumor "
              f"{rows[-1]['direct']['tumor']:.4f} | sw128 liver {rows[-1]['sw128']['liver']:.4f} "
              f"tumor {rows[-1]['sw128']['tumor']:.4f} | {m_direct:.0f}MB vs {m_sw:.0f}MB")

    summary = {}
    for k in ["direct", "resize", "sw128", "sw256"]:
        summary[k] = {
            "liver_dice": float(np.mean([r[k]["liver"] for r in rows])),
            "tumor_dice": float(np.mean([r[k]["tumor"] for r in rows])),
            "mem_MB": float(np.mean([r[k]["mem_MB"] for r in rows])),
            "sec": float(np.mean([r[k]["sec"] for r in rows])),
        }
        print(k, summary[k])
    json.dump({"summary": summary, "per_case": rows},
              open(OUT / "sliding_window.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
