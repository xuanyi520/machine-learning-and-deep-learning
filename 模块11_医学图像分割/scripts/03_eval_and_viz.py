"""评估最佳 U-Net：Dice、HD95、ASSD、后处理前后、训练曲线、失败案例。"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import importlib.util
spec = importlib.util.spec_from_file_location(
    "trainmod", r"E:\AI学习\模块11_医学图像分割\scripts\01_train_unet.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT, FIG = ROOT / "outputs", ROOT / "figures"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    model = tm.UNet(1, 3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, batch=8):
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).float()
        xb = torch.clamp(xb, tm.HU_MIN, tm.HU_MAX)
        xb = ((xb - tm.HU_MIN) / (tm.HU_MAX - tm.HU_MIN))[:, None].to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out).astype(np.uint8)


def surface_dists(pred, gt, spacing=(1.0, 1.0)):
    """双向表面距离（mm）。任一方为空返回 None。"""
    if not pred.any() or not gt.any():
        return None
    st = ndimage.generate_binary_structure(2, 1)
    ps = pred ^ ndimage.binary_erosion(pred, st)
    gs = gt ^ ndimage.binary_erosion(gt, st)
    dt_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return np.concatenate([dt_g[ps], dt_p[gs]])


def hd95_assd(pred, gt, spacing=(1.0, 1.0)):
    d = surface_dists(pred, gt, spacing)
    if d is None:
        return None, None
    return float(np.percentile(d, 95)), float(d.mean())


def postprocess(mask, min_tumor=40, min_liver=500):
    """肝脏保留最大连通域；肿瘤去掉小碎块；再做闭运算 + 填洞。"""
    out = np.zeros_like(mask)
    liver = mask == 1
    if liver.any():
        lab, n = ndimage.label(liver)
        sizes = ndimage.sum(liver, lab, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        liver = lab == big
        liver = ndimage.binary_fill_holes(
            ndimage.binary_closing(liver, np.ones((3, 3))))
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


def eval_set(model, x, y, spacing=(1.0, 1.0), use_pp=False):
    pred = predict(model, x)
    if use_pp:
        pred = np.stack([postprocess(p) for p in pred])
    rows = []
    for i in range(len(y)):
        liver_d = tm.case_dice(pred[i], y[i], 1)
        has_t = bool((y[i] == 2).any())
        tumor_d = tm.case_dice(pred[i], y[i], 2) if has_t else None
        h, a = hd95_assd(pred[i] == 2, y[i] == 2, spacing) if has_t else (None, None)
        rows.append({
            "i": i, "liver_dice": liver_d, "tumor_dice": tumor_d,
            "hd95": h, "assd": a, "pixel_acc": float((pred[i] == y[i]).mean()),
            "has_tumor": has_t})
    agg = {
        "liver_dice": float(np.mean([r["liver_dice"] for r in rows])),
        "tumor_dice": float(np.mean([r["tumor_dice"] for r in rows if r["has_tumor"]])),
        "hd95": float(np.mean([r["hd95"] for r in rows if r["hd95"] is not None])),
        "assd": float(np.mean([r["assd"] for r in rows if r["assd"] is not None])),
        "pixel_acc": float(np.mean([r["pixel_acc"] for r in rows])),
        "n": len(rows), "n_tumor": sum(r["has_tumor"] for r in rows),
    }
    return pred, rows, agg


def plot_curve():
    log = json.load(open(OUT / "train_log.json"))
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [r["loss"] for r in log], color="#2b6cb0")
    ax[0].set_title("训练损失（CE + Dice）"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [r["liver_dice"] for r in log], label="肝脏", color="#2f855a")
    ax[1].plot(ep, [r["tumor_dice"] for r in log], label="肿瘤", color="#c53030")
    ax[1].set_title("验证集 Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(ep, [r["pixel_acc"] for r in log], color="#6b46c1")
    ax[2].set_title("验证集像素准确率"); ax[2].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "fig_training_curve.png", dpi=140)
    plt.close()


def show(ax, img, gt, pred, title):
    ax.imshow(np.clip(img, -200, 300), cmap="gray")
    if (gt == 1).any():
        ax.contour(gt == 1, [0.5], colors="#33ff66", linewidths=0.9)
    if (gt == 2).any():
        ax.contour(gt == 2, [0.5], colors="#ffe14d", linewidths=0.9)
    if (pred == 1).any():
        ax.contour(pred == 1, [0.5], colors="#00e5ff", linewidths=0.9, linestyles="--")
    if (pred == 2).any():
        ax.contour(pred == 2, [0.5], colors="#ff5c5c", linewidths=0.9, linestyles="--")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_examples(x, y, pred, rows, idx, name, ncol=4):
    n = len(idx)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, idx):
        t = rows[k]["tumor_dice"]
        tt = "无肿瘤" if t is None else f"肿瘤 Dice {t:.3f}"
        show(ax, x[k], y[k], pred[k], f"#{k} 肝 {rows[k]['liver_dice']:.3f} | {tt}")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=140)
    plt.close()


def main():
    d = np.load(OUT / "phantom.npz")
    tx, ty = d["te_x"], d["te_y"]

    plot_curve()
    model = load_model(OUT / "unet_best.pt")
    pred, rows, agg = eval_set(model, tx, ty)
    print("最佳模型 test:", json.dumps(agg, ensure_ascii=False))

    # 好例子：随机取肿瘤 Dice 高的 4 个 + 1 个无肿瘤
    order = sorted(rows, key=lambda r: -(r["tumor_dice"] or -1))
    good = [r["i"] for r in order[:4]]
    neg = [r["i"] for r in rows if not r["has_tumor"]][:1]
    plot_examples(tx, ty, pred, rows, good + neg, "fig_pred_examples.png", ncol=5)

    # 失败案例：肿瘤 Dice 最低的 8 个
    pos_rows = [r for r in rows if r["has_tumor"]]
    zero = [r for r in pos_rows if r["tumor_dice"] == 0][:4]
    low = [r for r in sorted(pos_rows, key=lambda r: r["tumor_dice"]) if r["tumor_dice"] > 0][:4]
    fail = [r["i"] for r in zero + low]
    plot_examples(tx, ty, pred, rows, fail, "fig_failures.png", ncol=4)

    # 误差图：一个中等案例的 FP/FN
    mid = sorted([r for r in rows if r["has_tumor"]],
                 key=lambda r: abs((r["tumor_dice"] or 0) - 0.6))[0]["i"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(np.clip(tx[mid], -200, 300), cmap="gray")
    ax[0].imshow(ty[mid], alpha=.45, cmap="jet", vmin=0, vmax=2)
    ax[0].set_title("原图 + 真值"); ax[0].axis("off")
    ax[1].imshow(np.clip(tx[mid], -200, 300), cmap="gray")
    ax[1].imshow(pred[mid], alpha=.45, cmap="jet", vmin=0, vmax=2)
    ax[1].set_title("原图 + 预测"); ax[1].axis("off")
    err = np.zeros((*ty[mid].shape, 3), np.uint8)
    fp = (pred[mid] > 0) & (ty[mid] == 0)
    fn = (ty[mid] > 0) & (pred[mid] == 0)
    err[fp] = [255, 60, 60]
    err[fn] = [60, 120, 255]
    ax[2].imshow(err)
    ax[2].set_title("红=假阳性  蓝=漏检"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "fig_error_map.png", dpi=140)
    plt.close()

    # 后处理前后
    pp_pred, pp_rows, pp_agg = eval_set(model, tx, ty, use_pp=True)
    print("最佳模型 + 后处理:", json.dumps(pp_agg, ensure_ascii=False))

    weak_path = OUT / "unet_weak.pt"
    weak = load_model(weak_path if weak_path.exists() else OUT / "unet_epoch2.pt")
    w_pred, w_rows, w_agg = eval_set(weak, tx, ty)
    w_pp_pred, w_pp_rows, w_pp_agg = eval_set(weak, tx, ty, use_pp=True)
    print("2 epoch 模型:", json.dumps(w_agg, ensure_ascii=False))
    print("2 epoch 模型 + 后处理:", json.dumps(w_pp_agg, ensure_ascii=False))

    # 失败案例图（弱模型）
    wpos = [r for r in w_rows if r["has_tumor"]]
    wfail = [r["i"] for r in sorted(wpos, key=lambda r: r["tumor_dice"])[:8]]
    plot_examples(tx, ty, w_pred, w_rows, wfail, "fig_failures_weak.png", ncol=4)

    # 保存
    np.savez_compressed(OUT / "test_pred.npz", pred=pred, pp_pred=pp_pred,
                        weak_pred=w_pred, weak_pp_pred=w_pp_pred)
    summary = {
        "test_best": agg, "test_best_pp": pp_agg,
        "test_weak": w_agg, "test_weak_pp": w_pp_agg,
        "per_case": rows, "per_case_weak": w_rows,
        "pp_changes": {
            "strong_before": agg["tumor_dice"], "strong_after": pp_agg["tumor_dice"],
            "weak_before": w_agg["tumor_dice"], "weak_after": w_pp_agg["tumor_dice"],
            "weak_liver_before": w_agg["liver_dice"], "weak_liver_after": pp_agg["liver_dice"],
        },
        "mid_case": mid,
    }
    json.dump(summary, open(OUT / "eval_summary.json", "w"), indent=2, ensure_ascii=False)
    print("saved", OUT / "eval_summary.json")


if __name__ == "__main__":
    main()
