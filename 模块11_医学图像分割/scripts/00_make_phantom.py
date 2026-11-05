"""生成合成的腹部 CT 样数据，用来在没有 LiTS 的情况下跑通分割全流程。

真值标签：0 背景 / 1 肝脏 / 2 肿瘤。
输出：E:/AI学习/模块11_医学图像分割/outputs/phantom.npz
"""
import numpy as np
from scipy.ndimage import gaussian_filter

OUT = r"E:\AI学习\模块11_医学图像分割\outputs\phantom.npz"


def ellipse_mask(yy, xx, cy, cx, ry, rx, ang=0.0, warp=None):
    dy, dx = yy - cy, xx - cx
    xr = dx * np.cos(ang) + dy * np.sin(ang)
    yr = -dx * np.sin(ang) + dy * np.cos(ang)
    r = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    if warp is not None:
        theta = np.arctan2(yr, xr)
        r = r / warp(theta)
    return r <= 1.0, r


def make_case(rng, size=256, hard=False, has_tumor=None, anatomy_scale=None):
    k = size / 256.0 if anatomy_scale is None else anatomy_scale
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = size / 2 + rng.normal(0, 4 * k)
    cx = size / 2 + rng.normal(0, 4 * k)
    ry, rx = rng.uniform(104, 116) * k, rng.uniform(92, 104) * k

    img = np.full((size, size), -1000.0, np.float32)
    body, _ = ellipse_mask(yy, xx, cy, cx, ry, rx)
    img[body] = 40.0 + rng.normal(0, 3)

    # 肝脏：偏图像左侧（病人右侧），轮廓做低频扰动，避免变成标准椭圆
    lcy = cy + rng.uniform(-8, 8) * k
    lcx = cx - rng.uniform(34, 44) * k
    ar, br = rng.uniform(48, 60) * k, rng.uniform(62, 78) * k
    theta = rng.uniform(-0.25, 0.25)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    amp = 0.05 if not hard else 0.08
    warp = lambda t: 1 + amp * np.sin(3 * t + p1) + 0.04 * np.sin(5 * t + p2)
    liver, _ = ellipse_mask(yy, xx, lcy, lcx, ar, br, theta, warp)
    liver &= body

    # 肝内血管：亮结构，是模型最常见的假阳性来源
    vessel = np.zeros_like(img, bool)
    for _ in range(rng.integers(3, 7)):
        vy, vx = lcy + rng.uniform(-40, 40) * k, lcx + rng.uniform(-50, 50) * k
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(25, 70) * k
        for t in np.linspace(-length / 2, length / 2, 60):
            py = int(round(vy + t * np.sin(ang)))
            px = int(round(vx + t * np.cos(ang)))
            if 0 <= py < size and 0 <= px < size:
                rr = max(1, int(rng.integers(1, 3) * k))
                vessel[max(0, py - rr):py + rr + 1, max(0, px - rr):px + rr + 1] |= True
    vessel &= liver

    # 肿瘤：1-3 个低密度灶，边界用高斯模糊软化
    if has_tumor is None:
        has_tumor = rng.random() < 0.85
    tumor = np.zeros_like(img, bool)
    tumor_soft = np.zeros_like(img)
    if has_tumor:
        n_tum = rng.choice([1, 1, 1, 2], p=[0.55, 0.2, 0.15, 0.1])
        # 只允许在肝脏内部找位置
        ly, lx = np.where(liver)
        for _ in range(n_tum):
            pick = rng.integers(0, len(ly))
            tcy, tcx = ly[pick], lx[pick]
            tr = (rng.uniform(5, 16) if not hard else rng.uniform(4, 10)) * k
            q1 = rng.uniform(0, 2 * np.pi)
            _, r = ellipse_mask(yy, xx, tcy, tcx, tr * rng.uniform(0.8, 1.2), tr,
                                rng.uniform(0, np.pi),
                                lambda t: 1 + 0.15 * np.sin(2 * t + q1))
            m = r <= 1.0
            tumor |= m
            tumor_soft = np.maximum(tumor_soft, np.clip(1.0 - r, 0, 1) ** 0.5)
        tumor &= liver
        tumor_soft = gaussian_filter(tumor_soft * tumor, 1.5 * k)
        tumor_soft = tumor_soft / (tumor_soft.max() + 1e-8)
    else:
        tumor_soft[:] = 0

    # 灰度：肝脏带轻微梯度，肿瘤比肝低 8-22 HU（低对比）
    liver_tissue = 62 + 10 * (xx - lcx) / (br + 1e-6) + 6 * (yy - lcy) / (ar + 1e-6)
    img[liver] = liver_tissue[liver] + rng.normal(0, 3)
    img[body & ~liver] = 45 + rng.normal(0, 3) 
    img[vessel] = rng.uniform(88, 120)
    delta = rng.uniform(8, 22) if not hard else rng.uniform(5, 12)
    img[tumor] = liver_tissue[tumor] - delta
    # 软边界：把肿瘤边缘的强度过渡做出来
    edge = gaussian_filter(tumor.astype(np.float32), 1.2 * k)
    img = img * (1 - 0.35 * edge) + (liver_tissue - delta) * 0.35 * edge

    # 脾脏：右侧的干扰结构，和肝脏灰度接近
    scy, scx = cy + rng.uniform(-10, 10) * k, cx + rng.uniform(32, 42) * k
    spleen, _ = ellipse_mask(yy, xx, scy, scx, rng.uniform(24, 32) * k, rng.uniform(28, 36) * k)
    spleen &= body
    img[spleen] = rng.uniform(42, 58) + rng.normal(0, 3)

    # 平滑 + 噪声 + 偏置场
    img = gaussian_filter(img, 0.8 * k)
    bias = 1 + 0.06 * gaussian_filter(rng.normal(0, 1, (size, size)), 25 * k)
    bias = bias / bias.mean()
    img = img * bias + rng.normal(0, rng.uniform(8, 15), (size, size))

    mask = np.zeros((size, size), np.uint8)
    mask[liver] = 1
    mask[tumor] = 2
    return img.astype(np.float32), mask


def build(n, seed, hard_ratio=0.25, tumor_ratio=0.85):
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    for i in range(n):
        hard = rng.random() < hard_ratio
        has_tumor = rng.random() < tumor_ratio
        im, mk = make_case(rng, hard=hard, has_tumor=has_tumor)
        imgs.append(im)
        masks.append(mk)
    return np.stack(imgs), np.stack(masks)


if __name__ == "__main__":
    tr_x, tr_y = build(600, 2026, hard_ratio=0.15)
    va_x, va_y = build(100, 7, hard_ratio=0.35)
    te_x, te_y = build(100, 99, hard_ratio=0.6)
    np.savez_compressed(OUT, tr_x=tr_x, tr_y=tr_y, va_x=va_x, va_y=va_y,
                        te_x=te_x, te_y=te_y)
    print("saved", OUT)
    print("train", tr_x.shape, tr_y.shape, "val", va_x.shape, "test", te_x.shape)
    for name, y in [("train", tr_y), ("val", va_y), ("test", te_y)]:
        fg = (y > 0).mean()
        tum = (y == 2).mean()
        print(f"{name}: foreground={fg:.4f} tumor={tum:.4f} "
              f"tumor-positive-cases={(y == 2).any(axis=(1, 2)).mean():.2f}")
    print("HU range", tr_x.min(), tr_x.max())
