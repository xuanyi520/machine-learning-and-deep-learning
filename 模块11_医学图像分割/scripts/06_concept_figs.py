"""概念图：Dice-IoU 曲线、位移敏感性、patch 采样对比。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
FIG = ROOT / "figures"

# 1. Dice-IoU
d = np.linspace(0.01, 0.99, 200)
plt.figure(figsize=(4.6, 3.8))
plt.plot(d, d / (2 - d), color="#2b6cb0")
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
plt.xlabel("Dice"); plt.ylabel("IoU"); plt.grid(alpha=.3)
plt.title("Dice 与 IoU 的换算：IoU = Dice/(2-Dice)")
plt.tight_layout(); plt.savefig(FIG / "fig_dice_iou.png", dpi=140); plt.close()

# 2. 位移敏感性
plt.figure(figsize=(5, 3.8))
for r, c in [(5, "#c53030"), (10, "#dd6b20"), (20, "#2f855a"), (30, "#2b6cb0")]:
    xs = np.arange(1, 6)
    vals = []
    for s in xs:
        yy, xx = np.mgrid[-40:41, -40:41]
        A = yy ** 2 + xx ** 2 <= r ** 2
        B = yy ** 2 + (xx - s) ** 2 <= r ** 2
        vals.append(2 * (A & B).sum() / (A.sum() + B.sum()))
    plt.plot(xs, vals, "o-", label=f"半径 {r}px", color=c)
plt.xlabel("整体平移（像素）"); plt.ylabel("Dice"); plt.ylim(0.4, 1.02)
plt.legend(); plt.grid(alpha=.3); plt.title("同样平移 2px，小病灶掉得更狠")
plt.tight_layout(); plt.savefig(FIG / "fig_shift.png", dpi=140); plt.close()

# 3. 随机 patch vs 正负采样
d = np.load(ROOT / "outputs" / "phantom.npz")
y = d["va_y"]
pos = np.where((y == 2).any(axis=(1, 2)))[0]
rng = np.random.default_rng(3)
fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
for j in range(6):
    k = rng.choice(pos)
    y0, x0 = rng.integers(0, 193, 2)
    p = y[k, y0:y0 + 64, x0:x0 + 64]
    axes[0, j].imshow(p, cmap="viridis", vmin=0, vmax=2)
    axes[0, j].set_title("含肿瘤" if (p == 2).any() else "纯背景", fontsize=8)
    axes[0, j].axis("off")
# 正样本：保证含肿瘤
for j in range(6):
    k = rng.choice(pos)
    ty, tx = np.argwhere(y[k] == 2)[rng.integers(0, (y[k] == 2).sum())]
    y0 = int(np.clip(ty - rng.integers(10, 54), 0, 192))
    x0 = int(np.clip(tx - rng.integers(10, 54), 0, 192))
    axes[1, j].imshow(y[k, y0:y0 + 64, x0:x0 + 64], cmap="viridis", vmin=0, vmax=2)
    axes[1, j].set_title("pos 采样", fontsize=8)
    axes[1, j].axis("off")
plt.suptitle("上：随机切 64×64 patch（常常整块没病灶）  下：按标签正采样")
plt.tight_layout(); plt.savefig(FIG / "fig_patch_sampling.png", dpi=140); plt.close()
print("done")
