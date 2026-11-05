"""从零实现 U-Net，在合成腹部 CT 上训练 3 类分割（背景/肝脏/肿瘤）。

运行：
    python 01_train_unet.py --epochs 60
产出：
    outputs/unet_best.pt        最佳权重
    outputs/train_log.json      每个 epoch 的 loss 和 Dice
    figures/fig_training_curve.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

HU_MIN, HU_MAX = -200.0, 300.0


# ---------------------------------------------------------------- 数据
class PhantomDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = np.clip(self.x[i], HU_MIN, HU_MAX)
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)          # -> [0,1]
        img = torch.from_numpy(img[None].astype(np.float32))
        msk = torch.from_numpy(self.y[i].astype(np.int64))

        if self.augment:
            if np.random.rand() < 0.5:                     # 左右翻转
                img, msk = torch.flip(img, [-1]), torch.flip(msk, [-1])
            if np.random.rand() < 0.5:                     # 上下翻转
                img, msk = torch.flip(img, [-2]), torch.flip(msk, [-2])
            k = np.random.randint(0, 4)
            if k:
                img, msk = torch.rot90(img, k, [-2, -1]), torch.rot90(msk, k, [-2, -1])
            if np.random.rand() < 0.5:                     # 强度扰动
                img = torch.clamp(img * np.random.uniform(0.85, 1.15), 0, 1)
        return img, msk


# ---------------------------------------------------------------- 模型
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4 次下采样 / 4 次上采样，通道 (32,64,128,256)，瓶颈 512。"""

    def __init__(self, in_ch=1, num_classes=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[3] * 2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[0] * 2, c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                 # 256
        e2 = self.enc2(self.pool(e1))     # 128
        e3 = self.enc3(self.pool(e2))     # 64
        e4 = self.enc4(self.pool(e3))     # 32
        b = self.bottleneck(self.pool(e4))  # 16

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------- 损失/指标
def dice_loss(logits, target, num_classes=3, smooth=1.0):
    """前景类（1,2）的 soft Dice。logits: (B,C,H,W)，target: (B,H,W)"""
    prob = torch.softmax(logits, 1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    loss = 0.0
    for c in range(1, num_classes):
        p, t = prob[:, c], onehot[:, c]
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        loss = loss + (1 - (2 * inter + smooth) / (denom + smooth)).mean()
    return loss / (num_classes - 1)


def case_dice(pred, target, cls):
    """单张图的 hard Dice；pred/target 是 (H,W)。两边都空返回 1。"""
    p, t = (pred == cls), (target == cls)
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float(2 * (p & t).sum() / denom)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    liver, tumor, acc = [], [], []
    tumor_cases = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for i in range(x.size(0)):
            p, t = pred[i].cpu(), y[i].cpu()
            liver.append(case_dice(p, t, 1))
            acc.append(float((p == t).float().mean()))
            if (t == 2).any():
                tumor.append(case_dice(p, t, 2))
                tumor_cases += 1
            n += 1
    return {
        "liver_dice": float(np.mean(liver)),
        "tumor_dice": float(np.mean(tumor)) if tumor else float("nan"),
        "pixel_acc": float(np.mean(acc)),
        "n": n,
        "n_tumor": tumor_cases,
    }


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    d = np.load(OUT / "phantom.npz")
    tr = PhantomDataset(d["tr_x"], d["tr_y"], augment=True)
    va = PhantomDataset(d["va_x"], d["va_y"], augment=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(1, 3).to(device)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce_weight = torch.tensor([0.5, 1.0, 3.0], device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    log = []
    best = -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                ce = F.cross_entropy(logits, y, weight=ce_weight)
                loss = ce + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
        sched.step()

        ev = evaluate(model, vl, device)
        score = (ev["liver_dice"] + (0 if np.isnan(ev["tumor_dice"]) else ev["tumor_dice"])) / 2
        log.append({"epoch": ep, "loss": run_loss / len(tr), **ev})
        flag = ""
        if score > best:
            best = score
            torch.save(model.state_dict(), OUT / "unet_best.pt")
            flag = " *best"
        print(f"ep {ep:3d} loss {run_loss/len(tr):.4f} "
              f"liver {ev['liver_dice']:.4f} tumor {ev['tumor_dice']:.4f} "
              f"acc {ev['pixel_acc']:.4f}{flag}")

    # 保存最后一轮，用来看欠训练模型的失败样子
    torch.save(model.state_dict(), OUT / "unet_last.pt")
    json.dump(log, open(OUT / "train_log.json", "w"), indent=2)
    print(f"done in {time.time()-t0:.1f}s, best score {best:.4f}")


if __name__ == "__main__":
    main()
