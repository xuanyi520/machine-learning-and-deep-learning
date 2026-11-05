"""补充实验：10-D 落盘、10-G 用验证集、10-C 最大 batch 与累积训练准确率。"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"E:\AI学习\模块10_迁移学习与训练技巧\_exp")
import common as C
from common import DEVICE, build_resnet18, evaluate, load_arrays, make_loader, norm, set_seed, train

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}


def dump():
    json.dump(out, open(os.path.join(HERE, "results_supp.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def cosine_warmup(opt, total, warm_frac=0.1):
    warm = max(1, int(total * warm_frac))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---------------- 10-D 落盘
imgs, labels, lm = load_arrays("bloodmnist", "train")
timgs, tlabels, _ = load_arrays("bloodmnist", "test")
idx = C.stratified_subset(labels, 3000, seed=0)
d_out = {}
for norm_kind in ["bn", "gn"]:
    for bs in [1, 2, 8, 32]:
        set_seed(42)
        m = C.SmallCNN(len(lm), in_ch=3, norm=norm_kind).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        tl = make_loader(imgs[idx], labels[idx], size=28, train=True, bs=bs)
        te = make_loader(timgs, tlabels, size=28, bs=256)
        t0 = time.time()
        train(m, tl, te, epochs=6, optimizer=opt)
        a, _, _, _ = evaluate(m, te)
        d_out[f"{norm_kind}_bs{bs}"] = {"acc": float(a), "sec": round(time.time() - t0, 1)}
        print(f"[D] {norm_kind} bs={bs} {a:.4f}", flush=True)
        out["10d"] = d_out
        dump()

# ---------------- 10-C 最大 batch
from torchvision import models


def make_model(nclass=8):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, nclass)
    return m.to(DEVICE)


def max_batch(dtype):
    ok = 0
    for b in [256, 384, 512, 640, 768, 896, 1024]:
        try:
            m = make_model()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            x = torch.randn(b, 3, 128, 128, device=DEVICE)
            y = torch.randint(0, 8, (b,), device=DEVICE)
            with torch.autocast("cuda", dtype=dtype, enabled=(dtype is not None)):
                loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            ok = b
            del m, opt, x, y, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return ok


out["10c_max_batch_fp32"] = max_batch(None)
out["10c_max_batch_fp16"] = max_batch(torch.float16)
print("[C] max batch", out, flush=True)
dump()

# ---------------- 10-C 梯度累积训练
Xtr, Ytr = imgs[:5000], labels[:5000]
te = make_loader(timgs, tlabels, size=28, bs=256)
accs = {}
for tag, bs, accum in [("真实 batch 256", 256, 1), ("batch 128 x 累积 2", 128, 2)]:
    set_seed(42)
    m = build_resnet18(len(lm), pretrained=False).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = make_loader(Xtr, Ytr, size=28, train=True, bs=bs)
    train(m, tl, te, epochs=3, optimizer=opt, accum=accum)
    a, _, _, _ = evaluate(m, te)
    accs[tag] = float(a)
    print(f"[C] {tag}: {a:.4f}", flush=True)
out["10c_accum_acc"] = accs
dump()

# ---------------- 10-G 用验证集
vimgs, vlabels, _ = load_arrays("bloodmnist", "val")
set_seed(42)
m = build_resnet18(len(lm), pretrained=True).to(DEVICE)
for n_, p in m.named_parameters():
    p.requires_grad = n_.startswith(("layer3", "layer4", "fc"))


def g(prefix, lr):
    return {"params": [p for n_, p in m.named_parameters()
                       if p.requires_grad and n_.startswith(prefix)], "lr": lr}


opt = torch.optim.AdamW([g("layer3", 3e-5), g("layer4", 1e-4), g("fc", 1e-3)],
                        lr=1e-3, weight_decay=1e-4)
tl = make_loader(imgs[idx], labels[idx], size=28, train=True, bs=128)
vl = make_loader(vimgs, vlabels, size=28, bs=256)
sched = cosine_warmup(opt, len(tl) * 20)
crit = nn.CrossEntropyLoss()
hist, best, bad, stopped = [], 1e9, 0, None
for ep in range(20):
    m.train()
    for xb, yb in tl:
        xb = norm(xb.to(DEVICE)); yb = yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = crit(m(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
    va, vl_, _, _ = evaluate(m, vl, crit)
    hist.append({"epoch": ep + 1, "lr": opt.param_groups[-1]["lr"],
                 "val_loss": float(vl_), "val_acc": float(va)})
    print(f"[G] ep{ep+1} lr={hist[-1]['lr']:.3e} val_loss={vl_:.4f}", flush=True)
    if vl_ < best - 1e-4:
        best, bad = vl_, 0
    else:
        bad += 1
        if bad >= 5:
            stopped = ep + 1
            break
out["10g"] = {"hist": hist, "stop_epoch": stopped,
              "best_epoch": int(np.argmin([h["val_loss"] for h in hist])) + 1}
dump()
print("SUPP DONE")
