"""补充二：梯度累积训练准确率 + 早停（验证集）。最大 batch 用主实验的 512 探测值。"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"E:\AI学习\模块10_迁移学习与训练技巧\_exp")
import common as C
from common import (DEVICE, build_resnet18, evaluate, load_arrays, make_loader,
                    norm, set_seed, stratified_subset, train)

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "results_supp.json")
out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def dump():
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def cosine_warmup(opt, total, warm_frac=0.1):
    warm = max(1, int(total * warm_frac))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


imgs, labels, lm = load_arrays("bloodmnist", "train")
timgs, tlabels, _ = load_arrays("bloodmnist", "test")
vimgs, vlabels, _ = load_arrays("bloodmnist", "val")
idx = stratified_subset(labels, 5000, seed=0)

accs = {}
for tag, bs, accum in [("真实 batch 256", 256, 1), ("batch 128 x 累积 2", 128, 2)]:
    set_seed(42)
    m = build_resnet18(len(lm), pretrained=False).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = make_loader(imgs[idx], labels[idx], size=28, train=True, bs=bs)
    te = make_loader(timgs, tlabels, size=28, bs=256)
    train(m, tl, te, epochs=3, optimizer=opt, accum=accum)
    a, _, _, _ = evaluate(m, te)
    accs[tag] = float(a)
    print(f"[C] {tag}: {a:.4f}", flush=True)
    out["10c_accum_acc"] = accs
    dump()

# 早停，用验证集
idx = stratified_subset(labels, 2000, seed=0)
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
    va, vloss, _, _ = evaluate(m, vl, crit)
    hist.append({"epoch": ep + 1, "lr": opt.param_groups[-1]["lr"],
                 "val_loss": float(vloss), "val_acc": float(va)})
    print(f"[G] ep{ep+1} lr={hist[-1]['lr']:.3e} val_loss={vloss:.4f}", flush=True)
    if vloss < best - 1e-4:
        best, bad = vloss, 0
    else:
        bad += 1
        if bad >= 5:
            stopped = ep + 1
            break
out["10g"] = {"hist": hist, "stop_epoch": stopped,
              "best_epoch": int(np.argmin([h["val_loss"] for h in hist])) + 1}
dump()
print("SUPP2 DONE")
