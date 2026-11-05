"""模块 10 参考答案：全部练习的实测脚本。

在 _exp/common.py 的基础上跑，输出到 _ans_run/results_mod10.json。
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, r"E:\AI学习\模块10_迁移学习与训练技巧\_exp")
import common as C
from common import (DEVICE, build_resnet18, count_trainable, evaluate, freeze_to,
                    load_arrays, make_loader, norm, save, set_seed, stratified_subset,
                    train, FocalLoss)

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results_mod10.json")
res = {}


def dump():
    with open(RES, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def cosine_warmup(opt, total, warm_frac=0.1):
    warm = max(1, int(total * warm_frac))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def acc_of(m, loader):
    a, _, _, _ = evaluate(m, loader)
    return float(a)


# ============================================================ 10-A
def ex_10a():
    imgs, labels, lm = load_arrays("bloodmnist", "train")
    timgs, tlabels, _ = load_arrays("bloodmnist", "test")
    NC, SIZE, EPOCHS, BS = len(lm), 28, 15, 128
    test_loader = make_loader(timgs, tlabels, size=SIZE, bs=256)

    def run(Xtr, Ytr, pretrained, keep, lr_groups):
        set_seed(42)
        m = build_resnet18(NC, pretrained=pretrained)
        if keep is not None:
            freeze_to(m, keep)
        groups = []
        for prefix, lr in lr_groups:
            ps = [p for n, p in m.named_parameters()
                  if p.requires_grad and n.startswith(prefix)]
            if ps:
                groups.append({"params": ps, "lr": lr})
        opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-4)
        tl = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=BS)
        sched = cosine_warmup(opt, len(tl) * EPOCHS)
        train(m, tl, test_loader, epochs=EPOCHS, optimizer=opt, sched=sched)
        return acc_of(m, test_loader)

    STRATS = {
        "scratch": (False, None, [("", 1e-3)]),
        "head_only": (True, ["fc"], [("fc", 1e-3)]),
        "partial_l4": (True, ["layer4", "fc"], [("layer4", 2e-4), ("fc", 1e-3)]),
        "full_diff": (True, None, [("layer1", 1e-5), ("layer2", 1e-5),
                                   ("layer3", 3e-5), ("layer4", 1e-4), ("fc", 1e-3)]),
    }
    # 参数量核对
    for k, (pre, keep, _) in STRATS.items():
        m = build_resnet18(NC, pretrained=pre)
        if keep is not None:
            freeze_to(m, keep)
        res.setdefault("10a_params", {})[k] = count_trainable(m)

    rows = []
    for n in [500, 1000, 2000, 5000]:
        idx = stratified_subset(labels, n, seed=0)
        Xtr, Ytr = imgs[idx], labels[idx]
        row = {"n": int(len(idx))}
        t0 = time.time()
        for k, (pre, keep, lrs) in STRATS.items():
            row[k] = run(Xtr, Ytr, pre, keep, lrs)
            print(f"[10A] n={len(idx)} {k:11s} {row[k]:.4f}", flush=True)
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        res["10a"] = rows
        dump()


# ============================================================ 10-B / 10-E
def derma_setup(size=28):
    imgs, labels, lm = load_arrays("dermamnist", "train")
    timgs, tlabels, _ = load_arrays("dermamnist", "test")
    return imgs, labels, lm, timgs, tlabels


def derma_run(imgs, labels, lm, timgs, tlabels, aug, epochs=12, loss_kind="ce",
              gamma=2.0, ls=0.0):
    set_seed(42)
    NC, SIZE, BS = len(lm), 28, 128
    m = C.SmallCNN(NC, in_ch=3).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = make_loader(imgs, labels, size=SIZE, train=True, bs=BS)
    te = make_loader(timgs, tlabels, size=SIZE, bs=256)
    if loss_kind == "focal":
        crit = FocalLoss(gamma=gamma, ls=ls)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=ls)
    train(m, tl, te, epochs=epochs, optimizer=opt, criterion=crit, aug=aug)
    _, _, probs, ys = evaluate(m, te)
    p = probs.argmax(1).numpy()
    y = ys.numpy()
    from sklearn.metrics import balanced_accuracy_score, recall_score
    out = {
        "acc": float((p == y).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y, p)),
        "recall_3": float(recall_score(y, p, labels=[3], average="macro", zero_division=0)),
        "recall_6": float(recall_score(y, p, labels=[6], average="macro", zero_division=0)),
    }
    return out


def ex_10b():
    imgs, labels, lm, timgs, tlabels = derma_setup()
    AUGS = {
        "none": {},
        "safe": dict(rot=10, trans=0.08, scale=0.1),
        "safe+hflip": dict(rot=10, trans=0.08, scale=0.1, hflip=0.5),
        "safe+vflip": dict(rot=10, trans=0.08, scale=0.1, vflip=0.5),
        "color": dict(jitter=0.3),
    }
    out = {}
    for k, aug in AUGS.items():
        out[k] = derma_run(imgs, labels, lm, timgs, tlabels, aug)
        print(f"[10B] {k:11s} {out[k]['acc']:.4f}", flush=True)
        res["10b"] = out
        dump()


def ex_10e():
    imgs, labels, lm, timgs, tlabels = derma_setup()
    out = {}
    out["gamma0"] = derma_run(imgs, labels, lm, timgs, tlabels, {}, loss_kind="ce")
    for g in [1, 2, 3]:
        out[f"gamma{g}"] = derma_run(imgs, labels, lm, timgs, tlabels, {},
                                     loss_kind="focal", gamma=g)
        print(f"[10E] gamma={g} {out[f'gamma{g}']}", flush=True)
        res["10e"] = out
        dump()
    res["10e"] = out
    dump()


# ============================================================ 10-C
def ex_10c():
    from torchvision import models

    def make_model():
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(512, 8)
        return m.to(DEVICE)

    xb = torch.randn(128, 3, 128, 128, device=DEVICE)
    yb = torch.randint(0, 8, (128,), device=DEVICE)

    def bench(dtype, batch):
        m = make_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        x, y = xb[:batch], yb[:batch]
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=(dtype is not None)):
                loss = F.cross_entropy(m(x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n = 30
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=(dtype is not None)):
                loss = F.cross_entropy(m(x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        del m, opt
        torch.cuda.empty_cache()
        return {"ms": ms, "mem_MB": mem, "img_s": batch / (ms / 1000)}

    # 最大 batch 探测（FP32 / AMP）
    def max_batch(dtype):
        ok = None
        for b in [64, 128, 256, 384, 512]:
            try:
                m = make_model()
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                x, y = xb[:min(b, 128)].repeat((b + min(b, 128) - 1) // min(b, 128), 1, 1, 1)[:b], \
                    yb[:min(b, 128)].repeat((b + min(b, 128) - 1) // min(b, 128))[:b]
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

    out = {"max_batch_fp32": max_batch(None), "max_batch_fp16": max_batch(torch.float16)}
    out["fp32_b64"] = bench(None, 64)
    out["fp16_b64"] = bench(torch.float16, 64)
    out["bf16_b64"] = bench(torch.bfloat16, 64)
    print("[10C]", out, flush=True)
    res["10c"] = out
    dump()

    # 梯度累积等价性：真 batch 32 vs 4 x batch 8
    accs = {}
    for mode in [False, True]:
        set_seed(0)
        m = make_model()
        m.train(mode)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(xb[:32]), yb[:32]).backward()
        g_full = [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]

        set_seed(0)
        m = make_model()
        m.train(mode)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        for i in range(4):
            sl = slice(i * 8, (i + 1) * 8)
            (F.cross_entropy(m(xb[sl]), yb[sl]) / 4).backward()
        g_acc = [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]
        num = sum(float(((a - b) ** 2).sum()) for a, b in zip(g_full, g_acc)) ** 0.5
        den = sum(float((a ** 2).sum()) for a in g_full) ** 0.5
        accs["eval" if not mode else "train"] = num / den
    out["accum_rel_err"] = accs
    print("[10C accum]", accs, flush=True)
    res["10c"] = out
    dump()


# ============================================================ 10-D
def ex_10d():
    imgs, labels, lm = load_arrays("bloodmnist", "train")
    timgs, tlabels, _ = load_arrays("bloodmnist", "test")
    idx = stratified_subset(labels, 3000, seed=0)
    out = {}
    for norm_kind in ["bn", "gn"]:
        for bs in [1, 2, 8, 32]:
            set_seed(42)
            m = C.SmallCNN(len(lm), in_ch=3, norm=norm_kind).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            tl = make_loader(imgs[idx], labels[idx], size=28, train=True, bs=bs)
            te = make_loader(timgs, tlabels, size=28, bs=256)
            t0 = time.time()
            train(m, tl, te, epochs=6, optimizer=opt)
            a = acc_of(m, te)
            out[f"{norm_kind}_bs{bs}"] = {"acc": a, "sec": round(time.time() - t0, 1)}
            print(f"[10D] {norm_kind} bs={bs} {a:.4f}", flush=True)
            res["10d"] = out
            dump()


# ============================================================ 10-F
def ex_10f():
    imgs, labels, lm = load_arrays("bloodmnist", "train")
    timgs, tlabels, _ = load_arrays("bloodmnist", "test")
    NC, SIZE, EPOCHS, BS = len(lm), 64, 8, 128
    idx = stratified_subset(labels, 2000, seed=0)
    Xtr, Ytr = imgs[idx], labels[idx]
    te = make_loader(timgs, tlabels, size=SIZE, bs=256)
    SOFT = dict(rot=8, trans=0.05, scale=0.05)

    @torch.no_grad()
    def tta(m):
        m.eval()
        P, Y = [], []
        for xb, yb in te:
            xb = xb.to(DEVICE)
            p = m(norm(xb)).softmax(1) + m(norm(torch.flip(xb, dims=[3]))).softmax(1)
            P.append((p / 2).float().cpu())
            Y.append(yb)
        P, Y = torch.cat(P), torch.cat(Y)
        return float((P.argmax(1) == Y).float().mean()), float(
            f1_score(Y.numpy(), P.argmax(1).numpy(), average="macro"))

    def run(pretrained, aug, sched_on, ls=0.0, use_tta=False, seed=42):
        set_seed(seed)
        m = build_resnet18(NC, pretrained=pretrained)
        if pretrained:
            def g(prefix, lr):
                return {"params": [p for n, p in m.named_parameters()
                                   if n.startswith(prefix)], "lr": lr}
            opt = torch.optim.AdamW([g("layer1", 1e-5), g("layer2", 1e-5),
                                     g("layer3", 3e-5), g("layer4", 1e-4),
                                     g("fc", 1e-3)], lr=1e-3, weight_decay=1e-4)
        else:
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        tl = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=BS)
        sched = cosine_warmup(opt, len(tl) * EPOCHS) if sched_on else None
        train(m, tl, te, epochs=EPOCHS, optimizer=opt, sched=sched,
              criterion=nn.CrossEntropyLoss(label_smoothing=ls), aug=aug)
        if use_tta:
            a, f = tta(m)
        else:
            a, _, probs, ys = evaluate(m, te)
            f = f1_score(ys.numpy(), probs.argmax(1).numpy(), average="macro")
        return {"acc": float(a), "macro_f1": float(f)}

    rows = []
    rows.append({"tag": "1 基线(从头训,常数LR,无增强)", **run(False, {}, False)})
    rows.append({"tag": "2 +安全增强", **run(False, SOFT, False)})
    rows.append({"tag": "3 +Cosine+warmup", **run(False, SOFT, True)})
    rows.append({"tag": "4 +预训练(差分LR)", **run(True, SOFT, True)})
    rows.append({"tag": "5 +LabelSmoothing 0.1", **run(True, SOFT, True, ls=0.1)})
    rows.append({"tag": "6 +TTA", **run(True, SOFT, True, ls=0.1, use_tta=True)})
    for r in rows:
        print(f"[10F] {r['tag']:32s} {r['acc']:.4f}", flush=True)
    res["10f"] = rows
    dump()

    seeds = []
    for s in [0, 1, 2]:
        r = run(True, SOFT, True, ls=0.1, use_tta=True, seed=s)
        seeds.append(r["acc"])
        print(f"[10F seed] {s} {r['acc']:.4f}", flush=True)
    res["10f_seeds"] = {"accs": seeds, "mean": float(np.mean(seeds)),
                        "std": float(np.std(seeds, ddof=1))}
    dump()


# ============================================================ 10-G
def ex_10g():
    imgs, labels, lm = load_arrays("bloodmnist", "train")
    timgs, tlabels, _ = load_arrays("bloodmnist", "test")
    NC, SIZE, BS = len(lm), 28, 128
    idx = stratified_subset(labels, 2000, seed=0)
    set_seed(42)
    m = build_resnet18(NC, pretrained=True).to(DEVICE)
    # 只解冻 layer3 以后 + 差分 LR
    for n_, p in m.named_parameters():
        p.requires_grad = n_.startswith(("layer3", "layer4", "fc"))
    def g(prefix, lr):
        return {"params": [p for n_, p in m.named_parameters()
                           if p.requires_grad and n_.startswith(prefix)], "lr": lr}
    opt = torch.optim.AdamW([g("layer3", 3e-5), g("layer4", 1e-4), g("fc", 1e-3)],
                            lr=1e-3, weight_decay=1e-4)
    tl = make_loader(imgs[idx], labels[idx], size=SIZE, train=True, bs=BS)
    te = make_loader(timgs, tlabels, size=SIZE, bs=256)
    sched = cosine_warmup(opt, len(tl) * 20)
    crit = nn.CrossEntropyLoss()
    hist = []
    best = 1e9
    bad = 0
    patience = 5
    stopped = None
    for ep in range(20):
        m.train()
        for xb, yb in tl:
            xb = norm(xb.to(DEVICE)); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        va, vl, _, _ = evaluate(m, te, crit)
        hist.append({"epoch": ep + 1, "lr": opt.param_groups[-1]["lr"],
                     "val_loss": vl, "val_acc": va})
        print(f"[10G] ep{ep+1} lr={opt.param_groups[-1]['lr']:.2e} "
              f"val_loss={vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best = vl
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                stopped = ep + 1
                break
    res["10g"] = {"hist": hist, "stop_epoch": stopped, "best_epoch": int(np.argmin([h["val_loss"] for h in hist])) + 1}
    dump()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    todo = a.only.split(",") if a.only else ["a", "d", "b", "e", "c", "f", "g"]
    if "a" in todo: ex_10a()
    if "b" in todo: ex_10b()
    if "c" in todo: ex_10c()
    if "d" in todo: ex_10d()
    if "e" in todo: ex_10e()
    if "f" in todo: ex_10f()
    if "g" in todo: ex_10g()
    print("ALL DONE")
