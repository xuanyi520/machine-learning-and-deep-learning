"""实验 A5：64x64 输入下 数据量 × 策略（对比 28x28 的结论反转）。"""
import numpy as np, torch, torch.nn as nn, math
from common import *

NAME, SIZE, EPOCHS, BS = 'bloodmnist', 64, 12, 128
imgs, labels, lm = load_arrays(NAME, 'train')
timgs, tlabels, _ = load_arrays(NAME, 'test')
NC = len(lm)
test_loader = make_loader(timgs, tlabels, size=SIZE, bs=256)


def cosine_warmup(opt, total, warm_frac=0.1):
    warm = max(1, int(total * warm_frac))
    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def run(Xtr, Ytr, pretrained, keep, lr_groups):
    set_seed(42)
    m = build_resnet18(NC, pretrained=pretrained)
    if keep is not None:
        freeze_to(m, keep)
    groups = []
    for prefix, lr in lr_groups:
        ps = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if ps:
            groups.append({'params': ps, 'lr': lr})
    opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-4)
    tl = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=BS)
    sched = cosine_warmup(opt, len(tl) * EPOCHS)
    train(m, tl, test_loader, epochs=EPOCHS, optimizer=opt, sched=sched)
    acc, _, _, _ = evaluate(m, test_loader)
    return float(acc)


STRATS = {
    'scratch':    (False, None, [('', 1e-3)]),
    'head_only':  (True, ['fc'], [('fc', 1e-3)]),
    'partial_l4': (True, ['layer4', 'fc'], [('layer4', 2e-4), ('fc', 1e-3)]),
    'full_diff':  (True, None, [('layer1', 1e-5), ('layer2', 1e-5), ('layer3', 3e-5),
                                ('layer4', 1e-4), ('fc', 1e-3)]),
}

out = []
for n in [500, 2000]:
    idx = stratified_subset(labels, n, seed=0)
    Xtr, Ytr = imgs[idx], labels[idx]
    row = {'n': int(len(idx))}
    print(f'=== n={len(idx)} ===', flush=True)
    for k, (pre, keep, lrs) in STRATS.items():
        a = run(Xtr, Ytr, pre, keep, lrs)
        row[k] = a
        print(f'  {k:12s} acc={a:.4f}', flush=True)
    out.append(row)
    save('a5_sweep64.json', out)
