"""实验 A：数据量 × 微调策略（BloodMNIST, ResNet18 @28x28）。"""
import numpy as np, torch, torch.nn as nn, time
from common import *

import sys
NAME, SIZE, EPOCHS, BS = 'bloodmnist', int(sys.argv[1]), int(sys.argv[2]), 128
imgs, labels, lm = load_arrays(NAME, 'train')
timgs, tlabels, _ = load_arrays(NAME, 'test')
NC = len(lm)
test_loader = make_loader(timgs, tlabels, size=SIZE, bs=256)


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
    hist, el, _ = train(m, tl, test_loader, epochs=EPOCHS, optimizer=opt)
    acc, _, _, _ = evaluate(m, test_loader)
    return float(acc), count_trainable(m), el


STRATS = {
    'scratch':       (False, None, [('', 1e-3)]),
    'head_only':     (True, ['fc'], [('fc', 1e-3)]),
    'partial_l4':    (True, ['layer4', 'fc'], [('layer4', 2e-4), ('fc', 1e-3)]),
    'full_diff':     (True, None, [('layer1', 1e-5), ('layer2', 1e-5), ('layer3', 3e-5),
                                   ('layer4', 1e-4), ('fc', 1e-3)]),
}

out = []
for n in [200, 500, 1000, 2000, 5000, len(labels)]:
    idx = stratified_subset(labels, n, seed=0) if n < len(labels) else np.arange(len(labels))
    Xtr, Ytr = imgs[idx], labels[idx]
    row = {'n': int(len(idx))}
    print(f'=== n={len(idx)} ===', flush=True)
    for k, (pre, keep, lrs) in STRATS.items():
        a, p, t = run(Xtr, Ytr, pre, keep, lrs)
        row[k] = a; row[k + '_params'] = p
        print(f'  {k:12s} acc={a:.4f}  trainable={p:>9,}  {t:.0f}s', flush=True)
    out.append(row)
    save(f'a2_sweep_{SIZE}.json', out)
