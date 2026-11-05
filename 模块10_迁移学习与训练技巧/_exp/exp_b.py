"""实验 B：增强的安全性。
B1 OrganAMNIST 左右定位任务（hflip 会直接翻转标签）
B2 OrganAMNIST 器官识别 3 类（hflip 不改变标签）
B3 DermaMNIST 病灶 7 类（非定位任务）
"""
import numpy as np, torch, torch.nn as nn
from common import *

EPOCHS, BS = 10, 128


def run_cnn(Xtr, Ytr, Xte, Yte, nclass, aug_kind, epochs=EPOCHS, bs=BS, seed=42):
    set_seed(seed)
    m = SmallCNN(nclass, 3)
    tr = make_loader(Xtr, Ytr, size=28, train=True, bs=bs)
    te = make_loader(Xte, Yte, size=28, bs=256)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    hist, el, _ = train(m, tr, te, epochs=epochs, optimizer=opt, aug=build_aug(aug_kind))
    acc, _, _, _ = evaluate(m, te)
    return float(acc), el


imgs, labels, lm = load_arrays('organamnist', 'train')
timgs, tlabels, _ = load_arrays('organamnist', 'test')
LR6 = [1, 2, 4, 5, 7, 8]
mtr, mte = np.isin(labels, LR6), np.isin(tlabels, LR6)
X, Y = imgs[mtr], labels[mtr]
Xt, Yt = timgs[mte], tlabels[mte]
left = np.array([1, 4, 7])
Yside = np.isin(Y, left).astype(np.int64)
Yside_t = np.isin(Yt, left).astype(np.int64)
Yorg = np.select([np.isin(Y, [1, 2]), np.isin(Y, [4, 5]), np.isin(Y, [7, 8])], [0, 1, 2]).astype(np.int64)
Yorg_t = np.select([np.isin(Yt, [1, 2]), np.isin(Yt, [4, 5]), np.isin(Yt, [7, 8])], [0, 1, 2]).astype(np.int64)

idx = stratified_subset(Yside, 6000, seed=0)
X6, Ys6, Yo6 = X[idx], Yside[idx], Yorg[idx]
print('laterality train', X6.shape, np.bincount(Ys6), 'test', Xt.shape, np.bincount(Yside_t), flush=True)

out = {'laterality': {}, 'organ3': {}, 'derma': {}}

print('--- B1 左右定位（二分类）---', flush=True)
for kind in ['none', 'hflip', 'vflip', 'safe', 'safe+hflip']:
    a, el = run_cnn(X6, Ys6, Xt, Yside_t, 2, kind)
    out['laterality'][kind] = a
    print(f'[左右定位] {kind:10s} acc={a:.4f} ({el:.0f}s)', flush=True)
    save('b_augment.json', out)

print('--- B2 器官识别（3 类，非定位）---', flush=True)
for kind in ['none', 'hflip', 'vflip', 'safe']:
    a, el = run_cnn(X6, Yo6, Xt, Yorg_t, 3, kind)
    out['organ3'][kind] = a
    print(f'[器官3类] {kind:10s} acc={a:.4f} ({el:.0f}s)', flush=True)
    save('b_augment.json', out)

print('--- B3 皮肤病灶 7 类 ---', flush=True)
di, dl, dlm = load_arrays('dermamnist', 'train')
ti, tl, _ = load_arrays('dermamnist', 'test')
for kind in ['none', 'safe', 'safe+hflip', 'safe+vflip', 'color']:
    a, el = run_cnn(di, dl, ti, tl, len(dlm), kind, epochs=12, bs=128)
    out['derma'][kind] = a
    print(f'[皮肤病灶] {kind:10s} acc={a:.4f} ({el:.0f}s)', flush=True)
    save('b_augment.json', out)
