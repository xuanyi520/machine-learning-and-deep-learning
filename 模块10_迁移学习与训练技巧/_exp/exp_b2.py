"""实验 B2：合成"定位"任务——标签直接依赖方向，翻转必然冲突。
背景用 BloodMNIST 真实细胞图，在四个角之一贴一个亮块，标签=角的位置。
"""
import numpy as np, torch, torch.nn as nn
from common import *

N, SIZE = 4000, 28
imgs, _, _ = load_arrays('bloodmnist', 'train')
rng = np.random.RandomState(0)

def make_corner_dataset(n, seed=1):
    r = np.random.RandomState(seed)
    idx = r.choice(len(imgs), n, replace=True)
    X = imgs[idx].copy()                       # (n,28,28,3)
    y = r.randint(0, 4, n)
    p = 4                                      # 亮块边长
    for i in range(n):
        c = y[i]
        r0 = 1 if c < 2 else SIZE - p - 1      # 上/下
        c0 = 1 if c % 2 == 0 else SIZE - p - 1 # 左/右
        X[i, r0:r0 + p, c0:c0 + p, :] = 255
    return X, y

Xtr, Ytr = make_corner_dataset(N, seed=1)
Xte, Yte = make_corner_dataset(1000, seed=2)
print('corner task', Xtr.shape, np.bincount(Ytr), flush=True)

out = {}
for kind in ['none', 'hflip', 'vflip', 'safe']:
    set_seed(42)
    m = SmallCNN(4, 3)
    tr = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=128)
    te = make_loader(Xte, Yte, size=SIZE, bs=256)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    train(m, tr, te, epochs=8, optimizer=opt, aug=build_aug(kind))
    acc, _, probs, ys = evaluate(m, te)
    # 看混淆矩阵里是否左右/上下互串
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ys.numpy(), probs.argmax(1).numpy(), labels=[0, 1, 2, 3])
    out[kind] = {'acc': float(acc), 'cm': cm.tolist()}
    print(f'[四角定位] {kind:10s} acc={acc:.4f}  cm={cm.tolist()}', flush=True)
    save('b2_corner.json', out)
