"""实验 D：损失函数进阶（DermaMNIST 7 类，极不平衡）。"""
import numpy as np, torch, torch.nn as nn
from common import *
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score

imgs, labels, lm = load_arrays('dermamnist', 'train')
timgs, tlabels, _ = load_arrays('dermamnist', 'test')
NC = len(lm)
print('train', np.bincount(labels), 'test', np.bincount(tlabels), flush=True)
cnt = np.bincount(labels, minlength=NC)
w = torch.tensor(cnt.sum() / (NC * cnt), dtype=torch.float32, device=DEVICE)
print('weights', [round(float(x), 2) for x in w.cpu()], flush=True)

tr = make_loader(imgs, labels, size=28, train=True, bs=128)
te = make_loader(timgs, tlabels, size=28, bs=256)

cfgs = {
    'CE': nn.CrossEntropyLoss(),
    'CE+LS0.1': nn.CrossEntropyLoss(label_smoothing=0.1),
    'CE+weight': nn.CrossEntropyLoss(weight=w),
    'Focal g=1': FocalLoss(1.0),
    'Focal g=2': FocalLoss(2.0),
    'Focal g=3': FocalLoss(3.0),
    'Focal g=2+w': FocalLoss(2.0, weight=w),
    'Focal g=2+LS': FocalLoss(2.0, ls=0.1),
}
out = {}
for name, crit in cfgs.items():
    set_seed(42)
    m = SmallCNN(NC, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    hist, el, _ = train(m, tr, te, epochs=12, criterion=crit, optimizer=opt)
    acc, _, probs, ys = evaluate(m, te)
    yp = probs.argmax(1).numpy(); yt = ys.numpy()
    rec = recall_score(yt, yp, average=None, labels=list(range(NC)), zero_division=0)
    out[name] = {'acc': float(acc),
                 'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
                 'bal_acc': float(balanced_accuracy_score(yt, yp)),
                 'recall_3': float(rec[3]), 'recall_6': float(rec[6]),
                 'recall_5': float(rec[5])}
    print(f'{name:14s} acc={acc:.4f} macroF1={out[name]["macro_f1"]:.4f} '
          f'balAcc={out[name]["bal_acc"]:.4f} rec3={rec[3]:.2f} rec6={rec[6]:.2f} ({el:.0f}s)',
          flush=True)
    save('d_loss.json', out)
