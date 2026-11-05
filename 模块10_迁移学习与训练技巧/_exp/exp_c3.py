"""实验 C3：梯度累积等价性 + BN vs GroupNorm（C1/C2 已在 c_amp.json / c_amp_batch.json）。"""
import numpy as np, torch, torch.nn as nn
from common import *

print('=== C3 梯度累积 ===', flush=True)
set_seed(0)
m1 = build_resnet18(8, pretrained=False).to(DEVICE).eval()
m2 = build_resnet18(8, pretrained=False).to(DEVICE)
m2.load_state_dict(m1.state_dict()); m2.eval()
crit = nn.CrossEntropyLoss()
X = torch.randn(128, 3, 64, 64, device=DEVICE)
Y = torch.randint(0, 8, (128,), device=DEVICE)

m1.zero_grad(); crit(m1(X), Y).backward()
g_big = [p.grad.clone() for p in m1.parameters() if p.grad is not None]
m2.zero_grad()
for k in range(4):
    crit(m2(X[k * 32:(k + 1) * 32]), Y[k * 32:(k + 1) * 32]).backward()
g_acc = [p.grad.clone() for p in m2.parameters() if p.grad is not None]
num = sum((a - b).norm() ** 2 for a, b in zip(g_big, g_acc)).sqrt()
den = sum(a.norm() ** 2 for a in g_big).sqrt()
err_eval = float(num / den)
print(f'eval 模式：梯度相对误差 = {err_eval:.3e}', flush=True)

m3 = build_resnet18(8, pretrained=False).to(DEVICE).train()
m4 = build_resnet18(8, pretrained=False).to(DEVICE)
m4.load_state_dict(m3.state_dict()); m4.train()
m3.zero_grad(); crit(m3(X), Y).backward()
m4.zero_grad()
for k in range(4):
    crit(m4(X[k * 32:(k + 1) * 32]), Y[k * 32:(k + 1) * 32]).backward()
num2 = sum((a.grad - b.grad).norm() ** 2 for a, b in zip(m3.parameters(), m4.parameters())).sqrt()
den2 = sum(a.grad.norm() ** 2 for a in m3.parameters()).sqrt()
err_train = float(num2 / den2)
print(f'train 模式（BN 统计量不同）：梯度相对误差 = {err_train:.3e}', flush=True)
save('c_accum.json', {'rel_grad_error_eval': err_eval, 'rel_grad_error_train': err_train})

print('\n=== C4 BN vs GroupNorm（BloodMNIST 3000）===', flush=True)
imgs, labels, lm = load_arrays('bloodmnist', 'train')
timgs, tlabels, _ = load_arrays('bloodmnist', 'test')
idx = stratified_subset(labels, 3000, seed=0)
Xtr, Ytr = imgs[idx], labels[idx]
te = make_loader(timgs, tlabels, size=28, bs=256)
res4 = {}
for norm in ['bn', 'gn']:
    for bs in [2, 8, 32]:
        set_seed(42)
        m = SmallCNN(len(lm), 3, norm)
        tr = make_loader(Xtr, Ytr, size=28, train=True, bs=bs)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        hist, el, _ = train(m, tr, te, epochs=6, optimizer=opt)
        acc, _, _, _ = evaluate(m, te)
        res4[f'{norm}_bs{bs}'] = {'acc': float(acc), 'time': el}
        print(f'  {norm.upper():3s} bs={bs:2d}  acc={acc:.4f}  ({el:.0f}s)', flush=True)
        save('c_bn_gn.json', res4)
