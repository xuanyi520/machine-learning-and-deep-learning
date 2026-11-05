"""实验 C：AMP 显存/速度、梯度累积等价性、BN vs GroupNorm。"""
import numpy as np, torch, torch.nn as nn, time
from common import *


def measure(bs, size, amp, dtype=torch.float16, iters=30, warmup=5):
    set_seed(0)
    model = build_resnet18(8, pretrained=False).to(DEVICE)
    x = torch.randn(bs, 3, size, size, device=DEVICE)
    y = torch.randint(0, 8, (bs,), device=DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for i in range(iters + warmup):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=amp):
            loss = crit(model(x), y)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del model, x, y, opt
    torch.cuda.empty_cache()
    return dt, peak


print('=== C1 ResNet18 @128x128, batch=128 ===', flush=True)
res = {}
for tag, amp, dt_ in [('fp32', False, torch.float16), ('amp_fp16', True, torch.float16),
                      ('amp_bf16', True, torch.bfloat16)]:
    t, p = measure(128, 128, amp, dt_)
    res[tag] = {'ms_per_iter': round(t * 1000, 2), 'peak_MB': round(p, 1),
                'img_per_s': round(128 / t, 1)}
    print(f'{tag:9s} {t*1000:7.2f} ms/iter  peak {p:7.1f} MB  {128/t:7.0f} img/s', flush=True)
save('c_amp.json', res)

print('\n=== C2 最大可用 batch @128x128 ===', flush=True)
res2 = {}
for tag, amp, dt_ in [('fp32', False, torch.float16), ('amp_fp16', True, torch.float16)]:
    ok = []
    for bs in [64, 128, 256, 384, 512, 768, 1024]:
        try:
            t, p = measure(bs, 128, amp, dt_, iters=8, warmup=2)
            ok.append({'bs': bs, 'peak_MB': round(p, 1), 'ms': round(t * 1000, 1)})
            print(f'  {tag} bs={bs}: peak {p:.0f} MB, {t*1000:.1f} ms', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'  {tag} bs={bs}: OOM', flush=True)
            torch.cuda.empty_cache(); break
    res2[tag] = ok
save('c_amp_batch.json', res2)

print('\n=== C3 梯度累积 ===', flush=True)
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
print(f'train 模式（BN 用当前 batch 统计量）：梯度相对误差 = {err_train:.3e}', flush=True)
save('c_accum.json', {'rel_grad_error_eval': err_eval, 'rel_grad_error_train': err_train})

print('\n=== C4 BN vs GroupNorm @ 小 batch（BloodMNIST 3000）===', flush=True)
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
