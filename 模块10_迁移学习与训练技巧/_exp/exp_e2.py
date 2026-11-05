"""实验 E2：64x64 下的技巧消融（固定预训练 backbone）+ 从头训参照。"""
import numpy as np, torch, torch.nn as nn, math
from common import *
from sklearn.metrics import f1_score

NAME, SIZE, EPOCHS, BS = 'bloodmnist', 64, 20, 128
imgs, labels, lm = load_arrays(NAME, 'train')
timgs, tlabels, _ = load_arrays(NAME, 'test')
NC = len(lm)
idx = stratified_subset(labels, 2000, seed=0)
Xtr, Ytr = imgs[idx], labels[idx]
test_loader = make_loader(timgs, tlabels, size=SIZE, bs=256)
SOFT = dict(rot=8, trans=0.05, scale=0.05)


def cosine_warmup(opt, total, warm_frac=0.1):
    warm = max(1, int(total * warm_frac))
    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def tta_acc(model, loader):
    model.eval()
    P, Y = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        p = model(norm(xb)).softmax(1) + model(norm(torch.flip(xb, dims=[3]))).softmax(1)
        P.append((p / 2).float().cpu()); Y.append(yb)
    P, Y = torch.cat(P), torch.cat(Y)
    return float((P.argmax(1) == Y).float().mean()), float(
        f1_score(Y.numpy(), P.argmax(1).numpy(), average='macro'))


def run(tag, pretrained, aug, use_sched, ls=0.0, tta=False, seed=42):
    set_seed(seed)
    m = build_resnet18(NC, pretrained=pretrained)
    if pretrained:
        def g(prefix, lr):
            return {'params': [p for n, p in m.named_parameters() if n.startswith(prefix)], 'lr': lr}
        opt = torch.optim.AdamW([g('layer1', 1e-5), g('layer2', 1e-5), g('layer3', 3e-5),
                                 g('layer4', 1e-4), g('fc', 1e-3)], lr=1e-3, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=BS)
    sched = cosine_warmup(opt, len(tl) * EPOCHS) if use_sched else None
    train(m, tl, test_loader, epochs=EPOCHS, optimizer=opt, sched=sched,
          criterion=nn.CrossEntropyLoss(label_smoothing=ls), aug=aug)
    if tta:
        acc, f1 = tta_acc(m, test_loader)
    else:
        acc, _, probs, ys = evaluate(m, test_loader)
        f1 = f1_score(ys.numpy(), probs.argmax(1).numpy(), average='macro')
    print(f'{tag:30s} acc={acc:.4f} f1={f1:.4f}', flush=True)
    return {'tag': tag, 'acc': float(acc), 'macro_f1': float(f1)}


out = []
out.append(run('0 从头训参照', False, {}, False))
out.append(run('1 基线 预训练+常数LR', True, {}, False))
out.append(run('2 +安全增强', True, SOFT, False))
out.append(run('3 +Cosine+warmup', True, SOFT, True))
out.append(run('4 +LabelSmoothing 0.1', True, SOFT, True, ls=0.1))
out.append(run('5 +TTA', True, SOFT, True, ls=0.1, tta=True))
save('e2_ablation64.json', out)
