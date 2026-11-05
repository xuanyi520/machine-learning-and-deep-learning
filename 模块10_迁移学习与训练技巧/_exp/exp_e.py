"""实验 E：完整消融表（BloodMNIST 2000 张，ResNet18 @28x28）。"""
import numpy as np, torch, torch.nn as nn, math
from common import *
from sklearn.metrics import f1_score

NAME, SIZE, EPOCHS, BS = 'bloodmnist', 28, 15, 128
imgs, labels, lm = load_arrays(NAME, 'train')
timgs, tlabels, _ = load_arrays(NAME, 'test')
NC = len(lm)
idx = stratified_subset(labels, 2000, seed=0)
Xtr, Ytr = imgs[idx], labels[idx]
test_loader = make_loader(timgs, tlabels, size=SIZE, bs=256)


def cosine_warmup(opt, total_steps, warm_frac=0.1):
    warm = max(1, int(total_steps * warm_frac))
    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict_tta(model, loader, use_flip=True):
    model.eval()
    P, Y = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        p = model(norm(xb)).softmax(1)
        if use_flip:
            p = p + model(norm(torch.flip(xb, dims=[3]))).softmax(1)
            p = p / 2
        P.append(p.float().cpu()); Y.append(yb)
    return torch.cat(P), torch.cat(Y)


def run(tag, pretrained, aug, sched_kind, epochs=EPOCHS, tta=False, ls=0.0, seed=42):
    set_seed(seed)
    m = build_resnet18(NC, pretrained=pretrained)
    if pretrained:
        def grp(prefix, lr):
            return {'params': [p for n, p in m.named_parameters() if n.startswith(prefix)], 'lr': lr}
        opt = torch.optim.AdamW([grp('layer1', 1e-5), grp('layer2', 1e-5), grp('layer3', 3e-5),
                                 grp('layer4', 1e-4), grp('fc', 1e-3)],
                                lr=1e-3, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = make_loader(Xtr, Ytr, size=SIZE, train=True, bs=BS)
    steps = len(tl) * epochs
    sched = cosine_warmup(opt, steps) if sched_kind == 'cosine' else None
    hist, el, _ = train(m, tl, test_loader, epochs=epochs, optimizer=opt,
                        criterion=nn.CrossEntropyLoss(label_smoothing=ls),
                        aug=build_aug(aug), sched=sched)
    if tta:
        probs, ys = predict_tta(m, test_loader, True)
        acc = (probs.argmax(1) == ys).float().mean().item()
    else:
        acc, _, probs, ys = evaluate(m, test_loader)
    f1 = f1_score(ys.numpy(), probs.argmax(1).numpy(), average='macro')
    print(f'{tag:32s} acc={acc:.4f} macroF1={f1:.4f} ({el:.0f}s)', flush=True)
    return {'tag': tag, 'acc': float(acc), 'macro_f1': float(f1), 'time': el}


out = []
out.append(run('1 基线 从头训+常数LR', False, 'none', 'const'))
out.append(run('2 +安全增强', False, 'safe', 'const'))
out.append(run('3 +Cosine+warmup', False, 'safe', 'cosine'))
out.append(run('4 +预训练(差分LR)', True, 'safe', 'cosine'))
out.append(run('5 +LabelSmoothing', True, 'safe', 'cosine', ls=0.1))
out.append(run('6 +TTA', True, 'safe', 'cosine', ls=0.1, tta=True))
save('e_ablation.json', out)

seeds = []
for s in [0, 1, 2]:
    r = run(f'seed{s}', True, 'safe', 'cosine', ls=0.1, tta=True, seed=s)
    seeds.append(r['acc'])
print('3 seeds', seeds, 'mean', round(float(np.mean(seeds)), 4), 'std', round(float(np.std(seeds)), 4), flush=True)
save('e_seeds.json', {'seeds': seeds, 'mean': float(np.mean(seeds)), 'std': float(np.std(seeds))})
