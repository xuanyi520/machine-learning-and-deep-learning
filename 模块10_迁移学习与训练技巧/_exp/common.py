"""模块 10 实验公共工具。所有增强在 GPU 上按整批完成（28x28 图上 CPU 逐样本增强是瓶颈）。"""
import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = r'E:\AI学习\练习代码\data\medmnist'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------- 数据
_CLS = {'dermamnist': 'DermaMNIST', 'bloodmnist': 'BloodMNIST',
        'organamnist': 'OrganAMNIST', 'pneumoniamnist': 'PneumoniaMNIST',
        'pathmnist': 'PathMNIST'}


def load_arrays(name, split):
    import medmnist, importlib
    from medmnist import INFO
    ds = getattr(importlib.import_module('medmnist'), _CLS[name])(
        split=split, download=False, root=ROOT, size=28)
    imgs = np.stack([np.array(ds[i][0]) for i in range(len(ds))])
    labels = np.array([int(np.array(ds[i][1]).ravel()[0]) for i in range(len(ds))])
    return imgs, labels, INFO[name]['label']


def stratified_subset(labels, n, seed=0):
    rng = np.random.RandomState(seed)
    cls = np.unique(labels)
    per = max(1, n // len(cls))
    idx = []
    for c in cls:
        pool = np.where(labels == c)[0]
        idx.extend(rng.choice(pool, min(per, len(pool)), replace=False).tolist())
    return np.array(sorted(idx))


def prepare(imgs, size=28):
    """(N,H,W) 或 (N,H,W,C) uint8 -> (N,3,size,size) float32 in [0,1]"""
    x = torch.from_numpy(np.ascontiguousarray(imgs))
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    x = x.permute(0, 3, 1, 2).float()
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    return x.div_(255.)


class ArrayDS(Dataset):
    """返回 [0,1] 的 float 张量，增强和标准化都在 GPU 上做。"""
    def __init__(self, imgs, labels, size=28):
        self.imgs = prepare(imgs, size)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


def make_loader(imgs, labels, size=28, train=False, bs=64, workers=0):
    return DataLoader(ArrayDS(imgs, labels, size), batch_size=bs, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=train,
                      persistent_workers=False)


# ---------------------------------------------------------------- GPU 增强
def batch_aug(x, rot=0.0, trans=0.0, scale=0.0, hflip=0.0, vflip=0.0, jitter=0.0):
    """一次对整个 batch 做几何/颜色增强。x: (B,3,H,W) in [0,1]，在 GPU 上。"""
    B = x.size(0)
    dev = x.device
    if rot or trans or scale:
        ang = (torch.rand(B, device=dev) * 2 - 1) * rot * math.pi / 180
        tx = (torch.rand(B, device=dev) * 2 - 1) * trans * 2
        ty = (torch.rand(B, device=dev) * 2 - 1) * trans * 2
        sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * scale
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    if hflip:
        m = torch.rand(B, device=dev) < hflip
        x[m] = torch.flip(x[m], dims=[3])
    if vflip:
        m = torch.rand(B, device=dev) < vflip
        x[m] = torch.flip(x[m], dims=[2])
    if jitter:
        b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
        x = (x * b).clamp(0, 1)
    return x


AUGS = {
    'none': {},
    'safe': dict(rot=10, trans=0.08, scale=0.1),
    'hflip': dict(hflip=0.5),
    'vflip': dict(vflip=0.5),
    'safe+hflip': dict(rot=10, trans=0.08, scale=0.1, hflip=0.5),
    'safe+vflip': dict(rot=10, trans=0.08, scale=0.1, vflip=0.5),
    'color': dict(jitter=0.3),
}


def build_aug(kind):
    return AUGS[kind]


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ---------------------------------------------------------------- 模型
class SmallCNN(nn.Module):
    def __init__(self, nclass, in_ch=1, norm='bn'):
        super().__init__()
        def n(c):
            return nn.BatchNorm2d(c) if norm == 'bn' else nn.GroupNorm(8, c)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), n(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), n(64), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2),
                                  nn.Linear(64 * 7 * 7, nclass))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(nclass, pretrained=True):
    from torchvision import models
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    m.fc = nn.Linear(512, nclass)
    return m


def freeze_to(model, keep_prefixes):
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(k) for k in keep_prefixes)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 损失
class FocalLoss(nn.Module):
    """FL = -alpha_t * (1-p_t)^gamma * log(p_t)，可选标签平滑。"""
    def __init__(self, gamma=2.0, weight=None, ls=0.0):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, ls

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)                      # (B,C)
        nll = F.nll_loss(logp, target, reduction='none')     # (B,)
        if self.ls > 0:
            smooth = -logp.mean(1)                           # 均匀分布那部分的 CE
            nll = (1 - self.ls) * nll + self.ls * smooth
        pt = logp.exp().gather(1, target[:, None]).squeeze(1).detach()
        loss = (1 - pt) ** self.gamma * nll
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


# ---------------------------------------------------------------- 训练
@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    probs_all, y_all = [], []
    for xb, yb in loader:
        xb = norm(xb.to(DEVICE, non_blocking=True))
        yb = yb.to(DEVICE, non_blocking=True)
        out = model(xb)
        if criterion is not None:
            loss_sum += criterion(out, yb).item() * yb.size(0)
        probs_all.append(out.softmax(1).float().cpu()); y_all.append(yb.cpu())
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return (correct / total, (loss_sum / total if criterion else None),
            torch.cat(probs_all), torch.cat(y_all))


def train(model, train_loader, val_loader, epochs=15, lr=1e-3, wd=1e-4,
          criterion=None, sched=None, amp=False, accum=1, tag='run',
          log_every=0, optimizer=None, aug=None, clip=None, eval_every=0):
    model = model.to(DEVICE)
    criterion = criterion or nn.CrossEntropyLoss()
    aug = aug or {}
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        run_loss = nb = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            if aug:
                xb = batch_aug(xb, **aug)
            xb = norm(xb)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss = criterion(model(xb), yb) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                if clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
            run_loss += loss.item() * accum; nb += 1
        vacc = vloss = None
        if eval_every and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            vacc, vloss, _, _ = evaluate(model, val_loader, criterion)
        hist.append({'epoch': ep + 1, 'train_loss': run_loss / nb,
                     'val_acc': vacc, 'val_loss': vloss})
        if log_every and (ep + 1) % log_every == 0:
            extra = f' val_acc={vacc:.4f}' if vacc is not None else ''
            print(f'  [{tag}] ep{ep+1:02d} loss={run_loss/nb:.4f}{extra}', flush=True)
    return hist, time.time() - t0, None


def save(name, obj):
    p = os.path.join(OUT, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print('saved', p, flush=True)
