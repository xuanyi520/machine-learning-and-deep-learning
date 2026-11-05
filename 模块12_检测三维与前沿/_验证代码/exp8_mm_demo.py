import numpy as np, torch, torch.nn as nn
import medmnist
from sklearn.metrics import roc_auc_score

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
root = r'E:\AI学习\练习代码\data\medmnist'
tr = medmnist.OrganAMNIST(split='train', root=root, download=False, size=28)
te = medmnist.OrganAMNIST(split='test', root=root, download=False, size=28)
Xtr = (tr.imgs.astype(np.float32)/255.)[:, None]
ytr = tr.labels.ravel().astype(np.int64)
Xte = (te.imgs.astype(np.float32)/255.)[:, None]
yte = te.labels.ravel().astype(np.int64)
rng = np.random.default_rng(0)
i = rng.choice(len(Xtr), 3000, replace=False); Xtr, ytr = Xtr[i], ytr[i]
j = rng.choice(len(Xte), 1000, replace=False); Xte, yte = Xte[j], yte[j]

# 构造 5 维“临床指标”（没有真实表格，按类别均值生成）
rng2 = np.random.default_rng(42)
MU = rng2.normal(0, 1.0, (11, 4))
def clin(y, rng):
    c = MU[y] + rng.normal(0, 1.2, (len(y), 4))
    p = 1/(1+np.exp(-MU[y, 0]*0.8))
    b = rng.binomial(1, p).astype(np.float32)
    return np.concatenate([c.astype(np.float32), b[:, None]], 1)
Ctr = clin(ytr, np.random.default_rng(1)); Cte = clin(yte, np.random.default_rng(2))
mu, sd = Ctr[:, :4].mean(0), Ctr[:, :4].std(0)
Ctr[:, :4] = (Ctr[:, :4]-mu)/sd; Cte[:, :4] = (Cte[:, :4]-mu)/sd

Xt = torch.tensor(Xtr).to(dev); yt = torch.tensor(ytr).to(dev)
Xe = torch.tensor(Xte).to(dev); ye = torch.tensor(yte).to(dev)
Ct = torch.tensor(Ctr).to(dev); Ce = torch.tensor(Cte).to(dev)

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 64), nn.ReLU())
    def forward(self, x): return self.net(x)

class Img(nn.Module):
    def __init__(self): super().__init__(); self.e = Enc(); self.h = nn.Linear(64, 11)
    def forward(self, x, c=None): return self.h(self.e(x))

class Early(nn.Module):
    def __init__(self):
        super().__init__(); self.e = Enc()
        self.h = nn.Sequential(nn.Linear(69, 64), nn.ReLU(), nn.Linear(64, 11))
    def forward(self, x, c): return self.h(torch.cat([self.e(x), c], 1))

class Late(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = Img()
        self.clin = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 11))
    def forward(self, x, c): return self.img(x), self.clin(c)

def run(m, epochs=4):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for k in range(0, len(Xt), 128):
            idx = perm[k:k+128]
            opt.zero_grad()
            out = m(Xt[idx], Ct[idx])
            loss = lf(out[0], yt[idx]) + lf(out[1], yt[idx]) if isinstance(out, tuple) else lf(out, yt[idx])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out = m(Xe, Ce)
        p = 0.5*torch.softmax(out[0], 1)+0.5*torch.softmax(out[1], 1) if isinstance(out, tuple) else torch.softmax(out, 1)
        p = p.cpu().numpy()
    return (p.argmax(1) == yte).mean(), roc_auc_score(yte, p, multi_class='ovr', average='macro')

for name, mk in [('image', Img), ('early', Early), ('late', Late)]:
    torch.manual_seed(0)
    acc, auc = run(mk().to(dev))
    print(f'{name:6s} acc {acc:.4f}  AUC {auc:.4f}')
