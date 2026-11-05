import numpy as np, torch, torch.nn as nn, time
import medmnist
from medmnist import INFO

torch.manual_seed(0); np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
root = r'E:\AI学习\练习代码\data\medmnist'

tr = medmnist.OrganAMNIST(split='train', root=root, download=False, size=28)
te = medmnist.OrganAMNIST(split='test', root=root, download=False, size=28)
Xtr = (tr.imgs.astype(np.float32) / 255.)[:, None, :, :]
ytr = tr.labels.ravel().astype(np.int64)
Xte = (te.imgs.astype(np.float32) / 255.)[:, None, :, :]
yte = te.labels.ravel().astype(np.int64)
print('全量:', Xtr.shape, Xte.shape, '类别数', len(INFO['organamnist']['label']))

rng = np.random.default_rng(0)
def subsample(X, y, n):
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]

Xte_s, yte_s = subsample(Xte, yte, 3000)
Xte_t = torch.tensor(Xte_s).to(dev); yte_t = torch.tensor(yte_s).to(dev)

class CNN(nn.Module):
    def __init__(self, n_cls=11, emb=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, emb), nn.ReLU())
        self.head = nn.Linear(emb, n_cls)
    def forward(self, x):
        z = self.f(x)
        return self.head(z)

class TinyViT(nn.Module):
    def __init__(self, n_cls=11, dim=64, depth=2, heads=4, patch=4, img=28):
        super().__init__()
        self.patch = nn.Conv2d(1, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)
    def forward(self, x):
        z = self.patch(x).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(z.size(0), -1, -1), z + self.pos], 1)
        z = self.enc(z)
        return self.head(self.norm(z[:, 0]))

def train_eval(model, X, y, epochs=8, bs=128, lr=1e-3):
    Xt = torch.tensor(X).to(dev); yt = torch.tensor(y).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1)
    return (pred == yte_t).float().mean().item()

for n_train, ep in [(500, 40), (2000, 20), (8000, 12)]:
    Xs, ys = subsample(Xtr, ytr, n_train)
    print()
    print("=== train", n_train, "epochs", ep, "===")
    for name, mk in [("CNN", CNN), ("TinyViT", TinyViT)]:
        accs = []
        for seed in [1, 2, 3]:
            torch.manual_seed(seed)
            m = mk().to(dev)
            acc = train_eval(m, Xs, ys, epochs=ep)
            accs.append(acc)
        nparam = sum(p.numel() for p in m.parameters())
        print("  %-8s params %6.1fK  acc %.4f +- %.4f  seeds %s" % (name, nparam/1e3, np.mean(accs), np.std(accs), [round(a,4) for a in accs]))

m = TinyViT().to(dev)
x = torch.randn(2, 1, 28, 28, device=dev)
with torch.no_grad():
    p = m.patch(x)
print()
print("TinyViT patch out", tuple(p.shape), "tokens", p.flatten(2).shape[-1])
