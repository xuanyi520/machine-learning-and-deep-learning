import numpy as np, torch, torch.nn as nn, time
import medmnist
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
root = r'E:\AI学习\练习代码\data\medmnist'

tr = medmnist.OrganAMNIST(split='train', root=root, download=False, size=28)
te = medmnist.OrganAMNIST(split='test', root=root, download=False, size=28)
Xtr = (tr.imgs.astype(np.float32) / 255.)[:, None]
ytr = tr.labels.ravel().astype(np.int64)
Xte = (te.imgs.astype(np.float32) / 255.)[:, None]
yte = te.labels.ravel().astype(np.int64)

rng = np.random.default_rng(0)
idx = rng.choice(len(Xtr), 8000, replace=False)
Xtr, ytr = Xtr[idx], ytr[idx]
idx = rng.choice(len(Xte), 3000, replace=False)
Xte, yte = Xte[idx], yte[idx]
print('train', Xtr.shape, 'test', Xte.shape)

# ---------- 构造“临床指标”：5 维，与类别相关 ----------
# 说明：OrganAMNIST 没有真实临床表格。这里按类别均值生成 5 维指标，
# 模拟“临床指标和器官/疾病类别有相关性”的真实情况。
rng2 = np.random.default_rng(42)
CLASS_MU = rng2.normal(0, 1.0, size=(11, 4))

def make_clin(labels, rng):
    cont = CLASS_MU[labels] + rng.normal(0, 1.2, size=(len(labels), 4))
    p = 1 / (1 + np.exp(-CLASS_MU[labels, 0] * 0.8))
    binv = rng.binomial(1, p).astype(np.float32)
    return np.concatenate([cont.astype(np.float32), binv[:, None]], 1)

Ctr = make_clin(ytr, np.random.default_rng(1))
Cte = make_clin(yte, np.random.default_rng(2))
mu, sd = Ctr[:, :4].mean(0), Ctr[:, :4].std(0)
Ctr[:, :4] = (Ctr[:, :4] - mu) / sd
Cte[:, :4] = (Cte[:, :4] - mu) / sd
C_ALL_MEAN = np.zeros(5, dtype=np.float32)   # 模拟临床数据全部缺失

Xtr_t = torch.tensor(Xtr).to(dev); ytr_t = torch.tensor(ytr).to(dev)
Xte_t = torch.tensor(Xte).to(dev); yte_t = torch.tensor(yte).to(dev)
Ctr_t = torch.tensor(Ctr).to(dev); Cte_t = torch.tensor(Cte).to(dev)
Cmean_t = torch.tensor(C_ALL_MEAN).to(dev)

class ImgEncoder(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, emb), nn.ReLU())
    def forward(self, x): return self.net(x)

class ImgOnly(nn.Module):
    def __init__(self): super().__init__(); self.enc = ImgEncoder(); self.head = nn.Linear(64, 11)
    def forward(self, x, c=None): return self.head(self.enc(x))

class ClinOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 11))
    def forward(self, x, c): return self.net(c)

class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = ImgEncoder()
        self.head = nn.Sequential(nn.Linear(69, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 11))
    def forward(self, x, c): return self.head(torch.cat([self.enc(x), c], 1))

class LateFusion(nn.Module):
    """两个分支各自出 logits，训练时各自算 CE，推理时概率平均。"""
    def __init__(self):
        super().__init__(); self.img = ImgOnly(); self.clin = ClinOnly()
    def forward(self, x, c):
        return self.img(x), self.clin(x, c)

class InterFusion(nn.Module):
    """中间融合：影像/临床各编码成一个 token，做 cross-attention 后拼接。"""
    def __init__(self, dim=64):
        super().__init__()
        self.enc = ImgEncoder(dim)
        self.clin = nn.Sequential(nn.Linear(5, dim), nn.ReLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim*2, batch_first=True, dropout=0.1)
        self.fuse = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.Linear(dim*2, 64), nn.ReLU(), nn.Linear(64, 11))
    def forward(self, x, c):
        zi = self.enc(x)[:, None, :]
        zc = self.clin(c)[:, None, :]
        z = self.fuse(torch.cat([zi, zc], 1))
        return self.head(torch.cat([z[:, 0], z[:, 1]], 1))

def run(model, epochs=12, lr=1e-3, bs=128, use_clin=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xtr_t[j], Ctr_t[j]) if use_clin else model(Xtr_t[j], None)
            if isinstance(out, tuple):
                loss = lossf(out[0], ytr_t[j]) + lossf(out[1], ytr_t[j])
            else:
                loss = lossf(out, ytr_t[j])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Xte_t, Cte_t) if use_clin else model(Xte_t, None)
        if isinstance(out, tuple):
            p = (0.5*torch.softmax(out[0], 1) + 0.5*torch.softmax(out[1], 1)).cpu().numpy()
        else:
            p = torch.softmax(out, 1).cpu().numpy()
        if use_clin:
            out_m = model(Xte_t, Cmean_t.expand(len(Xte_t), -1))
            if isinstance(out_m, tuple):
                p_missing = (0.5*torch.softmax(out_m[0], 1) + 0.5*torch.softmax(out_m[1], 1)).cpu().numpy()
            else:
                p_missing = torch.softmax(out_m, 1).cpu().numpy()
        else:
            p_missing = p
    acc = (p.argmax(1) == yte).mean()
    acc_miss = (p_missing.argmax(1) == yte).mean()
    auc = roc_auc_score(yte, p, multi_class='ovr', average='macro')
    auc_miss = roc_auc_score(yte, p_missing, multi_class='ovr', average='macro')
    return acc, auc, acc_miss, auc_miss

models = [
    ('图像单模态', ImgOnly, False),
    ('临床单模态', ClinOnly, True),
    ('早融合', EarlyFusion, True),
    ('晚融合', LateFusion, True),
    ('中间融合', InterFusion, True),
]
print()
for name, mk, use_clin in models:
    res = []
    t0 = time.time()
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        m = mk().to(dev)
        res.append(run(m, use_clin=use_clin))
    res = np.array(res)
    print('%-10s acc %.4f+-%.4f  AUC %.4f+-%.4f | 临床缺失 acc %.4f  AUC %.4f | %.1fs' % (
        name, res[:,0].mean(), res[:,0].std(), res[:,1].mean(), res[:,1].std(),
        res[:,2].mean(), res[:,3].mean(), time.time()-t0))
