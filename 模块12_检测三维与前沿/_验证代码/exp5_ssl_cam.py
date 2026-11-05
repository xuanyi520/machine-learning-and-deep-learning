import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import medmnist
from sklearn.linear_model import LogisticRegression

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
root = r'E:\AI学习\练习代码\data\medmnist'
tr = medmnist.OrganAMNIST(split='train', root=root, download=False, size=28)
te = medmnist.OrganAMNIST(split='test', root=root, download=False, size=28)
Xtr = (tr.imgs.astype(np.float32)/255.)[:, None]
ytr = tr.labels.ravel().astype(np.int64)
Xte = (te.imgs.astype(np.float32)/255.)[:, None]
yte = te.labels.ravel().astype(np.int64)
rng = np.random.default_rng(0)
i1 = rng.choice(len(Xtr), 5000, replace=False); Xu = Xtr[i1]
i2 = rng.choice(np.setdiff1d(np.arange(len(Xtr)), i1), 2000, replace=False)
Xl, yl = Xtr[i2], ytr[i2]
i3 = rng.choice(len(Xte), 3000, replace=False); Xt, yt = Xte[i3], yte[i3]
print('无标注', Xu.shape, '有标注', Xl.shape, '测试', Xt.shape)

# ---------- 1. NT-Xent 手算 ----------
def nt_xent(z1, z2, tau=0.2):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)                 # 2N
    sim = z @ z.t() / tau                      # 2N x 2N
    sim.fill_diagonal_(-1e9)
    # 正样本：i 与 i+N 互为对方
    target = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, target)

torch.manual_seed(0)
z1 = F.normalize(torch.randn(8, 128, device=dev), dim=1)
z2 = F.normalize(torch.randn(8, 128, device=dev), dim=1)
print('\n随机特征 NT-Xent (N=8):', f'{nt_xent(z1, z2):.4f}', ' 理论下限 log(2N-1)=', f'{np.log(15):.4f}')
z2 = z1.clone()
print('完全相同正样本 NT-Xent:', f'{nt_xent(z1, z2):.4f}')

# ---------- 2. 小 CNN 编码器 ----------
class Enc(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, emb))
    def forward(self, x): return self.net(x)

def augment(x):
    B = x.size(0)
    pad = F.pad(x, (4, 4, 4, 4), mode='reflect')
    idx = torch.randint(0, 9, (2, B), device=x.device)
    out = torch.stack([pad[b, :, idx[0, b]:idx[0, b]+28, idx[1, b]:idx[1, b]+28] for b in range(B)])
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    out = out * (0.8 + 0.4*torch.rand(B, 1, 1, 1, device=x.device))
    out = out + 0.05*torch.randn_like(out)
    return out.clamp(0, 1)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = Enc(); self.proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
    def forward(self, x1, x2):
        return self.proj(self.enc(x1)), self.proj(self.enc(x2))

torch.manual_seed(0)
m = SimCLR().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
bs = 256
Xut = torch.tensor(Xu)
n = len(Xut)
losses = []
for ep in range(30):
    perm = torch.randperm(n)
    tot = 0.
    for i in range(0, n, bs):
        xb = Xut[perm[i:i+bs]].to(dev)
        x1, x2 = augment(xb), augment(xb)
        z1, z2 = m(x1, x2)
        loss = nt_xent(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(xb)
    losses.append(tot/n)
print('\nSimCLR 30 epoch loss: 首轮 %.4f  末轮 %.4f' % (losses[0], losses[-1]))

# ---------- 3. 线性探针 ----------
def extract(model, X, bs=512):
    model.eval(); feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            enc = model.enc if hasattr(model, 'enc') else model
            feats.append(enc(torch.tensor(X[i:i+bs]).to(dev)).cpu())
    return torch.cat(feats).numpy()

for name, model in [('随机初始化', Enc().to(dev)), ('SimCLR 预训练', m)]:
    Ftr = extract(model, Xl); Fte = extract(model, Xt)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Ftr, yl)
    print(f'{name:12s} 线性探针准确率 {clf.score(Fte, yt):.4f}')

# ---------- 4. MAE 掩码比例 ----------
print('\n=== MAE 掩码 ===')
for img, p in [(224, 16), (28, 4)]:
    n = (img//p)**2
    for ratio in [0.5, 0.75, 0.9]:
        vis = int(round(n*(1-ratio)))
        print(f'  {img}px patch{p}: {n} tokens, mask {ratio:.0%} -> 可见 {vis}, 被遮 {n-vis}')

# ---------- 5. CAM ----------
# 训练一个普通分类器（图像单模态）
class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 11)
    def forward(self, x, cam=False):
        f = self.features(x)
        z = self.gap(f).flatten(1)
        out = self.fc(z)
        return (out, f) if cam else out

torch.manual_seed(0)
cm = Cls().to(dev)
opt = torch.optim.AdamW(cm.parameters(), lr=1e-3, weight_decay=1e-4)
lf = nn.CrossEntropyLoss()
Xl_t = torch.tensor(Xl).to(dev); yl_t = torch.tensor(yl).to(dev)
for ep in range(15):
    perm = torch.randperm(len(Xl_t), device=dev)
    for i in range(0, len(Xl_t), 128):
        j = perm[i:i+128]
        opt.zero_grad(); loss = lf(cm(Xl_t[j]), yl_t[j]); loss.backward(); opt.step()
cm.eval()
with torch.no_grad():
    acc = (cm(torch.tensor(Xt).to(dev)).argmax(1).cpu().numpy() == yt).mean()
print('\nCAM 用分类器测试准确率:', f'{acc:.4f}')

Xte_t = torch.tensor(Xt).to(dev)
with torch.no_grad():
    logits, feats = cm(Xte_t, cam=True)          # feats: N,128,7,7
    w = cm.fc.weight[:, :, None, None]           # 11,128,1,1
    cam = (feats[:, None] * w[None]).sum(2)      # N,11,7,7
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=28, mode='bilinear', align_corners=False)
    pred = logits.argmax(1)
    cam_p = cam[torch.arange(len(pred)), pred]   # 预测类的 CAM
    cam_p = cam_p / (cam_p.sum((1,2), keepdim=True) + 1e-8)

# 中心 14x14（占 25% 面积）内的 CAM 质量
center = cam_p[:, 7:21, 7:21].sum((1,2))
print('CAM 落在中心 14x14 区域的质量比例: 中位数 %.3f  均值 %.3f' % (center.median().item(), center.mean().item()))
print('若 CAM 均匀分布，期望值 = 0.250')
# 5x5 峰值区域与图像中心距离
flat = cam_p.flatten(1)
peak = flat.argmax(1)
py, px = (peak // 28).float(), (peak % 28).float()
dist = torch.sqrt((py-13.5)**2 + (px-13.5)**2)
print('CAM 峰值到图像中心距离: 中位数 %.2f 像素 (图像半宽 14)' % dist.median().item())
print('预测正确样本的平均 CAM 中心质量: %.3f' % center[pred.cpu()==torch.tensor(yt)].mean().item())
