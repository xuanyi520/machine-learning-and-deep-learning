import numpy as np, torch, torch.nn as nn
np.set_printoptions(precision=3, suppress=True)

# ---------- 1. 手算用的 3x3 attention ----------
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 0.], [0., 1.], [1., 1.]])
d = 2
S = Q @ K.T
S_scaled = S / np.sqrt(d)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
A = softmax(S_scaled)
O = A @ V
print('QK^T =\n', S)
print('除以 sqrt(2) =\n', S_scaled)
print('softmax 行和 =', A.sum(1))
print('A =\n', A)
print('O = A V =\n', O)

# 代码版（PyTorch，带 batch 和 head 维度）
q = torch.tensor(Q).view(1, 1, 3, 2)   # (B, heads, N, d)
k = torch.tensor(K).view(1, 1, 3, 2)
v = torch.tensor(V).view(1, 1, 3, 2)
attn = torch.softmax(q @ k.transpose(-2, -1) / (2 ** 0.5), dim=-1)
print('torch A=\n', attn[0, 0].numpy())
print('torch O=\n', (attn @ v)[0, 0].numpy())

# ---------- 2. patch embedding 数量 ----------
print('\n=== patch 数量 ===')
for img, p in [(224, 16), (224, 8), (512, 16), (512, 32), (96, 16)]:
    n = (img // p) ** 2
    print(f'  {img}x{img}, patch {p} -> {img//p}x{img//p} = {n} tokens, 序列长度 {n}')

# ---------- 3. ViT token 化实测 ----------
from torchvision.models import vit_b_16
m = vit_b_16(weights=None)
n_param = sum(p.numel() for p in m.parameters())
print(f'\nvit_b_16 参数量: {n_param/1e6:.1f}M')
print('patch 层:', m.conv_proj)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    patches = m.conv_proj(x)
print('conv_proj 输出:', tuple(patches.shape), '-> 展平后', patches.flatten(2).shape)

# ---------- 4. 3D 卷积参数量与显存 ----------
import time
print('\n=== 2D vs 3D 卷积 ===')
c2 = nn.Conv2d(32, 64, 3, padding=1)
c3 = nn.Conv3d(32, 64, 3, padding=1)
print(f'Conv2d(32,64,3) 参数: {sum(p.numel() for p in c2.parameters()):,}')
print(f'Conv3d(32,64,3) 参数: {sum(p.numel() for p in c3.parameters()):,}')
print(f'比值: {sum(p.numel() for p in c3.parameters())/sum(p.numel() for p in c2.parameters()):.2f}x')

if torch.cuda.is_available():
    dev = 'cuda'
    c3 = c3.to(dev); c2 = c2.to(dev)
    x3 = torch.randn(1, 32, 64, 128, 128, device=dev)
    for _ in range(3): y3 = c3(x3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(10): y3 = c3(x3)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'3D 输入 {tuple(x3.shape)} 输出 {tuple(y3.shape)}')
    print(f'  10 次平均 {(t1-t0)*100:.1f} ms, 峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
    x2 = torch.randn(1, 32, 128, 128, device=dev)
    for _ in range(3): y2 = c2(x2)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): y2 = c2(x2)
    torch.cuda.synchronize(); t1 = time.time()
    print(f'2D 输入 {tuple(x2.shape)} 10 次平均 {(t1-t0)*100:.2f} ms')

# ---------- 5. 3mm 结节对应多少像素 ----------
print('\n=== 3mm 结节的像素大小 ===')
for spacing in [0.5, 0.7, 0.8, 1.0, 2.5]:
    px = 3 / spacing
    print(f'  层内 spacing={spacing}mm -> 3mm 直径 = {px:.1f} 像素, 面积约 {np.pi*(px/2)**2:.1f} 平方像素')
print('  512x512 图像中占面积比例:', f'{np.pi*(3/0.7/2)**2/512**2*100:.4f}%')
print('  224x224 缩放后（512->224, 0.7mm）:', f'{3/0.7*224/512:.1f} 像素')
