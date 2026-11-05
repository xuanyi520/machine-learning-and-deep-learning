"""只训练 8 个 epoch 的弱模型，专门用来演示后处理能救回多少。"""
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(r"E:\AI学习\模块11_医学图像分割")
OUT = ROOT / "outputs"
spec = importlib.util.spec_from_file_location("tm", str(ROOT / "scripts" / "01_train_unet.py"))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

torch.manual_seed(1)
device = "cuda"
d = np.load(OUT / "phantom.npz")
tr = tm.PhantomDataset(d["tr_x"], d["tr_y"], augment=True)
va = tm.PhantomDataset(d["va_x"], d["va_y"])
tl = DataLoader(tr, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
vl = DataLoader(va, batch_size=8)

model = tm.UNet(1, 3).to(device)
opt = torch.optim.AdamW(model.parameters(), 3e-4)
ce_w = torch.tensor([0.5, 1.0, 3.0], device=device)
scaler = torch.amp.GradScaler("cuda")
for ep in range(1, 9):
    model.train()
    for x, y in tl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=ce_w) + tm.dice_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    ev = tm.evaluate(model, vl, device)
    print(ep, {k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()})
torch.save(model.state_dict(), OUT / "unet_weak.pt")
print("saved unet_weak.pt")
