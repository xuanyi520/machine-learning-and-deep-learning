"""2D / 2.5D / 3D 训练显存与单步耗时实测。"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet

torch.manual_seed(0)
OUT = Path(r"E:\AI学习\模块11_医学图像分割\outputs")
device = "cuda"


def bench(model, shape, steps=5):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(*shape, device=device)
    y = torch.randint(0, 3, shape[:1] + shape[2:], device=device)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return {"mem_MB": torch.cuda.max_memory_allocated() / 1e6,
            "sec_per_step": (time.time() - t) / steps,
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6}


def unet2d(in_ch):
    return UNet(spatial_dims=2, in_channels=in_ch, out_channels=3,
                channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2)


def unet3d():
    return UNet(spatial_dims=3, in_channels=1, out_channels=3,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)


res = {}
res["2d_256_b8"] = bench(unet2d(1), (8, 1, 256, 256))
res["2.5d_256_b8"] = bench(unet2d(3), (8, 3, 256, 256))
res["3d_96_b2"] = bench(unet3d(), (2, 1, 96, 96, 96))
res["3d_128_b2"] = bench(unet3d(), (2, 1, 128, 128, 128))
for k, v in res.items():
    print(k, {kk: round(vv, 3) for kk, vv in v.items()})
json.dump(res, open(OUT / "memory_bench.json", "w"), indent=2)
