# -*- coding: utf-8 -*-
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from torchvision import transforms
from medmnist import PneumoniaMNIST

ROOT = r'E:\AI学习\练习代码\data\medmnist'
tf = transforms.Compose([transforms.ToTensor()])
ds = PneumoniaMNIST(split='train', transform=tf, download=True, root=ROOT)

print("train_set 是什么类型 :", type(ds))
print()
print("里面有哪些字段:")
for k, v in vars(ds).items():
    t = type(v).__name__
    if hasattr(v, 'shape'):
        t += f" shape={tuple(v.shape)} dtype={getattr(v, 'dtype', None)}"
    elif isinstance(v, list):
        t += f" len={len(v)}"
    print(f"  {k:20s} : {t}")
print()
print("imgs 前两个元素 :", repr(ds.imgs[0])[:150])
print()
try:
    print("imgs 内存占用   : %.2f MB" % (ds.imgs.nbytes / 1024 / 1024))
except AttributeError:
    print("imgs 不是 numpy，是 list，元素个数", len(ds.imgs))
print("labels 前 5 个  :", np.asarray(ds.labels).ravel()[:5])
