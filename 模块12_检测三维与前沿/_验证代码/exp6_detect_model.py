import numpy as np, torch, torch.nn as nn, time
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# ---------- 合成“结节”数据 ----------
def make_sample(seed):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.12, 0.03, (1, 128, 128)).astype(np.float32)
    cy, cx = rng.integers(25, 103, 2)
    r = rng.integers(7, 14)
    yy, xx = np.mgrid[0:128, 0:128]
    mask = (yy-cy)**2 + (xx-cx)**2 <= r*r
    img[0][mask] = rng.uniform(0.65, 0.95)
    # 模糊一下，接近真实结节的边界
    import torch.nn.functional as F
    t = torch.tensor(img)[None]
    t = F.avg_pool2d(F.pad(t, (2,2,2,2), mode='replicate'), 5, 1)
    img = t[0].numpy()
    box = [cx-r, cy-r, cx+r, cy+r]
    return img, np.array(box, dtype=np.float32), r

train = [make_sample(s) for s in range(60)]
test = [make_sample(s) for s in range(1000, 1012)]
print('合成数据: 训练 60 张, 测试 12 张, 每张 1 个“结节”')

def to_tensors(data):
    imgs = [torch.tensor(x[0]) for x in data]
    targets = [{'boxes': torch.tensor(x[1])[None], 'labels': torch.tensor([1])} for x in data]
    return imgs, targets

# ---------- 模型 ----------
m = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
in_feat = m.roi_heads.box_predictor.cls_score.in_features
m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)   # 背景 + 结节
m.to(dev)
print('替换检测头后参数量: %.2fM' % (sum(p.numel() for p in m.parameters())/1e6))

# 冻结 backbone 前几层，只训练 RPN + ROI 头（小数据）
for p in m.backbone.parameters():
    p.requires_grad = False

imgs, targets = to_tensors(train)
t_imgs, t_targets = to_tensors(test)
opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.005, momentum=0.9)

m.train()
bs = 2
loss_hist = []
t0 = time.time()
for it in range(40):
    idx = np.random.default_rng(it).choice(len(imgs), bs, replace=False)
    xb = [imgs[i].to(dev) for i in idx]
    yb = [{k: v.to(dev) for k, v in targets[i].items()} for i in idx]
    loss_dict = m(xb, yb)
    loss = sum(v for v in loss_dict.values())
    opt.zero_grad(); loss.backward(); opt.step()
    loss_hist.append(loss.item())
    if it % 10 == 0:
        print('  iter %2d loss %.4f  %s' % (it, loss.item(), {k: round(v.item(),3) for k, v in loss_dict.items()}))
print('训练 40 iter 用时 %.1fs, 前5轮均值 %.3f, 后10轮均值 %.3f, 最低 %.3f' % (time.time()-t0, np.mean(loss_hist[:5]), np.mean(loss_hist[-10:]), min(loss_hist)))

# ---------- 评估：recall@IoU0.5 ----------
def iou1(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

m.eval()
hits, nbox, scores_hit = 0, [], []
with torch.no_grad():
    for img, tgt in zip(t_imgs, t_targets):
        gt = tgt['boxes'][0]
        out = m([img.to(dev)])[0]
        keep = out['scores'] > 0.3
        boxes = out['boxes'][keep].cpu().numpy(); sc = out['scores'][keep].cpu().numpy()
        nbox.append(len(boxes))
        best = max([iou1(b, gt.numpy()) for b in boxes], default=0)
        if best >= 0.5:
            hits += 1; scores_hit.append(sc.max())
print('测试 recall@IoU0.5 (score>0.3): %d/%d = %.3f' % (hits, len(t_targets), hits/len(t_targets)))
print('每张图平均保留框数: %.2f' % np.mean(nbox))
print('命中样本的最高分:', [round(s, 3) for s in scores_hit])

# ---------- 单图推理速度 ----------
x = t_imgs[0].to(dev)
for _ in range(3): m([x])
torch.cuda.synchronize() if dev == 'cuda' else None
t0 = time.time()
for _ in range(10): m([x])
torch.cuda.synchronize() if dev == 'cuda' else None
print('320 短边单图推理: %.1f ms/张 (%.1f FPS)' % ((time.time()-t0)*100, 10/(time.time()-t0)))
