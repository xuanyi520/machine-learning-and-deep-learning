import numpy as np

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def anchors_at(cx, cy, scales, ratios=(0.5, 1.0, 2.0)):
    out = []
    for s in scales:
        for r in ratios:
            w = s * np.sqrt(r); h = s / np.sqrt(r)
            out.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    return np.array(out)

# 3mm 结节，0.7mm 像素 -> 4.3 px，中心 (100,100)
d = 3/0.7
gt = np.array([100-d/2, 100-d/2, 100+d/2, 100+d/2])
print('GT 边长 %.2f px, 面积 %.2f px^2' % (d, d*d))

for scales in [(8, 16, 32), (4, 8, 16), (2, 4, 8)]:
    A = anchors_at(100, 100, scales)
    ious = np.array([iou(gt, a) for a in A])
    best = ious.max()
    print('\nscales=%s, 每位置 %d 个 anchor' % (str(scales), len(A)))
    print('  最大 IoU %.3f, >=0.5 的 anchor 数 %d, >=0.3 的 %d' % (best, (ious>=0.5).sum(), (ious>=0.3).sum()))
    print('  各 scale 的最大 IoU:', [round(max(ious[i*3:(i+1)*3]), 3) for i in range(len(scales))])

# 特征图上的 anchor 总数
for size, stride in [(512, 16), (512, 8), (224, 16)]:
    loc = (size//stride)**2
    print(f'\n输入 {size}, stride {stride}: {loc} 个位置 x 9 anchor = {loc*9} 个 anchor')
