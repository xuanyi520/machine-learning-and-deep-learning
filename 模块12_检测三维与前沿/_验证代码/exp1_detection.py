import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---------- 1. IoU ----------
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

boxes = np.array([
    [10, 10, 60, 60],    # A
    [12, 14, 58, 62],    # B
    [100, 100, 150, 150],# C
    [104, 98, 152, 148], # D
    [200, 50, 240, 90],  # E
    [14, 12, 56, 58],    # F
], dtype=float)
scores = np.array([0.92, 0.85, 0.78, 0.65, 0.60, 0.55])
names = ['A', 'B', 'C', 'D', 'E', 'F']

print('=== IoU 矩阵 ===')
print('      ' + ' '.join(f'{n:>6s}' for n in names))
for i in range(6):
    row = ' '.join(f'{iou(boxes[i], boxes[j]):6.3f}' for j in range(6))
    print(f'{names[i]:>4s}  {row}')

# ---------- 2. NMS ----------
def nms(boxes, scores, thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= thr]
    return keep

keep = nms(boxes, scores, 0.5)
print('\n=== NMS 结果 (thr=0.5) ===')
print('保留索引:', keep, ' -> ', [names[i] for i in keep])
for i in keep:
    print(f'  {names[i]}  score={scores[i]:.2f}  box={boxes[i].astype(int).tolist()}')
print('被抑制:', [names[i] for i in range(6) if i not in keep])

# 不同阈值下的保留数量
print('\n=== 阈值敏感性 ===')
for t in [0.3, 0.5, 0.7, 0.8, 0.85]:
    k = nms(boxes, scores, t)
    print(f'  thr={t:.2f}  保留 {len(k)} 个: {[names[i] for i in k]}')

# ---------- 3. AP / mAP@0.5 ----------
# 两张图，单类别。GT 每图 1 个目标。
gt = {
    'img1': [np.array([0, 0, 50, 50.])],
    'img2': [np.array([100, 100, 160, 160.])],
}
preds = [
    ('img1', np.array([0, 0, 52, 48.]), 0.95),   # FP（IoU 0.96 但类别/位置判定为背景? -> 见下）
    ('img1', np.array([2, 1, 49, 51.]), 0.90),   # TP
    ('img2', np.array([102, 98, 158, 162.]), 0.85), # TP
    ('img2', np.array([10, 10, 40, 40.]), 0.70), # FP
]
# 为了让例子有教学意义：第一条故意设成 IoU 低于 0.5 的框
preds[0] = ('img1', np.array([60, 60, 110, 110.]), 0.95)

def match(preds, gt, thr=0.5):
    used = {k: [False]*len(v) for k, v in gt.items()}
    out = []
    for img, box, sc in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt[img]):
            v = iou(box, g)
            if v > best:
                best, best_j = v, j
        if best >= thr and not used[img][best_j]:
            used[img][best_j] = True
            out.append((sc, 1))
        else:
            out.append((sc, 0))
    return out

res = match(preds, gt)
res.sort(key=lambda x: -x[0])
print('\n=== 按分数排序后的匹配结果 ===')
tp = fp = 0
prec, rec = [], []
for sc, hit in res:
    tp += hit; fp += 1 - hit
    prec.append(tp/(tp+fp)); rec.append(tp/2)
    print(f'  score={sc:.2f}  {"TP" if hit else "FP"}  累计P={prec[-1]:.3f}  R={rec[-1]:.3f}')

# all-point AP（VOC2010 之后的面积法）
mrec = np.concatenate(([0.], rec, [1.]))
mpre = np.concatenate(([0.], prec, [0.]))
for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
idx = np.where(mrec[1:] != mrec[:-1])[0]
ap = np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1])
print(f'AP@0.5 = {ap:.4f}')

# ---------- 4. FROC ----------
rng = np.random.default_rng(0)
n_scan = 40
# 20 个扫描有 1 个结节（正样本），20 个没有
scores_pos = np.clip(rng.normal(0.62, 0.18, 20), 0.02, 0.99)
# 每个扫描产生若干假阳性候选
fp_scores = []
for _ in range(n_scan):
    n_fp = rng.poisson(1.2)
    fp_scores.extend(np.clip(rng.normal(0.30, 0.15, n_fp), 0.01, 0.95).tolist())
fp_scores = np.array(fp_scores)

ths = np.linspace(0.01, 0.99, 99)
sens = [(scores_pos >= t).mean() for t in ths]
fps = [((fp_scores >= t).sum() / n_scan) for t in ths]

print('\n=== FROC 采点 ===')
for target in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
    # 在不超过目标 FP/scan 的点里取最大敏感度
    cand = [(f, s, t) for f, s, t in zip(fps, sens, ths) if f <= target]
    if cand:
        f, s, t = max(cand, key=lambda x: x[1])
        print(f'  FP/scan<={target:<5}  sensitivity={s:.3f}  (@thr={t:.2f}, 实际FP/scan={f:.2f})')
print(f'  候选总数: 真结节 20, 假阳性候选 {len(fp_scores)}, 平均每扫描 {len(fp_scores)/n_scan:.2f} 个')
print(f'  随机打分下 AUC 式指标无法反映“每扫描几个FP”，FROC 才能')
