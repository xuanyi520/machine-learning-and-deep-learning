# -*- coding: utf-8 -*-
"""参考答案脚本：13-E 诊断试验与临床价值、13-F 样本量"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = df.columns.str.strip().str.replace(' ', '_')
y = (df['Class'] == 1).astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def make_pipe():
    return Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=5000))])

full = ['Age', 'Bilirubin', 'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime']
p_full = cross_val_predict(make_pipe(), df[full], y, cv=cv,
                           method='predict_proba')[:, 1]
p_age = cross_val_predict(make_pipe(), df[['Age']], y, cv=cv,
                          method='predict_proba')[:, 1]

print('=' * 60)
print('练习 13-E  诊断试验与临床价值')
print('=' * 60)

fpr, tpr, thr = roc_curve(y, p_full)
k = np.argmax(tpr - fpr)
cut = thr[k]
pred = (p_full >= cut).astype(int)
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
sens = tp / (tp + fn); spec = tn / (tn + fp)
print(f'AUC(6 项) = {roc_auc_score(y, p_full):.3f}，Youden 阈值 {cut:.3f}')
print(f'2x2: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
print(f'灵敏度 {sens:.3f}，特异度 {spec:.3f}，'
      f'PPV {tp/(tp+fp):.3f}，NPV {tn/(tn+fn):.3f}')

print('\n[1] PPV 随患病率变化（贝叶斯公式）')
print(f'{"患病率":>8}{"TP/10万":>10}{"FP/10万":>11}{"PPV":>9}{"NPV":>9}{"每发现1例需检测":>14}')
rows = []
for prev in [0.2065, 0.05, 0.01, 0.001]:
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    N = 100000
    tp2 = sens * N * prev
    fp2 = (1 - spec) * N * (1 - prev)
    rows.append([prev, ppv, npv, tp2, fp2])
    print(f'{prev:8.2%}{tp2:10.0f}{fp2:11.0f}{ppv:9.3%}{npv:9.3%}'
          f'{1/ppv:14.1f}')

print('\n同一模型、不同场景：')
for name, prev in [('本研究队列', 0.2065), ('肝硬化监测', 0.05),
                   ('高危门诊', 0.01), ('社区筛查', 0.001)]:
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    print(f'  {name:<8} 患病率 {prev:6.2%} -> PPV {ppv:7.3%}')

# ---------- DeLong：6 项 vs 仅年龄 ----------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out

def delong_test(y_true, p1, p2):
    order = np.argsort(-y_true)
    yt = y_true[order]
    preds = np.vstack((p1[order], p2[order]))
    m = int(yt.sum()); n = len(yt) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(2)])
    ty = np.array([_midrank(neg[r]) for r in range(2)])
    tz = np.array([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))

a1, a2, z, pdel = delong_test(y, p_full, p_age)
print(f'\n[2] DeLong 6 项 vs 仅年龄: AUC {a1:.3f} vs {a2:.3f}, '
      f'ΔAUC={a1-a2:+.3f}, z={z:.3f}, p={pdel:.4g}')

# ---------- DCA ----------
print('\n[3] 决策曲线分析（DCA）')
N = len(y)
prev = y.mean()
pts = np.arange(0.01, 0.61, 0.01)
nb_model, nb_all, nb_none = [], [], []
for pt in pts:
    pred_t = (p_full >= pt).astype(int)
    tp_ = ((pred_t == 1) & (y == 1)).sum() / N
    fp_ = ((pred_t == 1) & (y == 0)).sum() / N
    nb_model.append(tp_ - fp_ * pt / (1 - pt))
    nb_all.append(prev - (1 - prev) * pt / (1 - pt))
    nb_none.append(0.0)
nb_model, nb_all, nb_none = map(np.array, (nb_model, nb_all, nb_none))
better = (nb_model > nb_all) & (nb_model > 0)
print(f'模型净收益 > 全部治疗 且 > 0 的阈值区间: '
      f'{pts[better].min():.2f} - {pts[better].max():.2f}')
print(f'{"阈值":>6}{"模型 NB":>10}{"全部治疗 NB":>12}{"不治疗":>8}')
for pt in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    i = int(np.argmin(np.abs(pts - pt)))
    print(f'{pts[i]:6.2f}{nb_model[i]:10.4f}{nb_all[i]:12.4f}{0.0:8.4f}')

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(pts, nb_model, lw=2, label='6 项模型')
ax.plot(pts, nb_all, '--', label='全部治疗')
ax.axhline(0, color='gray', lw=1, label='不治疗')
ax.set_ylim(-0.05, max(nb_model.max(), 0.25) + 0.02)
ax.set_xlabel('阈值概率 pt')
ax.set_ylabel('净收益 net benefit')
ax.set_title('决策曲线分析（Hepatitis，OOF 预测）')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块13_医学统计与实验设计\fig_ans_dca.png', dpi=150)
plt.close()

print()
print('=' * 60)
print('练习 13-F  样本量')
print('=' * 60)

p1, p2 = 0.25, 0.15
alpha, power = 0.05, 0.80
z_a = stats.norm.isf(alpha / 2)
z_b = stats.norm.isf(1 - power)
pbar = (p1 + p2) / 2
num = (z_a * np.sqrt(2 * pbar * (1 - pbar))
       + z_b * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2
n = num / (p1 - p2) ** 2
print(f'z(1-alpha/2)={z_a:.4f}, z(1-beta)={z_b:.4f}')
print(f'p_bar=({p1}+{p2})/2={pbar:.2f}')
print(f'2*p_bar*(1-p_bar) = 2*{pbar:.2f}*{1-pbar:.2f} = {2*pbar*(1-pbar):.4f}, '
      f'sqrt = {np.sqrt(2*pbar*(1-pbar)):.4f}')
print(f'p1(1-p1)+p2(1-p2) = {p1*(1-p1):.4f}+{p2*(1-p2):.4f} = '
      f'{p1*(1-p1)+p2*(1-p2):.4f}, sqrt = {np.sqrt(p1*(1-p1)+p2*(1-p2)):.4f}')
print(f'分子 = ({z_a:.4f}*{np.sqrt(2*pbar*(1-pbar)):.4f} + '
      f'{z_b:.4f}*{np.sqrt(p1*(1-p1)+p2*(1-p2)):.4f})^2 = {num:.4f}')
print(f'分母 = ({p1}-{p2})^2 = {(p1-p2)**2:.4f}')
print(f'每组 n = {num/(p1-p2)**2:.2f} -> 向上取整 {int(np.ceil(n))}')
n_cc = n / 4 * (1 + np.sqrt(1 + 4 / (n * abs(p1 - p2)))) ** 2
print(f'连续性校正（Fleiss）：{n_cc:.2f} -> {int(np.ceil(n_cc))}')
n1 = int(np.ceil(n)); n2 = int(np.ceil(n / 0.85))
print(f'脱落校正：{n1} / (1-0.15) = {n1/0.85:.2f} -> 每组 {n2}，'
      f'两组共 {2*n2}')
print(f'（不校正脱落时两组共 {2*n1}）')

# ---------- AUC 0.80 vs 0.83 的模拟 ----------
print('\n[2] 模拟：检出 AUC 0.80 vs 0.83 需要多少例')
def sim_auc_from_d(d, rng):
    """生成本例分数：病例组 N(d/2,1)，对照组 N(-d/2,1)。"""
    m = rng.normal(d / 2, 1, 1)
    return m

def gen_scores(n, d1, d2, rng):
    m = int(round(n * 0.5))
    h = n - m
    z1 = rng.normal(0, 1, n)
    # 两个标志物相关 0.6
    z2 = 0.6 * z1 + np.sqrt(1 - 0.6**2) * rng.normal(0, 1, n)
    s1 = np.where(np.arange(n) < m, d1 / 2, -d1 / 2) + z1
    s2 = np.where(np.arange(n) < m, d2 / 2, -d2 / 2) + z2
    y_ = (np.arange(n) < m).astype(int)
    return y_, s1, s2

d1 = np.sqrt(2) * stats.norm.isf(1 - 0.80)
d2 = np.sqrt(2) * stats.norm.isf(1 - 0.83)
print(f'AUC 0.80 -> d={d1:.4f}; AUC 0.83 -> d={d2:.4f}; 差 {d2-d1:.4f}')

rng = np.random.default_rng(3)
n_rep = 300
for n in [200, 400, 600, 800, 1200, 1600]:
    hits = 0
    for _ in range(n_rep):
        ysim, s1, s2 = gen_scores(n, d1, d2, rng)
        _, _, _, pv = delong_test(ysim, s2, s1)
        hits += pv < 0.05
    print(f'  n={n:5d}: 功效 {hits/n_rep:.3f}')

print('\n[3] 用现有 300 例做事后功效的问题')
print('事后功效 = f(观测到的 p)；p 越小功效越高，等价于把 p 换个说法')
for pv in [0.0001, 0.001, 0.01, 0.05, 0.20, 0.60, 0.90]:
    zobs = stats.norm.isf(pv / 2)
    pw = stats.norm.cdf(zobs - 1.96)
    print(f'  p={pv:<7} z_obs={zobs:5.2f}  事后功效={pw:.3f}'
          f'（同一数据，功效只是 p 的单调变换）')
