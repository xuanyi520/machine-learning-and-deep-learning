# -*- coding: utf-8 -*-
"""实验：同预算下 网格搜索 vs 随机搜索
   A. 合成目标：5 维，只有 2 维真正重要（Bergstra & Bengio 2012 的设定）
   B. 真实数据：随机森林 5 折 AUC
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rng = np.random.default_rng(0)

# ---------------- A. 合成目标 ----------------
def f(x):
    # x: (..., 5)，只有第 1、2 维有效
    return 0.90 - (x[..., 0] - 0.65) ** 2 - (x[..., 1] - 0.35) ** 2

BUDGET = 32
# 网格：每维 2 个水平 -> 2^5 = 32
levels = np.array([0.0, 1.0])
grid_pts = np.array(np.meshgrid(*([levels] * 5))).T.reshape(-1, 5)
grid_best = f(grid_pts).max()

# 随机：32 个独立均匀样本
rand_best = []
for _ in range(2000):
    pts = rng.random((BUDGET, 5))
    rand_best.append(f(pts).max())
rand_best = np.array(rand_best)

print('=== A. 合成目标（只有 2/5 维重要）===')
print(f'网格搜索最优值: {grid_best:.4f}')
print(f'随机搜索 2000 次: 均值 {rand_best.mean():.4f}, 中位数 {np.median(rand_best):.4f}, '
      f'5%-95% [{np.percentile(rand_best,5):.3f}, {np.percentile(rand_best,95):.3f}]')
print(f'随机搜索优于网格的比例: {(rand_best > grid_best).mean()*100:.1f}%')

# ---------------- B. 真实数据 ----------------
df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])


def make_rf(params):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(n_estimators=60, random_state=0,
                                       n_jobs=1, **params)),
    ])


GRID = [
    dict(max_depth=d, min_samples_leaf=l, max_features=m, criterion=c,
         min_samples_split=2)
    for d in [2, 4, 6]
    for l in [1, 3]
    for m in ['sqrt', 0.5]
    for c in ['gini', 'entropy']
]   # 24 组


def sample_random():
    return dict(
        max_depth=int(rng.integers(1, 13)),
        min_samples_leaf=int(rng.integers(1, 11)),
        max_features=float(rng.uniform(0.15, 1.0)),
        criterion=str(rng.choice(['gini', 'entropy'])),
        min_samples_split=int(rng.integers(2, 21)),
    )


N_REP = 8
grid_scores, rand_scores = [], []
grid_time, rand_time = [], []
import time
for rep in range(N_REP):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + rep)
    t0 = time.time()
    gs = [cross_val_score(make_rf(p), X, y, cv=cv, scoring='roc_auc').mean() for p in GRID]
    grid_time.append(time.time() - t0)
    t0 = time.time()
    rs = [cross_val_score(make_rf(sample_random()), X, y, cv=cv, scoring='roc_auc').mean()
          for _ in range(24)]
    rand_time.append(time.time() - t0)
    grid_scores.append(max(gs))
    rand_scores.append(max(rs))
    print(f'rep {rep}: grid={max(gs):.4f} random={max(rs):.4f}')

grid_scores = np.array(grid_scores)
rand_scores = np.array(rand_scores)
print('\n=== B. 真实数据（预算各 24 次 5 折 CV）===')
print(f'网格搜索: {grid_scores.mean():.4f} +- {grid_scores.std(ddof=1):.4f}')
print(f'随机搜索: {rand_scores.mean():.4f} +- {rand_scores.std(ddof=1):.4f}')
print(f'随机搜索胜率: {(rand_scores > grid_scores).sum()}/{N_REP}')
print(f'平均耗时 grid {np.mean(grid_time):.1f}s, random {np.mean(rand_time):.1f}s')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(rand_best, bins=30, color='#4c72b0', alpha=0.8, label='随机搜索')
axes[0].axvline(grid_best, color='#c44e52', lw=2, label=f'网格搜索 {grid_best:.3f}')
axes[0].set_xlabel('找到的最优目标值')
axes[0].set_ylabel('次数')
axes[0].set_title('合成目标：只有 2/5 维重要')
axes[0].legend()

axes[1].plot(range(1, N_REP + 1), grid_scores, 'o-', label='网格搜索', color='#c44e52')
axes[1].plot(range(1, N_REP + 1), rand_scores, 's-', label='随机搜索', color='#4c72b0')
axes[1].set_xlabel('重复次数')
axes[1].set_ylabel('最优 5 折 AUC')
axes[1].set_title('真实数据：同预算 24 次')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_grid_vs_random.png', dpi=150)
print('saved fig_grid_vs_random.png')
