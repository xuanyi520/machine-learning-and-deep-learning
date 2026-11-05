# -*- coding: utf-8 -*-
"""实验：病人泄露（KFold vs GroupKFold）与时间泄露（KFold vs TimeSeriesSplit）"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(7)

# ---------- A. 纵向数据：同一病人多次随访 ----------
n_pat = 60
n_visit = 8
u = rng.normal(0, 1.0, n_pat)           # 病人层面的真实风险（不可直接观测）
b = rng.normal(0, 1.5, n_pat)           # 病人特有的“签名”：体质/设备/医院，和结局无关
rows = []
for i in range(n_pat):
    for v in range(n_visit):
        x1 = b[i] + rng.normal(0, 0.08)   # 签名指标，随访间几乎不变，但与结局无关
        x2 = u[i] + rng.normal(0, 1.2)    # 有信号但很弱
        x3 = rng.normal(0, 1.0)
        rows.append([i, v, x1, x2, x3])
sim = pd.DataFrame(rows, columns=['pid', 'visit', 'x1', 'x2', 'x3'])
# 结局是病人层面的（是否死亡），同一病人所有随访标签相同
logit = 1.2 * u + rng.normal(0, 0.6, n_pat)
y_pat = (logit > 0).astype(int)
sim['y'] = y_pat[sim['pid'].values]
X = sim[['x1', 'x2', 'x3']]
y = sim['y'].values
print('模拟纵向数据:', sim.shape, '病人', n_pat, '事件病人', y_pat.sum())

rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
gkf = GroupKFold(n_splits=5)
auc_kf = cross_val_score(rf, X, y, cv=kf, scoring='roc_auc')
auc_gkf = cross_val_score(rf, X, y, cv=gkf, groups=sim['pid'], scoring='roc_auc')
print('按行 KFold   :', auc_kf.round(3), f'mean={auc_kf.mean():.4f}')
print('按病人 GroupKFold:', auc_gkf.round(3), f'mean={auc_gkf.mean():.4f}')
print('虚高:', f'{auc_kf.mean()-auc_gkf.mean():+.4f}')

# 每个病人在训练/测试折中的分布：KFold 下测试病人有多少比例同时出现在训练集
leak = []
for tr, te in kf.split(X):
    pids_te = set(sim['pid'].iloc[te])
    pids_tr = set(sim['pid'].iloc[tr])
    leak.append(len(pids_te & pids_tr) / len(pids_te))
print('KFold 每折中“测试病人同时出现在训练集”的比例:', np.round(leak, 2))

# ---------- B. 时间漂移 ----------
n = 300
t = np.arange(n)
# 前 150 例：x 与 y 正相关；后 150 例：关系反转（治疗方案改变）
x = rng.normal(0, 1, n)
y = np.zeros(n, int)
y[:150] = (x[:150] + rng.normal(0, 1, 150) > 0).astype(int)
y[150:] = (-x[150:] + rng.normal(0, 1, 150) > 0).astype(int)
ts = pd.DataFrame({'x': x, 'y': y})
X2, y2 = ts[['x']], ts['y']

kf2 = KFold(n_splits=5, shuffle=True, random_state=0)
tscv = TimeSeriesSplit(n_splits=5)
a1 = cross_val_score(rf, X2, y2, cv=kf2, scoring='roc_auc')
a2 = cross_val_score(rf, X2, y2, cv=tscv, scoring='roc_auc')
print('\n时间漂移数据')
print('随机 KFold :', a1.round(3), f'mean={a1.mean():.4f}')
print('TimeSeriesSplit:', a2.round(3), f'mean={a2.mean():.4f}')

# ---------- C. 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(['按行 KFold', '按病人 GroupKFold'], [auc_kf.mean(), auc_gkf.mean()],
            color=['#c44e52', '#4c72b0'], width=0.55)
for i, v in enumerate([auc_kf.mean(), auc_gkf.mean()]):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('AUC')
axes[0].set_title('病人泄露：同一病人跨折')

axes[1].bar(['随机 KFold', 'TimeSeriesSplit'], [a1.mean(), a2.mean()],
            color=['#c44e52', '#55a868'], width=0.55)
for i, v in enumerate([a1.mean(), a2.mean()]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12)
axes[1].set_ylim(0, 1.05)
axes[1].set_title('时间泄露：关系随时间反转')
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_group_time_leak.png', dpi=150)
print('saved fig_group_time_leak.png')
