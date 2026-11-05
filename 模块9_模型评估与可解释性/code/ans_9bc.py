# -*- coding: utf-8 -*-
"""参考答案脚本：9-B 病人泄露、9-C 学习曲线"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import (KFold, GroupKFold, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print('=' * 60)
print('练习 9-B  病人泄露复现')
print('=' * 60)

# 80 个病人 × 6 次随访 = 480 行
rng = np.random.default_rng(7)
n_pat, n_visit = 80, 6
u = rng.normal(0, 1.0, n_pat)      # 真实风险（病人层面）
b = rng.normal(0, 1.5, n_pat)      # 病人"签名"，与结局无关
rows = []
for i in range(n_pat):
    for v in range(n_visit):
        rows.append([i,
                     b[i] + rng.normal(0, 0.02),      # x1 签名：随访间几乎不变
                     u[i] + rng.normal(0, 1.2),       # x2 很弱的真实信号
                     rng.normal(0, 1.0)])             # x3 噪声
sim = pd.DataFrame(rows, columns=['pid', 'x1', 'x2', 'x3'])
y_pat = (1.0 * u + rng.normal(0, 0.8, n_pat) > 0).astype(int)
sim['y'] = y_pat[sim['pid'].values]
print(f'形状 {sim.shape}，病人数 {sim.pid.nunique()}，'
      f'阳性行数 {sim.y.sum()}，阳性病人数 {len(set(sim.loc[sim.y==1,"pid"]))}')

# x1 组内相关系数，说明"签名"有多稳定
icc = sim.groupby('pid')['x1'].mean().var() / sim['x1'].var()
print(f'x1 病人间方差占比 {icc:.3f}（接近 1 表示随访间几乎不变）')

Xs, ys, groups = sim[['x1', 'x2', 'x3']], sim['y'].values, sim['pid'].values
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                            random_state=0, n_jobs=-1)

kf = KFold(5, shuffle=True, random_state=0)
gkf = GroupKFold(5)
a1 = cross_val_score(rf, Xs, ys, cv=kf, scoring='roc_auc')
a2 = cross_val_score(rf, Xs, ys, cv=gkf, groups=groups, scoring='roc_auc')
print('按行 KFold      :', np.round(a1, 3), f'{a1.mean():.4f}')
print('按病人 GroupKFold:', np.round(a2, 3), f'{a2.mean():.4f}')
print(f'虚高 {a1.mean()-a2.mean():+.4f}')

# 每折：测试集病人有多大比例在训练集里出现过
print('\n每折病人重叠比例（KFold）')
for k, (tr, te) in enumerate(kf.split(Xs, ys), 1):
    tr_pid, te_pid = set(groups[tr]), set(groups[te])
    ratio = len(te_pid & tr_pid) / len(te_pid)
    print(f'  折{k}: 测试病人 {len(te_pid)} 个，出现在训练集 {len(te_pid & tr_pid)} 个，'
          f'比例 {ratio:.1f}')

print('每折病人重叠比例（GroupKFold）')
for k, (tr, te) in enumerate(gkf.split(Xs, ys, groups=groups), 1):
    tr_pid, te_pid = set(groups[tr]), set(groups[te])
    ratio = len(te_pid & tr_pid) / len(te_pid)
    print(f'  折{k}: 测试病人 {len(te_pid)} 个，出现在训练集 {len(te_pid & tr_pid)} 个，'
          f'比例 {ratio:.1f}')
    assert len(tr_pid & te_pid) == 0, 'GroupKFold 出现病人跨折'

# 只保留 x1 会怎样：证明模型靠的是签名
a1_x1 = cross_val_score(rf, sim[['x1']], ys, cv=kf, scoring='roc_auc')
a2_x1 = cross_val_score(rf, sim[['x1']], ys, cv=gkf, groups=groups, scoring='roc_auc')
print(f'\n只用 x1（与结局无关的签名）: KFold {a1_x1.mean():.3f} '
      f'vs GroupKFold {a2_x1.mean():.3f}')

# 特征重要性
rf.fit(Xs, ys)
print('全量训练后特征重要性:', dict(zip(Xs.columns, np.round(rf.feature_importances_, 3))))

print()
print('=' * 60)
print('练习 9-C  学习曲线判读')
print('=' * 60)

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])

def make_rf():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
    ])

cv = StratifiedKFold(5, shuffle=True, random_state=42)
sizes, tr_s, va_s = learning_curve(
    make_rf(), X, y, cv=cv, scoring='roc_auc',
    train_sizes=np.linspace(0.15, 1.0, 8), shuffle=True, random_state=1, n_jobs=-1)

for n, a, b_ in zip(sizes, tr_s.mean(1), va_s.mean(1)):
    print(f'n={n:3.0f} train={a:.3f} val={b_:.3f} gap={a-b_:.3f}')

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sizes, tr_s.mean(1), 'o-', label='训练集 AUC')
ax.fill_between(sizes, tr_s.mean(1) - tr_s.std(1), tr_s.mean(1) + tr_s.std(1), alpha=0.15)
ax.plot(sizes, va_s.mean(1), 's-', label='验证集 AUC')
ax.fill_between(sizes, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.15)
ax.set_xlabel('训练样本量')
ax.set_ylabel('AUC（5 折交叉验证）')
ax.set_title('Hepatitis 随机森林学习曲线')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_ans_learning_curve.png', dpi=150)
plt.close()

i48 = int(np.argmin(np.abs(sizes - 48)))
i124 = int(np.argmin(np.abs(sizes - 124)))
print(f'48 例验证 {va_s.mean(1)[i48]:.3f}，124 例验证 {va_s.mean(1)[i124]:.3f}，'
      f'涨幅 {va_s.mean(1)[i124]-va_s.mean(1)[i48]:+.3f}')
