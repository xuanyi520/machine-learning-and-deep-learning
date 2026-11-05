# -*- coding: utf-8 -*-
"""实验：学习曲线 + 验证曲线"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])


def make_rf(**kw):
    p = dict(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1)
    p.update(kw)
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(**p)),
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sizes, tr_s, va_s = learning_curve(
    make_rf(), X, y, cv=cv, scoring='roc_auc',
    train_sizes=np.linspace(0.15, 1.0, 8), shuffle=True, random_state=1, n_jobs=-1)

print('=== 学习曲线 ===')
print(f'{"n":>5} {"train":>8} {"val":>8} {"gap":>8}')
for n, a, b in zip(sizes, tr_s.mean(1), va_s.mean(1)):
    print(f'{n:5.0f} {a:8.3f} {b:8.3f} {a-b:8.3f}')

# 验证曲线：max_depth
depths = [1, 2, 3, 4, 5, 6, 8, 10]
tr_v, va_v = validation_curve(make_rf(), X, y, param_name='clf__max_depth',
                              param_range=depths, cv=cv, scoring='roc_auc', n_jobs=-1)
print('\n=== 验证曲线 max_depth ===')
for d, a, b in zip(depths, tr_v.mean(1), va_v.mean(1)):
    print(f'{d:3d} train={a:.3f} val={b:.3f} gap={a-b:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(sizes, tr_s.mean(1), 'o-', label='训练集 AUC', color='#4c72b0')
axes[0].fill_between(sizes, tr_s.mean(1) - tr_s.std(1), tr_s.mean(1) + tr_s.std(1), alpha=0.15, color='#4c72b0')
axes[0].plot(sizes, va_s.mean(1), 's-', label='验证集 AUC', color='#c44e52')
axes[0].fill_between(sizes, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.15, color='#c44e52')
axes[0].set_xlabel('训练样本量')
axes[0].set_ylabel('AUC')
axes[0].set_title('学习曲线：随机森林')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

axes[1].plot(depths, tr_v.mean(1), 'o-', label='训练集', color='#4c72b0')
axes[1].plot(depths, va_v.mean(1), 's-', label='验证集', color='#c44e52')
axes[1].set_xlabel('max_depth')
axes[1].set_ylabel('AUC')
axes[1].set_title('验证曲线：树深')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_learning_curve.png', dpi=150)
print('saved fig_learning_curve.png')
