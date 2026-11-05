# -*- coding: utf-8 -*-
"""实验：概率校准
   XGBoost 未校准 vs Platt vs Isotonic；随机森林作为对照"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])


def base_xgb():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.3,
                              subsample=0.9, colsample_bytree=0.8,
                              eval_metric='logloss', random_state=0, n_jobs=4)),
    ])


def base_rf():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                       random_state=0, n_jobs=1)),
    ])


def ece(y_true, p, bins=8):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return e


N_SPLIT = 15
rows = {'XGB 未校准': [], 'XGB + Platt': [], 'XGB + Isotonic': [], 'RF 未校准': []}
pooled = {k: [] for k in rows}
pooled_y = []
for rep in range(N_SPLIT):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=rep)
    tr, te = next(sss.split(X, y))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

    models = {
        'XGB 未校准': base_xgb().fit(Xtr, ytr),
        'XGB + Platt': CalibratedClassifierCV(base_xgb(), method='sigmoid', cv=5).fit(Xtr, ytr),
        'XGB + Isotonic': CalibratedClassifierCV(base_xgb(), method='isotonic', cv=5).fit(Xtr, ytr),
        'RF 未校准': base_rf().fit(Xtr, ytr),
    }
    for k, m in models.items():
        p = m.predict_proba(Xte)[:, 1]
        rows[k].append([brier_score_loss(yte, p), log_loss(yte, p), ece(yte, p),
                        roc_auc_score(yte, p)])
        pooled[k].append(p)
    pooled_y.append(yte)

print(f'=== {N_SPLIT} 次 70/30 划分，测试集 mean +- std ===')
print(f'{"model":16s} {"Brier":>16s} {"LogLoss":>16s} {"ECE":>14s} {"AUC":>14s}')
for k, v in rows.items():
    v = np.array(v)
    print(f'{k:16s} {v[:,0].mean():.4f}+-{v[:,0].std():.4f}  '
          f'{v[:,1].mean():.4f}+-{v[:,1].std():.4f}  '
          f'{v[:,2].mean():.4f}+-{v[:,2].std():.4f}  '
          f'{v[:,3].mean():.4f}+-{v[:,3].std():.4f}')

py = np.concatenate(pooled_y)
print('\n合并后的可靠性分箱（8 箱）：')
bins_out = {}
for k in pooled:
    p = np.concatenate(pooled[k])
    edges = np.linspace(0, 1, 9)
    idx = np.digitize(p, edges[1:-1])
    print(f'--- {k} ---')
    print(f'{"bin":>12s} {"n":>5s} {"pred":>7s} {"obs":>7s}')
    b = []
    for i in range(8):
        m = idx == i
        if m.sum() > 0:
            print(f'[{edges[i]:.2f},{edges[i+1]:.2f}) {m.sum():5d} {p[m].mean():7.3f} {py[m].mean():7.3f}')
            b.append((p[m].mean(), py[m].mean(), m.sum()))
    bins_out[k] = b

fig, ax = plt.subplots(figsize=(5.8, 5.2))
colors = {'XGB 未校准': '#c44e52', 'XGB + Platt': '#4c72b0',
          'XGB + Isotonic': '#55a868', 'RF 未校准': '#8172b2'}
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='完美校准')
for k in ['XGB 未校准', 'XGB + Platt', 'XGB + Isotonic', 'RF 未校准']:
    xs = [a for a, b, n in bins_out[k] if n >= 5]
    ys = [b for a, b, n in bins_out[k] if n >= 5]
    ax.plot(xs, ys, 'o-', color=colors[k], label=k, ms=5)
ax.set_xlabel('预测概率')
ax.set_ylabel('实际阳性率')
ax.set_title('可靠性图（15 次划分的测试集合并）')
ax.legend(loc='upper left', fontsize=8.5)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_calibration.png', dpi=150)
print('saved fig_calibration.png')
