# -*- coding: utf-8 -*-
"""实验 1：普通 5 折 vs 嵌套 5 折；10 个种子的波动"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     cross_val_score, RepeatedStratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA = r'E:\AI学习\练习代码\data\uci\Hepatitis.csv'


def load():
    df = pd.read_csv(DATA, na_values=['?'])
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    y = (df['Class'] == 1).astype(int).values   # 1 = 死亡
    X = df.drop(columns=['Class'])
    return X, y


def make_lr():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=0.5)),
    ])


def make_rf(n_estimators=300, **kw):
    params = dict(n_estimators=n_estimators, random_state=0, n_jobs=1)
    params.update(kw)
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(**params)),
    ])


X, y = load()
print('shape', X.shape, 'deaths', y.sum(), 'survivors', (1 - y).sum())

# ---------- 1. 单次 5 折 ----------
for name, model in [('LR', make_lr()), ('RF', make_rf())]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print(f'{name} 5折 AUC: {s.round(3)} mean={s.mean():.4f} std={s.std():.4f}')

# ---------- 2. 10 个种子 ----------
print('\n=== 10 seeds ===')
for name, make in [('LR', make_lr), ('RF', make_rf)]:
    means = []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        s = cross_val_score(make(), X, y, cv=skf, scoring='roc_auc')
        means.append(s.mean())
    means = np.array(means)
    print(f'{name}: ' + ' '.join(f'{m:.3f}' for m in means))
    print(f'{name}: {means.mean():.4f} +- {means.std(ddof=1):.4f}')

# ---------- 3. 普通调参 vs 嵌套 ----------
param_grid = {
    'clf__max_depth': [2, 3, 4, 5, None],
    'clf__min_samples_leaf': [1, 2, 3, 5],
    'clf__max_features': ['sqrt', 0.5],
}

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

# 普通做法：全量数据上 GridSearchCV，然后把 best_score_ 当成性能
gs = GridSearchCV(make_rf(), param_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
gs.fit(X, y)
print('\n=== 普通做法（全量调参后直接报告 CV 分数）===')
print('best params:', gs.best_params_)
print('best_score_ (被 Optimistic 的估计):', round(gs.best_score_, 4))

# 嵌套做法：外层每折内重新调参，用外层测试折评估
outer_scores = []
for tr, te in outer.split(X, y):
    g = GridSearchCV(make_rf(), param_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
    g.fit(X.iloc[tr], y[tr])
    p = g.predict_proba(X.iloc[te])[:, 1]
    outer_scores.append(roc_auc_score(y[te], p))
outer_scores = np.array(outer_scores)
print('=== 嵌套 CV 外层各折 ===', outer_scores.round(3))
print(f'嵌套 CV AUC: {outer_scores.mean():.4f} +- {outer_scores.std():.4f}')
print(f'虚高幅度: {gs.best_score_ - outer_scores.mean():+.4f}')

# 每次外层选出的超参
print('\n每个外层折选出的参数：')
for i, (tr, te) in enumerate(outer.split(X, y)):
    g = GridSearchCV(make_rf(), param_grid, cv=inner, scoring='roc_auc', n_jobs=-1)
    g.fit(X.iloc[tr], y[tr])
    print(i, g.best_params_, round(g.best_score_, 3), '-> outer', round(roc_auc_score(y[te], g.predict_proba(X.iloc[te])[:, 1]), 3))

# ---------- 4. 重复交叉验证的折间波动 ----------
print('\n=== Repeated 5x10 ===')
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
s = cross_val_score(make_rf(), X, y, cv=rskf, scoring='roc_auc')
print(f'50 个折分数: mean={s.mean():.4f} std={s.std():.4f} min={s.min():.3f} max={s.max():.3f}')
