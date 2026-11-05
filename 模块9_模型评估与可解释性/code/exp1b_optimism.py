# -*- coding: utf-8 -*-
"""实验 1b：重复 10 次，比较 普通调参报告 vs 嵌套 CV 的平均差距
   并做标签打乱的对照实验，暴露机制"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA = r'E:\AI学习\练习代码\data\uci\Hepatitis.csv'
df = pd.read_csv(DATA, na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])

GRID_RF = {
    'clf__max_depth': [2, 3, 4, 5, 6, None],
    'clf__min_samples_leaf': [1, 2, 3, 5],
    'clf__max_features': ['sqrt', 0.5],
}
GRID_DT = {
    'clf__max_depth': [1, 2, 3, 4, 5, 6, 8, None],
    'clf__min_samples_leaf': [1, 2, 3, 5, 8],
    'clf__criterion': ['gini', 'entropy'],
    'clf__ccp_alpha': [0.0, 0.005, 0.01],
}


def make_rf():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)),
    ])


def make_dt():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', DecisionTreeClassifier(random_state=0)),
    ])


def compare(make, grid, y_used, n_rep=10, tag=''):
    naive, nested = [], []
    for rep in range(n_rep):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + rep)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=200 + rep)
        g = GridSearchCV(make(), grid, cv=inner, scoring='roc_auc', n_jobs=-1)
        g.fit(X, y_used)
        naive.append(g.best_score_)
        scores = []
        for tr, te in outer.split(X, y_used):
            gi = GridSearchCV(make(), grid, cv=inner, scoring='roc_auc', n_jobs=-1)
            gi.fit(X.iloc[tr], y_used[tr])
            scores.append(roc_auc_score(y_used[te], gi.predict_proba(X.iloc[te])[:, 1]))
        nested.append(np.mean(scores))
    naive, nested = np.array(naive), np.array(nested)
    print(f'{tag}: 普通报告 {naive.mean():.4f}+-{naive.std():.4f} | '
          f'嵌套 {nested.mean():.4f}+-{nested.std():.4f} | '
          f'平均虚高 {naive.mean()-nested.mean():+.4f} | '
          f'普通更高的次数 {(naive>nested).sum()}/{n_rep}')


print('=== 真实标签 ===')
compare(make_rf, GRID_RF, y, 3, 'RF 36 组')
compare(make_dt, GRID_DT, y, 3, 'DT 240 组')

print('\n=== 对照：把标签随机打乱（真实 AUC 应该=0.5）===')
rng = np.random.default_rng(0)
y_perm = rng.permutation(y)
compare(make_rf, GRID_RF, y_perm, 3, 'RF 36 组')
compare(make_dt, GRID_DT, y_perm, 3, 'DT 240 组')
