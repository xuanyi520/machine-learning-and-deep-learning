# -*- coding: utf-8 -*-
"""9-B 参数扫描：找到能稳定复现 >=0.08 虚高的设置"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def run(sig_noise, u_coef, y_noise, n_pat=80, n_visit=6, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.0, n_pat)
    b = rng.normal(0, 1.5, n_pat)
    rows = []
    for i in range(n_pat):
        for v in range(n_visit):
            rows.append([i, b[i] + rng.normal(0, sig_noise),
                         u[i] + rng.normal(0, 1.2), rng.normal(0, 1.0)])
    sim = pd.DataFrame(rows, columns=['pid', 'x1', 'x2', 'x3'])
    y_pat = (u_coef * u + rng.normal(0, y_noise, n_pat) > 0).astype(int)
    sim['y'] = y_pat[sim['pid'].values]
    X, y, g = sim[['x1', 'x2', 'x3']], sim['y'].values, sim['pid'].values
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    a1 = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0),
                         scoring='roc_auc').mean()
    a2 = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=g,
                         scoring='roc_auc').mean()
    return a1, a2, a1 - a2

for sig_noise in [0.02, 0.05, 0.08]:
    for u_coef, y_noise in [(1.2, 0.6), (1.0, 0.8), (0.8, 1.0), (1.2, 1.0)]:
        vals = [run(sig_noise, u_coef, y_noise, seed=s) for s in (7, 8, 9)]
        gaps = [v[2] for v in vals]
        print(f'sig_noise={sig_noise} u_coef={u_coef} y_noise={y_noise}: '
              f'KFold={np.mean([v[0] for v in vals]):.3f} '
              f'Group={np.mean([v[1] for v in vals]):.3f} '
              f'gap={np.mean(gaps):+.3f} (逐个 {np.round(gaps,3)})')
