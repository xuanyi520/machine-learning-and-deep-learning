# -*- coding: utf-8 -*-
"""实验：多种子、bootstrap 置信区间、配对 t 检验、DeLong 检验、多重比较"""
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])


def make_lr(C=0.5):
    return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                     ('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=2000, C=C))])


def make_rf():
    return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1))])


def make_dt():
    return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                     ('clf', DecisionTreeClassifier(max_depth=3, random_state=0))])


def oof(model, seed=42):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]


# ---------- 1. 10 个种子 ----------
print('=== 1. 10 个种子的 AUC ===')
res = {}
for name, make in [('LR', make_lr), ('RF', make_rf), ('DT(depth=3)', make_dt)]:
    vals = []
    for seed in range(10):
        p = oof(make(), seed)
        vals.append(roc_auc_score(y, p))
    vals = np.array(vals)
    res[name] = vals
    print(f'{name:12s} ' + ' '.join(f'{v:.3f}' for v in vals))
    print(f'{"":12s} mean={vals.mean():.4f} std={vals.std(ddof=1):.4f}')

# ---------- 2. Bootstrap CI ----------
print('\n=== 2. Bootstrap 95% CI (2000 次) ===')
p_lr = oof(make_lr(), 42)
p_rf = oof(make_rf(), 42)
p_dt = oof(make_dt(), 42)


def boot_auc(y_true, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        out.append(roc_auc_score(y_true[idx], p[idx]))
    return np.array(out)


for name, p in [('LR', p_lr), ('RF', p_rf), ('DT', p_dt)]:
    b = boot_auc(y, p)
    print(f'{name}: AUC={roc_auc_score(y, p):.4f}  95% CI [{np.percentile(b,2.5):.4f}, {np.percentile(b,97.5):.4f}]')

# ---------- 3. DeLong ----------
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


def delong(y_true, p1, p2):
    order = np.argsort(-y_true)
    yt = y_true[order]
    preds = np.vstack((p1[order], p2[order]))
    m = int(yt.sum())
    n = len(yt) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(2)])
    ty = np.array([_midrank(neg[r]) for r in range(2)])
    tz = np.array([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    S = sx / m + sy / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


print('\n=== 3. DeLong 检验 ===')
for a, b, na, nb in [(p_rf, p_lr, 'RF', 'LR'), (p_rf, p_dt, 'RF', 'DT'),
                     (p_lr, p_dt, 'LR', 'DT'), (p_rf, p_rf, 'RF', 'RF自己')]:
    auc1, auc2, z, p = delong(y, a, b)
    print(f'{na} vs {nb}: AUC {auc1:.4f} vs {auc2:.4f}  z={z:+.3f}  p={p:.4f}')

# ---------- 4. 配对 t 检验（同一批折）----------
print('\n=== 4. 配对 t 检验（每折 AUC）===')
cv = StratifiedKFold(5, shuffle=True, random_state=42)
fold_lr, fold_rf = [], []
for tr, te in cv.split(X, y):
    m1 = make_lr().fit(X.iloc[tr], y[tr])
    m2 = make_rf().fit(X.iloc[tr], y[tr])
    fold_lr.append(roc_auc_score(y[te], m1.predict_proba(X.iloc[te])[:, 1]))
    fold_rf.append(roc_auc_score(y[te], m2.predict_proba(X.iloc[te])[:, 1]))
fold_lr, fold_rf = np.array(fold_lr), np.array(fold_rf)
print('LR folds', fold_lr.round(3), 'RF folds', fold_rf.round(3))
t, p = ttest_rel(fold_rf, fold_lr)
print(f'paired t={t:.3f}, p={p:.4f}, 平均差={np.mean(fold_rf-fold_lr):+.4f}')

# ---------- 5. 多重比较 ----------
print('\n=== 5. 多重比较模拟（6 个无信号模型，15 对比较）===')
rng = np.random.default_rng(0)
n_sim = 1000
any_raw = 0
any_bonf = 0
for s in range(n_sim):
    scores = rng.normal(size=(len(y), 6))
    ps = []
    for i in range(6):
        for j in range(i + 1, 6):
            ps.append(delong(y, scores[:, i], scores[:, j])[3])
    ps = np.array(ps)
    any_raw += (ps < 0.05).any()
    any_bonf += (ps < 0.05 / len(ps)).any()
print(f'不校正：至少一对 p<0.05 的比例 = {any_raw/n_sim*100:.1f}%')
print(f'Bonferroni（阈值 {0.05/15:.4f}）：{any_bonf/n_sim*100:.1f}%')
print(f'理论值：不校正 {100*(1-0.95**15):.1f}%')
