# -*- coding: utf-8 -*-
"""参考答案脚本：9-F 单个病人的 SHAP 解释、9-G 两个模型到底谁好"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, norm, false_discovery_control

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])

def make_lr():
    return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                     ('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=2000, C=0.5))])

def make_rf():
    return Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0,
                                                    n_jobs=-1))])

print('=' * 60)
print('练习 9-F  给医生写解释（手写 permutation SHAP）')
print('=' * 60)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
p_oof = cross_val_predict(make_rf(), X, y, cv=cv, method='predict_proba')[:, 1]
cand = np.where((y == 1) & (p_oof > 0.8))[0]
print(f'真实死亡且 OOF 预测 >0.8 的病人: {len(cand)} 个，索引 {cand[:10]}')
i = int(cand[0])
print(f'选中病人 #{i}，真实 y={y[i]}，OOF 预测 p={p_oof[i]:.3f}')

# 用全量拟合的模型做解释（部署场景），背景取训练分布的一个子样本
model = make_rf().fit(X, y)
Xmat = X.fillna(X.median()).values.astype(float)
rng = np.random.default_rng(0)
bg_idx = rng.choice(len(Xmat), 80, replace=False)
background = Xmat[bg_idx]

def shap_values(model, Xmat, background, n_perm=120, seed=0):
    rng = np.random.default_rng(seed)
    n, d = Xmat.shape
    base = model.predict_proba(background)[:, 1].mean()
    out = np.zeros((n, d))
    for i in range(n):
        phi = np.zeros(d)
        C = np.empty((n_perm, d + 1, d))
        orders = np.empty((n_perm, d), dtype=int)
        for p in range(n_perm):
            b = background[rng.integers(0, len(background))].copy()
            order = rng.permutation(d)
            orders[p] = order
            cur = b.copy()
            C[p, 0] = cur
            for s, j in enumerate(order, start=1):
                cur[j] = Xmat[i, j]
                C[p, s] = cur
        P = model.predict_proba(C.reshape(-1, d))[:, 1].reshape(n_perm, d + 1)
        for p in range(n_perm):
            for s, j in enumerate(orders[p]):
                phi[j] += P[p, s + 1] - P[p, s]
        out[i] = phi / n_perm
    return out, base

PHI, base = shap_values(model, Xmat[[i]], background, n_perm=120, seed=0)
pred = model.predict_proba(Xmat[[i]])[:, 1][0]
err = float(abs(base + PHI.sum() - pred))
print(f'\nbase={base:.4f}  预测={pred:.4f}  加性误差={err:.4f}')

s = pd.Series(PHI[0], index=X.columns).sort_values(key=np.abs, ascending=False)
print('\n贡献最大的 8 个特征（概率单位）')
for name, v in s.head(8).items():
    print(f'  {name:<16} 取值 {X.iloc[i][name]:>8}  SHAP {v:+.4f}')
top5 = s.head(5)
print(f'\n前 5 项合计 {top5.sum():+.4f}；其余 {s.iloc[5:].sum():+.4f}；'
      f'base + 全部 = {base + s.sum():.4f}')

print()
print('=' * 60)
print('练习 9-G  两个模型到底谁好')
print('=' * 60)

def oof_auc(make_model, X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(make_model(), X, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, p), p

auc_lr, auc_rf = [], []
for seed in range(10):
    a1, _ = oof_auc(make_lr, X, y, seed)
    a2, _ = oof_auc(make_rf, X, y, seed)
    auc_lr.append(a1); auc_rf.append(a2)
    print(f'seed={seed}: LR {a1:.4f}  RF {a2:.4f}')

auc_lr, auc_rf = np.array(auc_lr), np.array(auc_rf)
print(f'\nLR 10 种子: {auc_lr.mean():.4f} ± {auc_lr.std(ddof=1):.4f}')
print(f'RF 10 种子: {auc_rf.mean():.4f} ± {auc_rf.std(ddof=1):.4f}')

# OOF 预测（seed=42）用于 bootstrap 与 DeLong
p_lr = cross_val_predict(make_lr(), X, y, cv=cv, method='predict_proba')[:, 1]
p_rf = cross_val_predict(make_rf(), X, y, cv=cv, method='predict_proba')[:, 1]

def bootstrap_auc(y_true, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], p[idx]))
    return np.array(vals)

for name, p in [('LR', p_lr), ('RF', p_rf)]:
    b = bootstrap_auc(y, p, seed=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'{name} OOF AUC {roc_auc_score(y, p):.4f}，'
          f'95% CI {lo:.4f}-{hi:.4f}')

# 配对 t：同一批 5 折
cv42 = StratifiedKFold(5, shuffle=True, random_state=42)
fold_lr, fold_rf = [], []
for tr, te in cv42.split(X, y):
    m1 = make_lr().fit(X.iloc[tr], y[tr])
    m2 = make_rf().fit(X.iloc[tr], y[tr])
    fold_lr.append(roc_auc_score(y[te], m1.predict_proba(X.iloc[te])[:, 1]))
    fold_rf.append(roc_auc_score(y[te], m2.predict_proba(X.iloc[te])[:, 1]))
t, pt = ttest_rel(fold_rf, fold_lr)
print(f'\n配对 t: LR {np.round(fold_lr,3)}')
print(f'        RF {np.round(fold_rf,3)}')
print(f'        t={t:.3f}, p={pt:.4f}, 平均差={np.mean(np.array(fold_rf)-np.array(fold_lr)):+.4f}')

# ---------- DeLong 完整实现 ----------
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
    """返回 (auc1, auc2, z, p)。H0: 两个 AUC 相等。"""
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
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * norm.sf(abs(z))

a1, a2, z, pdl = delong_test(y, p_rf, p_lr)
print(f'\nDeLong RF vs LR: AUC {a1:.4f} vs {a2:.4f}, z={z:.3f}, p={pdl:.4f}')
print('自检 RF vs RF:', delong_test(y, p_rf, p_rf))
print('sklearn AUC 校验:', round(roc_auc_score(y, p_rf), 12))

# ---------- 6 个模型，15 对比较 ----------
models = {
    'LR': make_lr,
    'RF': make_rf,
    'DT3': lambda: Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                             ('clf', DecisionTreeClassifier(max_depth=3, random_state=0))]),
    'KNN': lambda: Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                             ('sc', StandardScaler()),
                             ('clf', KNeighborsClassifier(n_neighbors=15))]),
    'SVM': lambda: Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                             ('sc', StandardScaler()),
                             ('clf', SVC(probability=True, random_state=0))]),
    'GB': lambda: Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                            ('clf', GradientBoostingClassifier(random_state=0))]),
}
P, aucs = {}, {}
for name, mk in models.items():
    P[name] = cross_val_predict(mk(), X, y, cv=cv, method='predict_proba')[:, 1]
    aucs[name] = roc_auc_score(y, P[name])
    print(f'{name:<4} OOF AUC {aucs[name]:.4f}')

names = list(models)
pairs, pvals = [], []
for i_ in range(len(names)):
    for j_ in range(i_ + 1, len(names)):
        a, b_, z_, p_ = delong_test(y, P[names[i_]], P[names[j_]])
        pairs.append((names[i_], names[j_], a, b_, z_, p_))
        pvals.append(p_)
pvals = np.array(pvals)
qvals = false_discovery_control(pvals, method='bh')
print(f'\n共 {len(pairs)} 对比较，Bonferroni 阈值 {0.05/len(pairs):.4f}')
print(f'{"比较":<12}{"AUC1":>7}{"AUC2":>7}{"z":>8}{"p":>9}{"BH q":>9}{"Bonf":>7}')
for (n1, n2, a, b_, z_, p_), q in zip(pairs, qvals):
    print(f'{n1+" vs "+n2:<12}{a:>7.3f}{b_:>7.3f}{z_:>8.2f}{p_:>9.4f}{q:>9.4f}'
          f'{"  是" if p_ < 0.05/len(pairs) else "  否":>7}')
print(f'原始 p<0.05: {(pvals<0.05).sum()} 对；BH q<0.05: {(qvals<0.05).sum()} 对；'
      f'Bonferroni: {(pvals<0.05/len(pairs)).sum()} 对')
