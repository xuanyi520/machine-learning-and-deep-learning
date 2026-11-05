# -*- coding: utf-8 -*-
"""参考答案脚本：9-D 同预算搜索对比、9-E 概率校准"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from scipy.stats import randint, uniform

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values
X = df.drop(columns=['Class'])

print('=' * 60)
print('练习 9-D  同预算（24 次 5 折）网格 vs 随机')
print('=' * 60)

# 为省时间用 60 棵树；方向不变
def make_rf(depth=None, leaf=1, feat='sqrt', crit='gini', split=2):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', RandomForestClassifier(
            n_estimators=60, max_depth=depth, min_samples_leaf=leaf,
            max_features=feat, criterion=crit, min_samples_split=split,
            random_state=0, n_jobs=-1)),
    ])

grid = {
    'clf__max_depth': [2, 4, 6],
    'clf__min_samples_leaf': [1, 3],
    'clf__max_features': ['sqrt', 0.5],
    'clf__criterion': ['gini', 'entropy'],
}
dist = {
    'clf__max_depth': randint(1, 13),
    'clf__min_samples_leaf': randint(1, 11),
    'clf__max_features': uniform(0.15, 0.85),
    'clf__criterion': ['gini', 'entropy'],
    'clf__min_samples_split': randint(2, 21),
}

g_scores, r_scores = [], []
for rep in range(8):
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    gs = GridSearchCV(make_rf(), grid, cv=cv, scoring='roc_auc',
                      n_jobs=-1).fit(X, y)
    rs = RandomizedSearchCV(make_rf(), dist, n_iter=24, cv=cv,
                            scoring='roc_auc', random_state=rep,
                            n_jobs=-1).fit(X, y)
    g_scores.append(gs.best_score_)
    r_scores.append(rs.best_score_)
    print(f'rep{rep+1}: 网格 {gs.best_score_:.4f}  随机 {rs.best_score_:.4f}  '
          f'{"网格胜" if gs.best_score_ > rs.best_score_ else "随机胜"}')

g_scores, r_scores = np.array(g_scores), np.array(r_scores)
print(f'\n网格 24 组: {g_scores.mean():.4f} +- {g_scores.std(ddof=1):.4f}')
print(f'随机 24 组: {r_scores.mean():.4f} +- {r_scores.std(ddof=1):.4f}')
print(f'差值 {g_scores.mean()-r_scores.mean():+.4f}，'
      f'网格胜 {(g_scores > r_scores).sum()}/8')

print()
print('=' * 60)
print('练习 9-E  概率校准（15 次 70/30）')
print('=' * 60)

from xgboost import XGBClassifier

def base_xgb():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('clf', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.3,
                              subsample=0.9, colsample_bytree=0.8,
                              eval_metric='logloss', random_state=0,
                              n_jobs=4)),
    ])

def ece_score(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.sum() / len(p) * abs(y_true[m].mean() - p[m].mean())
    return e

res = {k: {'brier': [], 'll': [], 'ece': [], 'auc': []}
       for k in ['未校准', 'Platt', 'Isotonic']}
pooled = {k: [] for k in res}

for seed in range(15):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=seed)
    models = {
        '未校准': base_xgb().fit(Xtr, ytr),
        'Platt': CalibratedClassifierCV(base_xgb(), method='sigmoid',
                                        cv=5).fit(Xtr, ytr),
        'Isotonic': CalibratedClassifierCV(base_xgb(), method='isotonic',
                                           cv=5).fit(Xtr, ytr),
    }
    for name, m in models.items():
        p = m.predict_proba(Xte)[:, 1]
        res[name]['brier'].append(brier_score_loss(yte, p))
        res[name]['ll'].append(log_loss(yte, p))
        res[name]['ece'].append(ece_score(yte, p))
        res[name]['auc'].append(roc_auc_score(yte, p))
        pooled[name].append((yte, p))

print(f'{"模型":<10}{"Brier":>16}{"LogLoss":>18}{"ECE":>16}{"AUC":>8}')
for name in res:
    r = res[name]
    f = lambda k: f'{np.mean(r[k]):.4f} ± {np.std(r[k], ddof=1):.4f}'
    print(f'{name:<10}{f("brier"):>18}{f("ll"):>20}{f("ece"):>18}'
          f'{np.mean(r["auc"]):>9.4f}')

# 合并测试集可靠性图
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='完美校准')
for name, mk in [('未校准', 'o'), ('Platt', 's'), ('Isotonic', '^')]:
    yy = np.concatenate([t[0] for t in pooled[name]])
    pp = np.concatenate([t[1] for t in pooled[name]])
    frac, meanp = calibration_curve(yy, pp, n_bins=8, strategy='uniform')
    ax.plot(meanp, frac, marker=mk, label=name)
ax.set_xlabel('预测概率')
ax.set_ylabel('实际死亡率')
ax.set_title('合并 15 次测试集的可靠性图')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_ans_calibration.png', dpi=150)
plt.close()

# 关键分箱：合并后预测 >= 0.9 的那一箱实际死亡率多少
yy = np.concatenate([t[0] for t in pooled['未校准']])
pp = np.concatenate([t[1] for t in pooled['未校准']])
hi = pp >= 0.9
print(f'\n未校准模型预测 >=0.9 的样本 {hi.sum()} 个，实际死亡率 {yy[hi].mean():.3f}')
mid = (pp >= 0.5) & (pp < 0.7)
print(f'未校准模型预测 0.5-0.7 的样本 {mid.sum()} 个，实际死亡率 {yy[mid].mean():.3f}')
