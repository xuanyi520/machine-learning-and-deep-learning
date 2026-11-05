# -*- coding: utf-8 -*-
"""实验：PDP / ICE / 手写 permutation-SHAP（本机未装 shap）
   全局 summary + 单个病例 waterfall + 相关特征导致归因分摊
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
y = (df['Class'] == 1).astype(int).values

FEATS = ['Age', 'Bilirubin', 'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime',
         'Ascites', 'Spiders']
CN = {'Age': '年龄', 'Bilirubin': '胆红素', 'Alk_Phosphate': '碱性磷酸酶',
      'Sgot': 'AST(SGOT)', 'Albumin': '白蛋白', 'Protime': '凝血酶原时间',
      'Ascites': '腹水', 'Spiders': '蜘蛛痣'}
Xraw = df[FEATS].copy()
# UCI 原始编码：1=有，2=无。改成 1/0，图才好读
Xraw['Ascites'] = df['Ascites'].map({1: 1.0, 2: 0.0})
Xraw['Spiders'] = df['Spiders'].map({1: 1.0, 2: 0.0})
med = Xraw.median()
Xi = Xraw.fillna(med).astype(float)

model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                               random_state=0, n_jobs=1).fit(Xi, y)
print('训练集 AUC（仅用于解释，不当作性能估计）:',
      round(__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(
          y, model.predict_proba(Xi)[:, 1]), 3))

# ---------------- PDP / ICE ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.ravel(), ['Bilirubin', 'Albumin', 'Protime', 'Age']):
    PartialDependenceDisplay.from_estimator(
        model, Xi, [FEATS.index(feat)], kind='average', ax=ax,
        line_kw={'color': '#c44e52', 'lw': 2})
    ax.set_title(f'PDP: {CN[feat]}')
    ax.set_xlabel(CN[feat])
    ax.set_ylabel('预测死亡概率')
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_pdp.png', dpi=150)
print('saved fig_pdp.png')

# ICE：单个特征，抽 30 个病人
fig, ax = plt.subplots(figsize=(6.5, 4.5))
PartialDependenceDisplay.from_estimator(
    model, Xi, [FEATS.index('Bilirubin')], kind='both', ax=ax,
    subsample=30, random_state=0,
    line_kw={'color': '#c44e52', 'lw': 2})
ax.set_title('ICE + 平均线：胆红素')
ax.set_xlabel('胆红素')
ax.set_ylabel('预测死亡概率')
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_ice.png', dpi=150)
print('saved fig_ice.png')

# ---------------- 手写 permutation SHAP ----------------
rng = np.random.default_rng(0)
Xi_arr = Xi.values
bg = Xi_arr[rng.choice(len(Xi_arr), 80, replace=False)]
base = model.predict_proba(bg)[:, 1].mean()


# 逐病人、逐排列，批量预测该排列的 d+1 个 coalition
def shap_values_final(Xmat, n_perm=100):
    out = np.zeros_like(Xmat, dtype=float)
    d = Xmat.shape[1]
    for i, x in enumerate(Xmat):
        phi = np.zeros(d)
        C = np.empty((n_perm, d + 1, d))
        orders = np.empty((n_perm, d), dtype=int)
        for p in range(n_perm):
            b = bg[rng.integers(0, len(bg))].copy()
            order = rng.permutation(d)
            orders[p] = order
            cur = b.copy()
            C[p, 0] = cur
            for s, j in enumerate(order, start=1):
                cur[j] = x[j]
                C[p, s] = cur
        P = model.predict_proba(C.reshape(-1, d))[:, 1].reshape(n_perm, d + 1)
        for p in range(n_perm):
            for s, j in enumerate(orders[p]):
                phi[j] += P[p, s + 1] - P[p, s]
        out[i] = phi / n_perm
    return out


print('计算 SHAP（155 人 × 100 排列）...')
PHI = shap_values_final(Xi_arr, n_perm=100)
pred = model.predict_proba(Xi_arr)[:, 1]
err = np.abs(base + PHI.sum(1) - pred)
print(f'加性检验：|base + sum(phi) - f(x)| 最大 {err.max():.6f}，平均 {err.mean():.6f}')
print(f'base(背景平均预测) = {base:.4f}')

# 全局重要性
mean_abs = np.abs(PHI).mean(0)
order = np.argsort(mean_abs)
print('\n=== 全局 mean|SHAP| ===')
for j in order[::-1]:
    print(f'{CN[FEATS[j]]:10s} {mean_abs[j]:.4f}')

# summary beeswarm
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.4]})
axes[0].barh([CN[FEATS[j]] for j in order], mean_abs[order], color='#4c72b0')
axes[0].set_xlabel('mean |SHAP|（概率单位）')
axes[0].set_title('全局重要性')

for j in order:
    vals = PHI[:, j]
    feat = Xi_arr[:, j]
    c = (feat - feat.min()) / (feat.max() - feat.min() + 1e-9)
    yj = order.tolist().index(j) + rng.normal(0, 0.08, len(vals))
    axes[1].scatter(vals, yj, c=c, cmap='coolwarm', s=10, alpha=0.75)
axes[1].set_yticks(range(len(order)))
axes[1].set_yticklabels([CN[FEATS[j]] for j in order])
axes[1].axvline(0, color='k', lw=0.8)
axes[1].set_xlabel('SHAP 值（推高风险为正）')
axes[1].set_title('每个病人的归因分布（颜色=特征值高低）')
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_shap_summary.png', dpi=150)
print('saved fig_shap_summary.png')

# ---------------- 单个病例 waterfall ----------------
cand = np.where((y == 1) & (pred > 0.7))[0]
idx = cand[np.argmax(pred[cand])]
print(f'\n=== 病例 #{idx} ===')
print('真实结局:', '死亡' if y[idx] == 1 else '存活', ' 预测概率:', round(pred[idx], 3))
for j in np.argsort(-np.abs(PHI[idx]))[:8]:
    arrow = '↑' if PHI[idx, j] > 0 else '↓'
    print(f'{CN[FEATS[j]]:10s} = {Xi_arr[idx, j]:7.2f}   SHAP {PHI[idx, j]:+.3f} {arrow}')

contrib = PHI[idx]
sid = np.argsort(np.abs(contrib))[::-1][:8]          # 贡献最大的在前
vals = contrib[sid]
lefts, c = [], base
for v in vals:
    lefts.append(c)
    c += v
ypos = np.arange(len(vals))[::-1]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = ['#c44e52' if v > 0 else '#4c72b0' for v in vals]
ax.barh(ypos, vals, left=lefts, color=colors, height=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels([f'{CN[FEATS[j]]}={Xi_arr[idx, j]:.1f}' for j in sid])
ax.axvline(base, color='gray', ls='--', lw=1)
ax.axvline(pred[idx], color='k', ls=':', lw=1.2)
ax.text(base, len(vals) - 0.2, f' base={base:.2f}', color='gray', va='bottom')
ax.text(pred[idx], -0.8, f' 预测={pred[idx]:.2f}', va='top', fontsize=9)
ax.set_xlabel('预测概率（从 base 累加到最终预测）')
ax.set_title('单病例 waterfall：每一段把风险推高/拉低多少')
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块9_模型评估与可解释性\fig_shap_waterfall.png', dpi=150)
print('saved fig_shap_waterfall.png')

# ---------------- 相关特征：把 Sgot 复制一份 ----------------
Xi2 = Xi.copy()
Xi2['Sgot_copy'] = Xi2['Sgot'] + rng.normal(0, 0.01, len(Xi2))
model2 = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                random_state=0, n_jobs=1).fit(Xi2, y)
Xi2_arr = Xi2.values
# 用同一个背景
bg2 = Xi2_arr[rng.choice(len(Xi2_arr), 80, replace=False)]


def shap_generic(model, Xmat, bgmat, n_perm=60):
    d = Xmat.shape[1]
    out = np.zeros_like(Xmat, dtype=float)
    for i, x in enumerate(Xmat):
        phi = np.zeros(d)
        C = np.empty((n_perm, d + 1, d))
        orders = np.empty((n_perm, d), dtype=int)
        for p in range(n_perm):
            b = bgmat[rng.integers(0, len(bgmat))].copy()
            o = rng.permutation(d)
            orders[p] = o
            cur = b.copy(); C[p, 0] = cur
            for s, j in enumerate(o, start=1):
                cur[j] = x[j]; C[p, s] = cur
        P = model.predict_proba(C.reshape(-1, d))[:, 1].reshape(n_perm, d + 1)
        for p in range(n_perm):
            for s, j in enumerate(orders[p]):
                phi[j] += P[p, s + 1] - P[p, s]
        out[i] = phi / n_perm
    return out


print('\n计算相关特征模型的 SHAP ...')
PHI2 = shap_generic(model2, Xi2_arr, bg2, n_perm=60)
ma2 = np.abs(PHI2).mean(0)
j_sgot = FEATS.index('Sgot')
j_copy = len(FEATS)
print('=== 加入 Sgot 的近似复制列后 ===')
print(f'Sgot       mean|SHAP| = {ma2[j_sgot]:.4f}')
print(f'Sgot_copy  mean|SHAP| = {ma2[j_copy]:.4f}')
print(f'两者合计 = {ma2[j_sgot]+ma2[j_copy]:.4f}')
print(f'原模型 Sgot 单独 = {mean_abs[j_sgot]:.4f}')
print(f'相关系数 corr(Sgot, Sgot_copy) = {np.corrcoef(Xi2_arr[:, j_sgot], Xi2_arr[:, j_copy])[0,1]:.4f}')
