# -*- coding: utf-8 -*-
"""参考答案脚本：13-C 多重检验校正（15 个指标）、13-D 生存分析（合成随访）"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PATH = r'E:\AI学习\练习代码\data\uci\Hepatitis.csv'
df = pd.read_csv(PATH, na_values=['?'])
df.columns = df.columns.str.strip().str.replace(' ', '_')
df['death'] = (df['Class'] == 1).astype(int)

print('=' * 60)
print('练习 13-C  15 个指标的 BH 与 Bonferroni')
print('=' * 60)

def p_value(var):
    sub = df[[var, 'death']].dropna()
    if sub[var].nunique() > 2:
        a = sub.loc[sub['death'] == 1, var]
        b = sub.loc[sub['death'] == 0, var]
        return stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
    tab = pd.crosstab(sub[var], sub['death']).values
    return stats.fisher_exact(tab)[1]

base10 = ['Age', 'Spleen_Palpable', 'Alk_Phosphate', 'Sex', 'Steroid',
          'Anorexia', 'Antivirals', 'Sgot', 'Liver_Firm', 'Liver_Big']
extra5 = ['Bilirubin', 'Albumin', 'Protime', 'Ascites', 'Varices']

def bh(raw):
    m = len(raw)
    order = np.argsort(raw)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, raw[idx] * m / rank)
        q[idx] = prev
    return q

def report(names, tag):
    raw = np.array([p_value(v) for v in names])
    q = bh(raw)
    bonf = np.minimum(raw * len(raw), 1.0)
    out = pd.DataFrame({'指标': names, '原始 p': raw, 'BH q': q,
                        'Bonferroni p': bonf})
    out = out.sort_values('原始 p').reset_index(drop=True)
    out['原始显著'] = np.where(out['原始 p'] < 0.05, '是', '否')
    out['BH 后显著'] = np.where(out['BH q'] < 0.05, '是', '否')
    out['Bonf 后显著'] = np.where(out['Bonferroni p'] < 0.05, '是', '否')
    print(f'\n--- {tag}（m={len(names)}）---')
    print(out.to_string(index=False, formatters={
        '原始 p': '{:.4g}'.format, 'BH q': '{:.4g}'.format,
        'Bonferroni p': '{:.4g}'.format}))
    print(f'原始 p<0.05: {(raw<0.05).sum()} 个 | '
          f'BH q<0.05: {(q<0.05).sum()} 个 | '
          f'Bonferroni: {(bonf<0.05).sum()} 个 | '
          f'阈值 0.05/{len(names)}={0.05/len(names):.4f}')
    return out

out10 = report(base10, '原来的 10 个指标')
out15 = report(base10 + extra5, '加入 5 个强信号后的 15 个指标')

# 边缘显著的指标在两种 family 下的 q 值对照
print('\n边缘指标的前后对照')
for v in ['Alk_Phosphate', 'Sex', 'Steroid']:
    r10 = out10[out10.指标 == v].iloc[0]
    r15 = out15[out15.指标 == v].iloc[0]
    print(f'  {v:<14} 10 指标: p={r10["原始 p"]:.4g}, q={r10["BH q"]:.4g} | '
          f'15 指标: p={r15["原始 p"]:.4g}, q={r15["BH q"]:.4g}')

# 用 scipy 校验
try:
    from scipy.stats import false_discovery_control
    raw15 = out15.set_index('指标').loc[base10 + extra5, '原始 p'].values
    q15 = out15.set_index('指标').loc[base10 + extra5, 'BH q'].values
    print('scipy BH 校验:', np.allclose(q15, false_discovery_control(raw15)))
except Exception as e:
    print('校验失败:', e)

print()
print('=' * 60)
print('练习 13-D  生存分析（随访时间为合成，仅演示方法）')
print('=' * 60)

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

for c in ['Age', 'Bilirubin', 'Albumin']:
    df[c] = df[c].fillna(df[c].median())

n = len(df)
high_bili = (df['Bilirubin'] >= 1.2).astype(int)
lp = (0.040 * (df['Age'] - 42) + 0.70 * high_bili - 1.00 * (df['Albumin'] - 3.8))
rate = 0.018 * np.exp(lp)
rng = np.random.default_rng(22)
event_time = -np.log(rng.uniform(size=n)) / rate
censor_time = rng.uniform(0, 100, n)
T = np.minimum(event_time, censor_time)
E = (event_time <= censor_time).astype(int)
surv = pd.DataFrame({'T': T, 'E': E, 'Age': df['Age'],
                     'HighBili': high_bili, 'Albumin': df['Albumin']})
print(f'【合成随访】样本 {n}，事件 {E.sum()}，删失 {n-E.sum()}')
print(f'随访时间中位数 {np.median(T):.1f} 月，'
      f'删失率 {1-E.mean():.1%}')

# KM + number at risk
fig, ax = plt.subplots(figsize=(8, 5.5))
kmf = KaplanMeierFitter()
for g, label in [(0, '胆红素正常'), (1, '胆红素升高')]:
    kmf.fit(T[high_bili == g], E[high_bili == g], label=label)
    kmf.plot_survival_function(ci_show=True, ax=ax)
    print(f'{label}: n={int((high_bili==g).sum())}, '
          f'中位生存={kmf.median_survival_time_:.1f} 月, '
          f'S(12)={kmf.predict(12):.3f}, S(24)={kmf.predict(24):.3f}')

times = np.array([0, 12, 24, 36, 48])
print('\nNumber at risk')
for g, label in [(0, '胆红素正常'), (1, '胆红素升高')]:
    at_risk = [int((T[high_bili == g] >= t).sum()) for t in times]
    print(f'  {label}: {at_risk}')
    ypos = 0.08 if g == 1 else 0.02
    for t, r in zip(times, at_risk):
        ax.text(t, -ypos, str(r), ha='center', fontsize=9)
ax.text(-3, -0.02, 'Number at risk', fontsize=9, ha='right')
ax.set_ylim(-0.12, 1.02)
ax.set_xlim(-3, 95)
ax.set_xlabel('随访时间（月）')
ax.set_ylabel('生存概率')
ax.set_title('Kaplan-Meier 曲线（合成随访数据，仅演示方法）')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\AI学习\模块13_医学统计与实验设计\fig_ans_km.png', dpi=150)
plt.close()

lr = logrank_test(T[high_bili == 1], T[high_bili == 0],
                  E[high_bili == 1], E[high_bili == 0])
print(f'\nlog-rank chi2={lr.test_statistic:.2f}, p={lr.p_value:.3g}')

cph = CoxPHFitter()
cph.fit(surv, duration_col='T', event_col='E')
print(f'\n多因素 Cox（n={len(surv)}, 事件={int(E.sum())}）')
for c in ['Age', 'HighBili', 'Albumin']:
    hr = np.exp(cph.params_[c])
    lo, hi = np.exp(cph.confidence_intervals_.loc[c].values)
    print(f'{c:9s} HR={hr:.3f} (95% CI {lo:.3f}-{hi:.3f}), '
          f'p={cph.summary.loc[c,"p"]:.3g}')
print(f'C-index={cph.concordance_index_:.3f}, '
      f'LR chi2={cph.log_likelihood_ratio_test().test_statistic:.2f}')
print('Age 每 10 岁 HR =', round(np.exp(10*cph.params_['Age']), 3))

pht = proportional_hazard_test(cph, surv, time_transform='rank')
print('\n比例风险检验（Schoenfeld 残差，rank 变换）')
print(pht.summary[['test_statistic', 'p']].to_string())

# 分段 Cox：前 12 个月 / 12 个月后
cut = 12
early = surv.copy()
early['E2'] = np.where((early['T'] <= cut) & (early['E'] == 1), 1, 0)
early['T2'] = np.minimum(early['T'], cut)
c1 = CoxPHFitter().fit(early[['T2', 'E2', 'Age', 'HighBili', 'Albumin']]
                       .rename(columns={'T2': 'T', 'E2': 'E'}),
                       duration_col='T', event_col='E')
print(f'\n前 12 个月（事件 {int(early.E2.sum())}）')
for c in ['Age', 'HighBili', 'Albumin']:
    print(f'  {c:9s} HR={np.exp(c1.params_[c]):.3f}, p={c1.summary.loc[c,"p"]:.3g}')

late = surv[surv['T'] > cut].copy()
late['T2'] = late['T'] - cut
c2 = CoxPHFitter().fit(late[['T2', 'E', 'Age', 'HighBili', 'Albumin']]
                       .rename(columns={'T2': 'T'}),
                       duration_col='T', event_col='E')
print(f'12 个月后（n={len(late)}, 事件 {int(late.E.sum())}）')
for c in ['Age', 'HighBili', 'Albumin']:
    print(f'  {c:9s} HR={np.exp(c2.params_[c]):.3f}, p={c2.summary.loc[c,"p"]:.3g}')
