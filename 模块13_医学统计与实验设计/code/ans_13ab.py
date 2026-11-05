# -*- coding: utf-8 -*-
"""参考答案脚本：13-A 表 1 与检验选择、13-B p 值/OR/CI"""
import numpy as np
import pandas as pd
from scipy import stats

PATH = r'E:\AI学习\练习代码\data\uci\Hepatitis.csv'
df = pd.read_csv(PATH, na_values=['?'])
df.columns = df.columns.str.strip().str.replace(' ', '_')
df['death'] = (df['Class'] == 1).astype(int)
df['male'] = df['Sex'].map({1: 1, 2: 0})
df['ascites'] = df['Ascites'].map({1: 1, 2: 0})
df['spiders'] = df['Spiders'].map({1: 1, 2: 0})
df['varices'] = df['Varices'].map({1: 1, 2: 0})
df['liver_firm'] = df['Liver_Firm'].map({2: 1, 1: 0})
df['steroid'] = df['Steroid'].map({2: 1, 1: 0})

print('=' * 60)
print('练习 13-A  表 1 与检验选择')
print('=' * 60)

def pick_test(data, var, group):
    sub = data[[var, group]].dropna()
    g1 = sub.loc[sub[group] == 1, var]
    g0 = sub.loc[sub[group] == 0, var]
    if pd.api.types.is_numeric_dtype(sub[var]) and sub[var].nunique() > 2:
        sw1 = stats.shapiro(g1).pvalue if len(g1) >= 3 else np.nan
        sw0 = stats.shapiro(g0).pvalue if len(g0) >= 3 else np.nan
        normal = sw1 > 0.05 and sw0 > 0.05
        if normal:
            p = stats.ttest_ind(g1, g0, equal_var=False).pvalue
            return ('mean ± SD', f'{g1.mean():.2f} ± {g1.std():.2f}',
                    f'{g0.mean():.2f} ± {g0.std():.2f}', p, 'Welch t', len(g1), len(g0))
        p = stats.mannwhitneyu(g1, g0, alternative='two-sided').pvalue
        return ('median (IQR)',
                f'{g1.median():.2f} ({g1.quantile(.25):.2f}-{g1.quantile(.75):.2f})',
                f'{g0.median():.2f} ({g0.quantile(.25):.2f}-{g0.quantile(.75):.2f})',
                p, 'Mann-Whitney U', len(g1), len(g0))
    tab = pd.crosstab(sub[var], sub[group])
    _, p_chi, _, exp = stats.chi2_contingency(tab, correction=False)
    if exp.min() < 5:
        p, test = stats.fisher_exact(tab.values)[1], 'Fisher'
    else:
        p, test = p_chi, 'chi-square'
    a = int(((sub[var] == 1) & (sub[group] == 1)).sum())
    b = int(((sub[var] == 1) & (sub[group] == 0)).sum())
    return ('n (%)', f'{a} ({a/len(g1):.1%})', f'{b} ({b/len(g0):.1%})',
            p, test, len(g1), len(g0), exp.min())

labels = {'Age': '年龄', 'Bilirubin': '胆红素', 'Alk_Phosphate': '碱性磷酸酶',
          'Sgot': 'AST', 'Protime': '凝血酶原时间', 'Albumin': '白蛋白',
          'male': '男性', 'ascites': '腹水', 'spiders': '蜘蛛痣',
          'varices': '静脉曲张', 'liver_firm': '肝质地硬', 'steroid': '使用激素'}

def build_table(data, cont_vars, tag):
    rows = []
    for var in cont_vars:
        out = pick_test(data, var, 'death')
        rows.append([labels[var], out[0], out[1], out[2], out[3], out[4],
                     out[5], out[6]])
    for var in ['male', 'ascites', 'spiders', 'varices', 'liver_firm', 'steroid']:
        out = pick_test(data, var, 'death')
        rows.append([labels[var], out[0], out[1], out[2], out[3], out[4],
                     out[5], out[6]])
    t = pd.DataFrame(rows, columns=['变量', '格式', '死亡组', '存活组', 'p', '检验',
                                    'n死亡组', 'n存活组'])
    t['p'] = t['p'].map(lambda x: '<0.001' if x < 0.001 else f'{x:.3f}')
    print(f'\n--- {tag} ---')
    print(t.to_string(index=False))
    return t

base = ['Age', 'Bilirubin', 'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime']
t0 = build_table(df, base, '原始表 1（连续变量含 AST）')

# 1) Sgot -> Protime：连续变量集合里去掉 Sgot，加入 Protime 已存在，改为对比
print('\n[1] 连续变量的 Shapiro-Wilk p 值与检验方法')
for var in ['Age', 'Bilirubin', 'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime']:
    sub = df[[var, 'death']].dropna()
    g1 = sub.loc[sub.death == 1, var]
    g0 = sub.loc[sub.death == 0, var]
    out = pick_test(df, var, 'death')
    print(f'  {labels[var]:<8} n(死亡/存活)={len(g1)}/{len(g0)}  '
          f'Shapiro p 死亡={stats.shapiro(g1).pvalue:.4g} '
          f'存活={stats.shapiro(g0).pvalue:.4g}  '
          f'偏度 死亡={stats.skew(g1):.2f} 存活={stats.skew(g0):.2f}  -> {out[4]}')

print('\n[1b] 把 Protime 加入、Sgot 移出后的表 1')
t1 = build_table(df, ['Age', 'Bilirubin', 'Alk_Phosphate', 'Protime', 'Albumin'],
                 'Protime 版')

# 2) 10 个死亡病人 Albumin 置缺失
rng = np.random.default_rng(0)
df2 = df.copy()
dead_idx = df2.index[df2.death == 1].to_numpy()
drop_idx = rng.choice(dead_idx, 10, replace=False)
df2.loc[drop_idx, 'Albumin'] = np.nan
print(f'\n[2] 人为置缺失的病人索引: {sorted(drop_idx)}')

out_before = pick_test(df, 'Albumin', 'death')
out_after = pick_test(df2, 'Albumin', 'death')
print(f'  缺失前: n={out_before[5]}/{out_before[6]}, '
      f'中位数 {out_before[1]} vs {out_before[2]}, p={out_before[3]:.4g}')
print(f'  缺失后: n={out_after[5]}/{out_after[6]}, '
      f'中位数 {out_after[1]} vs {out_after[2]}, p={out_after[3]:.4g}')
print(f'  p 变化倍数: {out_after[3]/out_before[3]:.1f} 倍')

# 被删掉的 10 个人长什么样
print('  被删 10 人的白蛋白:', np.round(df.loc[drop_idx, 'Albumin'].values, 2))
print('  剩余死亡组白蛋白中位数:', out_after[1])
# 敏感性：用同样的 10 人，但换成随机抽存活组
rng2 = np.random.default_rng(0)
alive_idx = df2.index[df2.death == 0].to_numpy()
df3 = df.copy()
drop_alive = rng2.choice(alive_idx, 10, replace=False)
df3.loc[drop_alive, 'Albumin'] = np.nan
out_alive = pick_test(df3, 'Albumin', 'death')
print(f'  对照：改抽 10 个存活组置缺失 -> n={out_alive[5]}/{out_alive[6]}, '
      f'p={out_alive[3]:.4g}')

print()
print('=' * 60)
print('练习 13-B  p 值、OR 和置信区间')
print('=' * 60)

a, b, c, d = 20, 60, 10, 90
OR = (a * d) / (b * c)
lnOR = np.log(OR)
SE = np.sqrt(1/a + 1/b + 1/c + 1/d)
lo, hi = np.exp(lnOR - 1.96 * SE), np.exp(lnOR + 1.96 * SE)
print(f'OR = ({a}×{d})/({b}×{c}) = {a*d}/{b*c} = {OR:.4f}')
print(f'ln(OR) = {lnOR:.6f}')
print(f'SE(lnOR) = sqrt(1/{a}+1/{b}+1/{c}+1/{d}) = sqrt({1/a + 1/b + 1/c + 1/d:.6f}) = {SE:.6f}')
print(f'95% CI = exp({lnOR:.4f} ± 1.96×{SE:.4f}) = {lo:.4f} 到 {hi:.4f}')

r1, r0 = a / (a + b), c / (c + d)
RR = r1 / r0
lnRR = np.log(RR)
SE_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
print(f'\n暴露组风险 = {a}/{a+b} = {r1:.4f}')
print(f'对照组风险 = {c}/{c+d} = {r0:.4f}')
print(f'RR = {RR:.4f}，ln(RR) = {lnRR:.4f}')
print(f'SE(lnRR) = {SE_rr:.4f}，95% CI = '
      f'{np.exp(lnRR-1.96*SE_rr):.4f}-{np.exp(lnRR+1.96*SE_rr):.4f}')
print(f'OR 比 RR 大 {(OR/RR-1)*100:.1f}%')
print(f'OR - RR = {OR-RR:.4f}')

tab = np.array([[a, b], [c, d]])
chi2, p_chi, _, exp = stats.chi2_contingency(tab, correction=False)
chi2c, p_chic, _, _ = stats.chi2_contingency(tab, correction=True)
print(f'\n卡方（不校正）chi2={chi2:.4f}, p={p_chi:.4f}')
print(f'卡方（Yates 校正）chi2={chi2c:.4f}, p={p_chic:.4f}')
print(f'Fisher p={stats.fisher_exact(tab)[1]:.4f}，最小期望频数={exp.min():.2f}')

# 用 13-2 的 fit_logit 验证
def fit_logit(X, y, names):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    for _ in range(50):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-12
        grad = X1.T @ (y - p)
        info = X1.T @ (X1 * W[:, None])
        step = np.linalg.solve(info, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X1 @ beta)))
    W = p * (1 - p)
    cov = np.linalg.inv(X1.T @ (X1 * W[:, None]))
    se = np.sqrt(np.diag(cov))
    out = []
    for i, name in enumerate(names, start=1):
        z = beta[i] / se[i]
        out.append([name, np.exp(beta[i]),
                    np.exp(beta[i] - 1.96 * se[i]),
                    np.exp(beta[i] + 1.96 * se[i]),
                    2 * stats.norm.sf(abs(z))])
    return pd.DataFrame(out, columns=['变量', 'OR', '95% CI 下限', '95% CI 上限', 'p'])

sim = pd.DataFrame({'exposed': np.r_[np.ones(a + b), np.zeros(c + d)],
                    'death': np.r_[np.ones(a), np.zeros(b),
                                   np.ones(c), np.zeros(d)]})
res = fit_logit(sim[['exposed']], sim['death'], ['exposed'])
print('\nfit_logit 结果')
print(res.to_string(index=False, formatters={'OR': '{:.4f}'.format,
      '95% CI 下限': '{:.4f}'.format, '95% CI 上限': '{:.4f}'.format,
      'p': '{:.6g}'.format}))
print(f'与手算 OR 差 {abs(res.loc[0,"OR"]-OR):.6f}，'
      f'与手算 CI 下限差 {abs(res.loc[0,"95% CI 下限"]-lo):.6f}，'
      f'上限差 {abs(res.loc[0,"95% CI 上限"]-hi):.6f}')
