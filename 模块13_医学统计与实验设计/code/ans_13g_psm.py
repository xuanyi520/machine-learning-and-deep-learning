# -*- coding: utf-8 -*-
"""13-G 选项 A：以 Antivirals 为暴露做 PSM，报告匹配前后 SMD"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv(r'E:\AI学习\练习代码\data\uci\Hepatitis.csv', na_values=['?'])
df.columns = df.columns.str.strip().str.replace(' ', '_')
df['death'] = (df['Class'] == 1).astype(int)

print('Antivirals 取值分布:', df['Antivirals'].value_counts(dropna=False).to_dict())
print('该变量各组死亡率:')
for v, g in df.groupby('Antivirals'):
    print(f'  取值 {v}: n={len(g)}, 死亡 {int(g.death.sum())} ({g.death.mean():.1%})')

# 该文件里 2 = 使用抗病毒药（2 组死亡率更低，符合临床）
df['exposed'] = df['Antivirals'].map({1: 1, 2: 0})  # 1=使用抗病毒药
covs = ['Age', 'Bilirubin', 'Albumin', 'Protime']
d = df[['exposed', 'death'] + covs].dropna().copy()
print(f'\n完整病例 n={len(d)}，暴露 {int(d.exposed.sum())}，'
      f'对照 {int((1-d.exposed).sum())}，事件 {int(d.death.sum())}')

def smd(a, b):
    """标准化均数差，连续变量用合并标准差。"""
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt((sa**2 + sb**2) / 2)
    return (ma - mb) / sp if sp > 0 else 0.0

print('\n匹配前 SMD:')
before = {}
for c in covs:
    v = smd(d.loc[d.exposed == 1, c], d.loc[d.exposed == 0, c])
    before[c] = v
    print(f'  {c:<12} {v:+.3f}')

# 倾向性评分：暴露 ~ 协变量
ps_model = LogisticRegression(max_iter=2000).fit(d[covs], d['exposed'])
d['ps'] = ps_model.predict_proba(d[covs])[:, 1]
d['logit_ps'] = np.log(d.ps / (1 - d.ps))

tr = d[d.exposed == 1].copy()
co = d[d.exposed == 0].copy()
caliper = 0.2 * d['logit_ps'].std()
print(f'\n卡钳值 0.2*SD(logit PS) = {caliper:.4f}')

nn = NearestNeighbors(n_neighbors=1).fit(co[['logit_ps']].values)
dist, idx = nn.kneighbors(tr[['logit_ps']].values)
tr['pair'] = idx[:, 0]
tr['dist'] = dist[:, 0]
matched_tr = tr[tr['dist'] <= caliper]
matched_co = co.iloc[matched_tr['pair'].values]
print(f'匹配对数 {len(matched_tr)}（暴露组 {len(tr)} 对，'
      f'成功匹配率 {len(matched_tr)/len(tr):.1%}）')

m = pd.concat([matched_tr[covs + ['exposed', 'death']],
               matched_co[covs + ['exposed', 'death']]])
print('\n匹配后 SMD:')
after = {}
for c in covs:
    v = smd(m.loc[m.exposed == 1, c], m.loc[m.exposed == 0, c])
    after[c] = v
    print(f'  {c:<12} {v:+.3f}   (匹配前 {before[c]:+.3f})')

print(f'\n最大 |SMD|：匹配前 {max(abs(v) for v in before.values()):.3f}，'
      f'匹配后 {max(abs(v) for v in after.values()):.3f}')

def or_ci(a, b, c_, dd):
    OR = (a * dd) / (b * c_)
    se = np.sqrt(1/a + 1/b + 1/c_ + 1/dd)
    return OR, np.exp(np.log(OR) - 1.96*se), np.exp(np.log(OR) + 1.96*se)

def tab2x2(data):
    a = int(((data.exposed == 1) & (data.death == 1)).sum())
    b = int(((data.exposed == 1) & (data.death == 0)).sum())
    c_ = int(((data.exposed == 0) & (data.death == 1)).sum())
    dd = int(((data.exposed == 0) & (data.death == 0)).sum())
    return a, b, c_, dd

t_raw = tab2x2(d)
t_mat = tab2x2(m)
print(f'\n粗 2x2: {t_raw}, OR={or_ci(*t_raw)[0]:.3f} '
      f'({or_ci(*t_raw)[1]:.3f}-{or_ci(*t_raw)[2]:.3f})')
print(f'匹配 2x2: {t_mat}, OR={or_ci(*t_mat)[0]:.3f} '
      f'({or_ci(*t_mat)[1]:.3f}-{or_ci(*t_mat)[2]:.3f})')
print(f'暴露组死亡率 粗 {t_raw[0]/(t_raw[0]+t_raw[1]):.1%} -> '
      f'匹配后 {t_mat[0]/(t_mat[0]+t_mat[1]):.1%}')
print(f'对照组死亡率 粗 {t_raw[2]/(t_raw[2]+t_raw[3]):.1%} -> '
      f'匹配后 {t_mat[2]/(t_mat[2]+t_mat[3]):.1%}')
