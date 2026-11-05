# -*- coding: utf-8 -*-
"""最小贝叶斯优化：高斯过程 + Expected Improvement
   与网格/随机在同预算 32 次下比较"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm


def f(x):
    return 0.90 - (x[..., 0] - 0.65) ** 2 - (x[..., 1] - 0.35) ** 2


def expected_improvement(gp, X_cand, best_y):
    mu, std = gp.predict(X_cand, return_std=True)
    std = np.maximum(std, 1e-9)
    z = (mu - best_y) / std
    return (mu - best_y) * norm.cdf(z) + std * norm.pdf(z)


rng = np.random.default_rng(0)
X_obs = rng.random((8, 5))
y_obs = f(X_obs)
trace = [y_obs.max()]

for step in range(24):                     # 总预算 32
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(5))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  random_state=0)
    gp.fit(X_obs, y_obs)
    X_cand = rng.random((2000, 5))
    ei = expected_improvement(gp, X_cand, y_obs.max())
    x_new = X_cand[np.argmax(ei)]
    y_new = f(x_new)
    X_obs = np.vstack([X_obs, x_new])
    y_obs = np.append(y_obs, y_new)
    trace.append(y_obs.max())

print('贝叶斯优化 32 次后的最优值:', round(y_obs.max(), 4))
print('每步累计最优:', np.round(trace, 3))

# 同预算随机搜索，重复 2000 次
rand_best = np.array([f(rng.random((32, 5))).max() for _ in range(2000)])
print(f'随机 32 次: 均值 {rand_best.mean():.4f}, 中位数 {np.median(rand_best):.4f}')

levels = np.array([0.0, 1.0])
grid_pts = np.array(np.meshgrid(*([levels] * 5))).T.reshape(-1, 5)
print('网格 32 点最优:', round(f(grid_pts).max(), 4))
