# -*- coding: utf-8 -*-
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

rng = np.random.default_rng(0)

# 真实分布（刚实测的）
val_n, val_pos = 524, 389
test_n, test_pos = 624, 390


def make_labels(n, pos):
    return np.array([1] * pos + [0] * (n - pos))


y_val = make_labels(val_n, val_pos)
y_test = make_labels(test_n, test_pos)

M = 200
acc_val = np.zeros(M)
acc_test = np.zeros(M)

for i in range(M):
    pv = rng.integers(0, 2, val_n)
    pt = rng.integers(0, 2, test_n)
    acc_val[i] = (pv == y_val).mean()
    acc_test[i] = (pt == y_test).mean()

print("200 个「什么都不懂、纯瞎猜」的模型\n")
print(f"验证集准确率 : 最低 {acc_val.min():.1%}  最高 {acc_val.max():.1%}  平均 {acc_val.mean():.1%}")
print(f"测试集准确率 : 最低 {acc_test.min():.1%}  最高 {acc_test.max():.1%}  平均 {acc_test.mean():.1%}")

print("\n【做法 A】直接在测试集上挑最好的那个")
best_test = int(np.argmax(acc_test))
print(f"  挑中的模型测试成绩 = {acc_test[best_test]:.1%}   <- 报出去的数字")

print("\n【做法 B】在验证集上挑，再去测试集报分")
best_val = int(np.argmax(acc_val))
print(f"  在验证集上挑中第 {best_val} 号，验证成绩 {acc_val[best_val]:.1%}")
print(f"  同一个模型在测试集上 = {acc_test[best_val]:.1%}   <- 报出去的数字")
