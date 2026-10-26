# 3-1 Tensor 与自动求导

**建议用时：6-8 小时**

PyTorch 的核心就两件事：**Tensor**（能跑在 GPU 上的数组）和 **autograd**（自动算梯度）。

第 3 节学完，你会发现 W6 你手推的那些梯度公式，PyTorch 一行就给你了。

***

## 一、Tensor 就是 numpy 数组的升级版

```python
import torch
import numpy as np

# 创建
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(2, 3)
c = torch.randn(2, 3)          # 标准正态随机
d = torch.arange(0, 10, 2)     # [0,2,4,6,8]

print(a.shape, b.shape, c.shape, d)

# 和 numpy 互转
n = np.array([[1., 2.], [3., 4.]])
t = torch.from_numpy(n)        # numpy -> tensor（共享内存）
back = t.numpy()               # tensor -> numpy
```

**运算几乎和 numpy 一样：**

```python
x = torch.randn(3, 4)
y = torch.randn(4, 5)

print((x @ y).shape)           # (3, 5) 矩阵乘法
print((x + 1).shape)           # 广播
print(x.sum(), x.mean())       # 聚合
print(x.T.shape)               # 转置
```

**那为什么还要用 Tensor？** 两个理由：

1. **能放到 GPU 上算**
2. **能自动求导**（本节重点）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 4).to(device)
print(x.device)
```

***

## 二、自动求导（autograd）—— PyTorch 的灵魂

回到 W6：你手推了 `∂loss/∂w = (2/N) Σ eᵢxᵢ`。

如果是个 50 层的神经网络，你手推试试？不可能。

**autograd 让你只写前向计算，梯度它帮你算。**

```python
import torch

# requires_grad=True 表示"这个变量需要求导"
x = torch.tensor(3.0, requires_grad=True)

# 前向：随便写个函数
y = x**2 + 3*x + 1

# 反向：一行搞定
y.backward()

# 取梯度
print(x.grad)     # 应该等于 2*3 + 3 = 9
```

**对照 W3 的手算：`f'(x) = 2x + 3`，在 x=3 处就是 9。** 完全一致。

你只需要做两件事：**写前向计算**，然后 **`.backward()`**。

***

## 三、为什么它叫"自动"

因为 PyTorch 记录了你的每一步运算，构建了一张"计算图"，然后沿着图反向应用链式法则。

```python
x = torch.tensor(2.0, requires_grad=True)

a = x * 3          # a = 3x
b = a + 1          # b = 3x + 1
c = b ** 2         # c = (3x+1)²

c.backward()

print(x.grad)
# 手动验证：dc/dx = 2(3x+1) * 3 = 2*7*3 = 42
print(2 * (3*2+1) * 3)
```

两个 42。**这就是 W3 的链式法则，PyTorch 自动做了。**

***

## 四、几个必须知道的坑

### 坑 1：梯度会累加

```python
x = torch.tensor(2.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print(x.grad)      # 4.0

y2 = x ** 3
y2.backward()
print(x.grad)      # 12.0，不是 12！是 4 + 12 = 16！
```

**PyTorch 的梯度是累加的，不是覆盖。** 所以训练循环里必须每步清零：

```python
optimizer.zero_grad()      # 或者 x.grad.zero_()
```

**这是新手最经典的 bug**：忘了 zero_grad，梯度越加越大，模型直接飞走。

### 坑 2：只有叶子节点保留梯度

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y ** 2
z.backward()

print(x.grad)      # 有值
print(y.grad)      # 是 None（中间变量默认不留）
```

想看中间变量的梯度，要 `.retain_grad()`。

### 坑 3：不需要梯度时要关掉

推理（预测）的时候不需要求导，关掉能省内存加速：

```python
with torch.no_grad():
    pred = model(x)
```

***

## 五、跟着做：用 autograd 重写 W6 的线性回归

**这是本节的核心练习。** 回到 W6 你手推梯度的那个模型，这次让 PyTorch 算梯度。

```python
import torch
import numpy as np

torch.manual_seed(42)

# 造数据
N = 200
x = torch.rand(N) * 10
y = 2.5 * x + 1.0 + torch.randn(N) * 1.5

# 参数：要学的东西，requires_grad=True
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01

for epoch in range(1000):
    # 前向
    y_pred = w * x + b

    # 损失
    loss = ((y_pred - y) ** 2).mean()

    # 反向 —— 就这一行，梯度自动算好了
    loss.backward()

    # 更新参数 —— 必须用 no_grad，否则会记进计算图
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度（坑 1）
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.4f}  w={w.item():.4f}  b={b.item():.4f}")

print(f"\n最终 w={w.item():.4f}  b={b.item():.4f}")
```

**跑通后对比 W6 的代码。** 你会发现：

- W6 你手写了 `dw = (2/N) * np.sum(error * x)`
- 现在你只写了 `loss.backward()`

**结果完全一样。** 这就是 PyTorch 的价值。

***

## 六、练习

### 练习 3-1-A（基础）

用 autograd 求 `f(x) = x³ - 2x² + 5` 在 x = 2 处的导数。

先手算（`f'(x) = 3x² - 4x`，代入得 4），再用代码验证。

### 练习 3-1-B（多变量）

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2 * y + y**3
```

求 z 对 x 和 y 的偏导数。先手推，再用代码验证。

（手推提示：`∂z/∂x = 2xy`，`∂z/∂y = x² + 3y²`）

### 练习 3-1-C（踩坑练习，重要）

下面这段代码有什么问题？会导致什么后果？

```python
w = torch.zeros(1, requires_grad=True)
for epoch in range(100):
    loss = (w * 2 - 4) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
```

**写出来，跑一遍，观察 w 变成什么样。** 然后修好它。

### 练习 3-1-D（numpy 与 torch 对照）

把你 W6 写的 numpy 线性回归，逐行改写成 PyTorch 版本，保持结构一致。

目的：**建立"这两个是同一件事"的肌肉记忆。**

### 练习 3-1-E（挑战）

用 autograd 求 `f(x) = sin(x) * exp(x)` 在 x = 1 处的导数，和手算对比。

（手算提示：`f'(x) = cos(x)eˣ + sin(x)eˣ = eˣ(cos x + sin x)`）

***

## 七、自测题

1. Tensor 和 numpy 数组的区别是什么？
Tensor 能放到 GPU、能自动求导；numpy 数组不能
2. `requires_grad=True` 做什么？
标记这个变量需要计算梯度，PyTorch 会在反向传播时填充 .grad
3. 为什么训练循环里要 `zero_grad()`？不写会怎样？
因为梯度默认累加而不是覆盖。不清零会导致梯度越加越大，训练发散
4. autograd 和你在 W3 学的链式法则是什么关系？
autograd 就是自动执行链式法则。 你在 W3 手推的每步求导+相乘，它都做了，只是自动化的
5. 推理时为什么用 `torch.no_grad()`？
节省显存、加快速度。因为推理不需要计算图，PyTorch 可以省掉记录中间结果的开销

***

## 出口标准

- [x] 能用 autograd 求出简单函数的梯度，且和手算对上
- [x] 能解释"梯度累加"这个坑，并知道怎么避免
- [x] 能用 PyTorch 重写一遍线性回归
- [x] 能说清楚 autograd 帮你做了什么

做到了进 3-2。**下一节开始搭真正的神经网络。**
