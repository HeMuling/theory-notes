#import "../prelude.typ": *

#hd2("基础知识")

#hd3("NFL定理")#index([NFL Theorem])

归纳偏好用于描述当特征相同时，哪些特征更为重要

假设样本空间 $cal(X)$ 和假设空间 $cal(H)$ 为离散。令 $P(h|X, xi_a)$ 代表算法 $xi_a$ 基于训练数据 $X$ 产生假设 $h$ 的概率；令 $f$ 代表希望学习的目标函数。因此，算法在训练集外产生的误差为：

$
E_(o t e) (xi_a|X,f) = sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)
$



其中 $bb(I)(dot)$ 为指示函数，当 $dot$ 为真时返回 1，否则返回 0。

若学习目标为二分类，则 $cal(X) arrow.bar {0,1}$ 且函数空间为 ${0, 1}^(|cal(X)|)$，其中 $|dot|$ 用于计算集合长度。

算法用于解决多个任务，则拥有多个学习的目标函数；假设这些目标函数均匀分布，则这些目标函数的误差总和为：

$
sum_f E_(o t e) (xi_a|X,f) &= sum_f sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)\
&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) sum_f bb(I)(h(bold(x)) eq.not f(bold(x)))\

&text(font: "STFangsong", "根据假设，总有一半是正确的，因此")\

&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) 1/2 2^(|cal(X|))\
&= 2^(|cal(X)|-1) sum_(x in cal(X)-X)P(X)
$

#h(2em) 因此可知，在多目标目标函数均匀分布的情况下，不同算法所得的误差总和相同。实际情况中，某一算法通常只用于解决单一问题，且其目标函数的分布不均匀（即目标函数重要性不同），因此不同算法所得的误差总和不同。

这告诉我们，在某一任务上表现好的算法在另一任务上表现不一定好。

#pagebreak()
#hd3("Monte-Carlo estimation") #index([Monte-Carlo estimation])

Monte-Carlo estimation可以用于估计复杂积分，假设 $f: bb(R)^d arrow bb(R)$，以及一个定义在 $cal(D) in bb(R)^d$ 上的pdf $p: bb(R)^d arrow bb(R)$，期望计算积分：
$
  I = integral_D f(bold(x)) d bold(x)
$
对于上式，假设 $bold(x) tilde p(bold(x))$，则可以变形为：
$
  I = integral_D p(bold(x)) f(bold(x))/p(bold(x)) d bold(x) = bb(E)_p [ f(bold(x))/p(bold(x))]
$
因此可以设计 Monte-Carlo estimator 为：
$
  hat(I)_N = 1/N sum_(i=1)^N f(bold(x_i))/p(bold(x_i))
$
且具有无偏性
- 无偏性：
$
  bb(E)[hat(I)_N] = I
$
- 方差：
$
  "Var"(hat(I)) = 1/N (bb(E)[(f(bold(x))/p(bold(x)))^2] - I^2)
$
特别的，当从均匀分布中采样时，$p(bold(x)) = 1/V$，其中 $V$ 为 $cal(D)$ 的体积，则：
$
  hat(I)_N = V/N sum_(i=1)^N f(bold(x_i))
$
当积分代表期望时，可以使用Monte-Carlo estimation：
$
  I &= integral_D p(bold(x)) f(bold(x)) d bold(x)\
  &= 1/N sum_(i=1)^N f(bold(x_i)), space.quad bold(x_i) tilde p(bold(x))
$


#pagebreak()
#hd3("CNN")

#hd4("Convolution")
卷积通常是指：
$
Y_(i,j) = (X * W)_(i,j) = sum_(m=0)^(M-1) sum_(n=0)^(N-1) X_(i+m, j+n) W_(m,n)
$
卷积操作包括Conv1d和Conv2d#index("Convolution","Convolution layer")

Conv1d: 通常对于一维特征图，给定输入 $X in bb(R)^(N times L times C_text("in"))$，filter $W in bb(R)^(K times C_text("in") times C_text("out"))$，卷积操作为：
$
  Y_(n, C_text("out"),l) = sum_(c_text("in")=0)^(C_text("in")-1) sum_(k=0)^(K-1) X_(n,l+k, c_text("in")) W_(k, c_text("in"), c_text("out"))
$

Conv2d: 通常对于图像或二维特征图，给定输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")) times C_(text("out")))$，其中 $C_(text("in"))$ 为输入通道数，$C_(text("out"))$ 为输出通道数，$K_H, K_W$ 为卷积核的高度和宽度，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c_text("in")=0)^(C_text("in") - 1) sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c_text("in")) W_(m, n, c_text("in"), c_text("out"))
$
其中，$Y in bb(R)^(N times H_text("out") times W_text("out") times C_text("out"))$，$h, w$ 为空间位置索引，$o$ 为输出通道索引

池化层类型包括：最大池化、平均池化。#index("Convolution","Pooling Layer")
$
  A_(i, j) = max_(m,n) (a, Y_i,j)\
  A_(i, j) = 1/(M*N) sum_(m=0)^(M-1) sum_(n=0)^(N-1) Y_(i+m, j+n)
$

```python
import torch.nn as nn
conv = nn.Conv1d(in_channels, out_channels, kernel_size)
conv = nn.Conv2d(in_channels, out_channels, kernel_size)
pool = nn.MaxPool2d(kernel_size, stride)
pool = nn.AvgPool2d(kernel_size, stride)
```

#hd4("Depthwise Separable Convolution") #index("Convolution","Depthwise Separable Convolution")

Depthwise Separable Convolution @howard2017mobilenets 是一种轻量级卷积操作，其包括两个步骤：Depthwise Convolution 和 Pointwise Convolution.

Depthwise Convolution 对每一个通道单独使用卷积核 filter.对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")))$，卷积操作为：
$
  Y_(n, h, w, c) = sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c) W_(m, n, c)
$
Pointwise Convolution 使用 $1 times 1$ 的卷积核，对每一个通道进行卷积操作。对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(1 times 1 times C_(text("in")) times C_(text("out")))$，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c=0)^(C_(text("in")) - 1) X_(n, h, w, c) W_(0, 0, c, o)
$
Depthwise Separable Convolution 结合以上两种卷积方式，首先使用 Depthwise Convolution ，然后使用 Pointwise Convolution. 

```py
import torch  
import torch.nn as nn  

depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                      groups=in_channels)
pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
deepwise_separable = pointwise(depthwise(input))
```

#hd3("State-Space model") #index("State-Space model")

State-Space model (SSM) 是用于描述时间序列数据的模型。对于任意时间序列输入 $u(t)$，SSM首先将其映射到 hidden space $x(t)$，然后进一步映射为输出空间 $y(t)$：
$
  u(t) arrow.bar x(t) arrow.bar y(t)
$
SSM以以下形式表示：
$
  x'(t) = A x(t) + B u(t)\
  y(t) = C x(t) + D u(t)
$
解为：
$
  y(t) = sum_(n=0)^t (C A^(t - n) B + D delta(t-n))u(n)
$ <SSM-soultion>
其中，$delta(t-n)$ 为Kronecker delta函数。

#hd3("Fourier Transform") #index("Fourier Transform")

#hd4("思想")

傅里叶变换的基本思想是：利用无穷个不同频率周期函数的线性组合来表示一个非周期函数。即：
$
  f(t) = sum_i a_i f_i (t)
$

最简单的周期函数为圆周运动，根据欧拉公式，我们可以得到：
$
  e^(i omega t) = cos(omega t) + i sin(omega t)
$
其中 $omega$ 表示旋转速度，正数时为逆时针旋转，负数时为顺时针旋转。圆周运动的频率为 $T = (2 pi)\/omega$. 同时注意到：旋转整数倍周期后回到原点，即
$
  integral_0^(n T) e^(i omega t) d t = 0
$

为了计算方便，我们可以令所有的周期都是 $2 pi \/omega$ 的整数倍，即：
$
  f(t) = sum_(-infinity)^(+infinity) c_k e^(i k omega_0 t)
$

这样一来，我们设定的是正交基：
$
  mat(dots, e^(-2i omega_0 t), e^(-i omega_0 t), 1, e^(i omega_0 t), e^(2i omega_0 t), dots)
$

两边同乘 $e^(i -n omega_0 t)$ 并积分：
$
  integral_0^T f(t) e^(-i n omega_0 t) d t &= sum_(-infinity)^(+infinity) c_k integral_0^T e^(i (k-n) omega_0 t) d t\
  &=T c_n
$
因此：
$
  c_n = 1/T integral_0^T f(t) e^(-i n omega_0 t) d t
$

这里的 $c_n$ 为不同角频率的圆周运动的系数

#hd4("傅里叶变换")

对于一个连续信号 $f: bb(R)^d arrow bb(C)$，其连续傅里叶变换 (CFT) 为 $cal(F): bb(R)^d arrow bb(C)$：
$
  cal(F)(f)(bold(K)) = integral_(bb(R)^d) f(bold(x)) e^(-2 pi i bold(K) dot bold(X)) d bold(X)
$
同时，可以进行逆变换：
$
  cal(F)^(-1)(f)(bold(X)) = integral_(bb(R)^d) f(bold(K)) e^(2 pi i bold(K) dot bold(X)) d bold(K)
$
对于不连续点序列 ${x[n]}_(0 <= n <= N)$，其离散傅里叶变换 (DFT) 为：
$
  cal(F) x[n] = sum_(n=0)^(N-1) x[n] e^(-2 pi i n k \/ N), space.quad k = 0, 1, dots, N-1
$
同理，可以进行逆变换：
$
  cal(F)^(-1) x[k] = 1/N sum_(k=0)^(N-1) x[k] e^(2 pi i n k \/ N), space.quad n = 0, 1, dots, N-1
$
同时，可以通过矩阵乘法表示：
$
  cal(F) x = W dot x
$
其中 $W$ 为 DFT 矩阵，${W_(i,j) = e^(-2 pi i n k \/ N)}_(n,j = 0, dots, N-1)$:
$
  W = 1/sqrt(N) mat(
    1 , 1 , 1 , dots , 1;
    1 , e^(-2 pi i 1 \/ N) , e^(-2 pi i 2 \/ N) , dots , e^(-2 pi i (N-1) \/ N);
    1 , e^(-2 pi i 2 \/ N) , e^(-2 pi i 4 \/ N) , dots , e^(-2 pi i 2(N-1) \/ N);
    dots , dots , dots , dots , dots;
    1 , e^(-2 pi i (N-1) \/ N) , e^(-2 pi i 2(N-1) \/ N) , dots , e^(-2 pi i (N-1)^2 \/ N)
  )
$
其中 $1\/sqrt(N)$ 为归一化系数，使得 $W$ 具有以下性质：
$
  W^(-1) dot W = I
$

#hd4("卷积和卷积定理") #index("Fourier Transform","Convolution Theorem")
时域上的卷积等价于频域上的乘积，若 $cal(F)[f(t)] = f(omega), cal(F)[g(t)] = g(omega)$，则：
$
  cal(F)[f(t) * g(t)] = F(omega) G(omega)\
  cal(F)[f(t) g(t)] = 1/(2 pi) F(omega) * G(omega)
$ <FFT-conv>
注意：这里的乘法不是矩阵的点乘，而是element-wise的乘法。即对于函数来说，每一点值的乘积作为新的函数值。

#pagebreak()
