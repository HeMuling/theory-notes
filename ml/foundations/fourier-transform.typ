#import "../../prelude.typ": *

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
