#import "../../prelude.typ": *

#hd3("Hyena Hierarchy")

#hd4("结合SSM的卷积") #index("Convolution", "SSM Convolution")

以普通的卷积为例，假设 $S in bb(R)^(L times L)$ 为filter，$U  in bb(R)^(L times C)$ 为输入，$y$ 为输出，则有：
$
  y_t = (h * u)_t = sum_(n=0)^(L-1) h_(t-n) u_n
$
这里 $h in bb(R)^(L)$，为了便于SSM的加入，令filter $S$ 为 Toeplitz 矩阵，即每一个对角线上元素相同： #index("Toeplitz Matrix")
$
  S_(i,j) = S_(i+1, j+1)\
  S = mat(
    h_0 , h_(-1) , dots, h_(-L+1) ;
    h_1 , h_0 , dots, h_(-L+2) ;
    dots , dots , dots , dots ;
    h_(L-1) , h_(L-2) , dots , h_0;
  )
$
根据 SSM SSM-soultion，我们可以得到filter:
$
  h_t = cases(
    0 space.quad & t < 0,
    C A^t B + D delta_t space.quad & t >= 0
  )
$

#hd4("FFT Conv")  #index("Convolution", "FFT Convolution")
卷积操作可以利用FFT优化运行速度。卷积操作的空间复杂度为$O(L^2)$，而利用FFT可以将其降低到$O(L log L)$。具体运用了傅里叶变换的卷积性质 FFT-conv.
考虑filter为Toeplitz 矩阵的特殊情况，循环矩阵：
$
  S_h = mat(
    h_0 , h_1 , dots , h_(L-1) ;
    h_(L-1) , h_0 , dots , h_(L-2) ;
    dots , dots , dots , dots ;
    h_1 , h_2 , dots , h_0;
  )
$
利用FFT，我们可以将循环矩阵对角化：
$
  S_h = W^(-1) D_H W
$
其中 $D_H$ 为对角矩阵，$W$ 为DFT矩阵。因此，卷积操作可以写为：
$
  y &= S_h u\
  &= W^(-1) D_H W u\
  &= text("iFFT")(D_h text("FFT")(u))
$
其中，对角矩阵 $D_H$ 的对角元素为循环矩阵的特征值，可以通过以下方式计算：
$
  p(lambda) = det(S_h - lambda I) = 0
$

#hd4("Order-N hyena operator")

假设 $(v, x^1, dots, x^N)_t$ 为输入 $u$ 的投影，同时filters $(h^1, dots, h^N)_t$为可学习的，hyena operator 执行以下循环操作：
$
  z_t^1 &= v\
  z_t^(n+1) &= x_t^n (h^n * z^n)_t space.quad n=1\,dots\,N,\
  y_t &= z_t^(N+1)
$
这一循环操作的卷积由FFT完成，空间复杂度是 $O(N L log L)$.同时注意到，每一步循环操作包括：
1. 对时域进行卷积 ($h^n_t * z^n_t$)，
2. 对频域进行卷积 (时域element-wise product, $x_t^n (h^n * z^n)_t$). 
作者认为：#margin-note("时域上的卷积被认为提高了记忆的长度，而频域上的卷积被认为提高了频率的精细度。")[卷积本质可以理解为对信号的加权和操作，在时域中反映为对历史信息（过往信号）的累积，而在频域中则反映为对信号的频率成分的加权调整] 

#hd4("self-attention operatior")
通常来说，self-attention 只包括3个部分：query, key, value:
$
  y &= text("self-attetnion")(u)\
  &= text("softmax")(1/sqrt(D) u M_q M_k^tack.b u^tack.b) u M_v\
  &= A(q, k) v
$
其中，$M_q, M_k, M_v in bb(R)^(D times D)$ 为输入 $u in bb(R)^(L times D)$ 可学习的投影。

在hyena的attention操作中，我们可以将其拓展为更多的部分。

首先，对于注意力矩阵，使用替代的注意力矩阵 $A(q, k)$，其计算方式为：
$
  A(q,k) = D_q S_(epsilon) D_k S_(phi)
$
其中，$D_q, D_k in bb(R)^(L times L)$ 分别为 $q,k$ 的对角矩阵。$S_epsilon, S_phi$ 为 Toeplitz 矩阵，其参数由 SSM 决定。

因此，3个部分的self-attention操作可以写为：
$
  H_3(q,k,v) = A(q, k) v = D_q S_(epsilon) D_k S_(phi) v
$
拓展到多个部分，我们令 $D_x^n = text("diag")(x^n) in bb(R)^(L times L)$，$S_h^n$ 为 Toeplitz 矩阵，来源于 filter $h^n$，则有：
$
  y = H(u)v = D_x^N S_h^N D_x^(N-1) S_h^(N-1) dots D_x^1 S_h^1 v
$ #index("Self-attention", "Hyena Self-attention")

#hd4("Hyena filter")
Hyena filter采用FFN更新：
$
  h_t = text("Window")(t) dot (text("FFN") circle.tiny text("Positional Encoding"))(t)
$
