#import "../../prelude.typ": *

#hd3("Conjugate gradient algorithm")  #index("Conjugate Gradient Algorithm")

#text(blue)[
  link: #link("https://en.wikipedia.org/wiki/Conjugate_gradient_method")[Conjugate gradient method-Wikipedia]
]

from: Rasmussen, C. (2006). Conjugate gradient algorithm, version 2006-09-08. available online.

共轭梯度算法（Conjugate Gradient Algorithm）是一种用于求解大规模线性系统 $bold(A) bold(x)=bold(b)$ 的迭代方法，其中 $bold(A)$ 是一个对称正定矩阵。这个算法尤其适用于稀疏矩阵，因为它可以避免直接求解矩阵的逆，降低了计算复杂度。

共轭梯度算法的基本思想是通过迭代的方法逐步逼近线性方程的解，利用前一步的解信息来加速收敛。其步骤通常如下：

1. 初始化：选择初始点 $bold(x)_0$，计算残差 $bold(r)_0 = b - bold(A) bold(x)_0$, 设置初始搜索方向 $bold(p)_0 = bold(r)_0$
2. 迭代：
  - 计算步长
  $ 
  alpha_k = (bold(r)_k^tack.b bold(r)_k)/(bold(p)_k^tack.b bold(A) bold(p)_k)
  $
  - 更新解
  $
  bold(x)_(k+1) = bold(x)_k + alpha_k bold(p)_k
  $
  - 更新残差
  $
  bold(r)_(k+1) = bold(r)_k - alpha_k bold(A) bold(p)_k
  $
  - 检查收敛条件，若满足则停止迭代
  $
  bold(r)_(k+1) = bold(c)
  $
  - 否则，计算新的搜索方向
  $
  bold(p)_(k+1) = bold(r)_(k+1) + beta_k bold(p)_k, beta_k = (bold(r)_(k+1)^tack.b bold(r)_(k+1))/(bold(r)_k^tack.b bold(r)_k)
  $
3. 重复步骤2，直到满足收敛条件

#pagebreak()
