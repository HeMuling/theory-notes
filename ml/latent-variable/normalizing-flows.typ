#import "../../prelude.typ": *

#hd3("Normalizing Flows") #index("Normalizing Flows")
来源与论文dinh2016density，与VAE类似，normalizing flows 假设潜变量 $z$，并可以根据数据 $x$ 得到潜变量，即：
$
  z = f_theta (x)
$
与VAE不同的是，normalizing flows 不使用 decoder 从 $z$ 获得 $x$，而是期望找到 $f^(-1)_theta (dot)$ 使得：
$
  x = f^(-1)_theta (z)
$
根据变量转换公式，我们有：
$
  p(x) &= p(z) abs(det (d f^(-1)_theta (z))/(d z))\
  &= p(f(x)) abs(det (d f_theta (x))/(d x)) 
$ <nf-change>
此时需要保证 $z$ 和 $x$ 的维度相同。同时，还需要保证 $f_theta (dot)$ 是可逆的，因此设计如下灵活且可解决的双射函数作为 coupling layer. 假设 $x in bb(R)^D$ 且 $d < D$，有：
$
  y_(1:d) &= x_(1:d)\
  y_(d+1:D) &= x_(d+1:D) dot.o exp(s(x_(1:d))) + t(x_(1:d))
$ <nf-forward>
#figure(
  image("../../assets/NormalizingFlowsCP.png", width: 80%)
)
可以很容易得到其逆变换：
$
  x_(1:d) &= y_(1:d)\
  x_(d+1:D) &= (y_(d+1:D) - t(y_(1:d))) dot.o exp(-s(y_(1:d)))
$
其中 $s,t: bb(R)^d arrow.bar bb(R)^(D-d)$. 这样，Jacobian 矩阵为：
$
  (partial y)/(partial x^tack.b) = mat(
    I_d, 0;
    (partial y_(d+1:D))/(partial x^tack.b_(1:d)), "diag"(exp[s(x_(1:d))])
  )
$
代入 nf-change，我们有：
$
  abs(det (d f_theta (x))/(d x)) = exp(sum_j s(x_(1:d))_j)
$
注意到，在 nf-forward 中，$y_(1:d) = x_(1:d)$ 并没有经过变换，我们可以结合多个不同的 coupling layer 来解决这个问题，对于在一个 coupling layer 上未经变换的部分，我们让其在下一个 coupling layer 进行变换。即：
$
  f_theta (x) = f^N circle.small dots.h.c circle.small f^1 (x)
$
因此，根据MLE：
$
  log p_theta (x) &= log p(f_theta (x)) + log abs(det (partial f_theta (x))/ (partial x^tack.b))\
  &= log p(f_theta (x)) + sum_i^N log abs(det (partial f^i)/ (partial f^(i-1)))
$

#pagebreak()
