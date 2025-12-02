#import "../../prelude.typ": *

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
