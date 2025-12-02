#import "../../prelude.typ": *

#hd3("GAN")

#hd4("GAN") #index("GAN")
GAN 的思想在于，使神经网络生成的分布 $q(x)$ 与真实分布 $p(x)$ 尽可能接近。接近程度由判别器决定：
$
  f(x;phi) = p(y=1|x, phi)
$ <gan1>
或者计算两个分布之间的举例：
$
  D_f (p||q)
$ <gan2>
通过判别器的指示，我们让 $q(x)$ 逐渐逼近 $p(x)$. 在这里我们需要对 $q(x)$ 这个分布进行采样，但$q(x)$ 的具体分布不知道，因此我们可以训练一个生成器：
$
  hat(x) = G(z;theta), space.quad z tilde p(z)
$
使得：
1. $z$ 容易从 $p(z)$ 中采样，例如 $cal(N)(0,I)$，
2. 采样的结果可以映射到 $q(x)$ 中，即 $hat(x) tilde q(x)$
如果使用 gan1, 考虑到这是一个二分类问题，且判别器需要尽可能分开 $p(x)$ 与 $q(x)$，因此其 loss 为：
$
  cal(L)(phi, theta) = bb(E)_(p(x))[log f(x;phi)] + bb(E)_(p(z))[log(1-f(G(z;theta);phi))]
$ <gan-binary>
上式等价于使用 gan2, 其loss为：
$
  cal(L)(phi, theta) = D_f (p(x)||q(x;theta))
$ <gan-df>
我们只需要
1. 更新判别器：
$
  phi = arg max_phi cal(L)(phi, theta)
$
2. 更新生成器（最小化 $p,q$ 距离）：
$
  theta^(t+1) = theta^t + nabla_theta cal(L)(phi, theta^t)
$
3. 重复1,2直至收敛
其中，$D_f$ 为 f-divergence，可以是任意的 divergence measure  #index("Divergence Mesure")
$
  D_f (P||Q) = integral_cal(X) f (p(x)/q(x)) q(x) d x
$
其中 $f$ 指明了 divergence 的形式：
$
  f(t) = cases(
    t log t \, & space.quad "KL-divergence",
    - log t \, & space.quad "Reverse KL-divergence",
    1/2 abs(t - 1) \, & space.quad "Total variation",
  )
$

#hd4("Optimal transport") #index("Optimal Transport")
根据 gan-binary 和 gan-df 我们知道：最小化generator的loss等价于最小化generator生成的分布与target分布的JS divergence。

但是用JS divergence来作为度量有个致命缺陷，就是在两个分布互不相交的情况下，两个分布的JS divergence永远都是常数 $log 2$，并且由于generator生成的分布和target分布的支撑集是在嵌在高维空间中的低维流形，所以他们重叠的部分的测度几乎为0。这样完全无法进行度量两个分布在不相交情况下的距离。计算梯度的时候也会出现0梯度的情况。ymhuang2024

因此我们需要采用新的度量方式，这就是optimal transport. Optimal transport是一个线性规划问题，它的目标是找到两个分布之间的最小运输成本。假设从 $x$ 运输到 $y$ 具有一定的成本 $c$，一般来说，定义为：
$
  c(x,y) = ||x-y||_k^k
$
原始定义为：
$
  L = arg min_(Gamma) sum_(i,j)^(M,N) Gamma_(i,j) c(x_i,y_j)
$
optimal transport 具有概率版本，其optimal transport divergence 为
$
  T(P,Q) = inf_(Gamma in P(x tilde P, y tilde Q)) bb(E)_((x,y) tilde Gamma) c(x,y)
$
其中，$Gamma$ 为 $P$ 到 $Q$ 的一个联合分布，表示从 $P$ 到 $Q$ 的运输方案，满足联合分布的性质。转换为对偶问题 (dual problem) 为：
$
  T(P,Q) = sup_(phi,psi in L_1) (bb(E)_(p(x))phi(x) + bb(E)_(q(y))psi(y))
$
其中 $L_1:{phi(x)+psi(x)<=c(x,y)}$

因此，需要保证神经网络的函数是光滑的 Lipschitz 函数，即：
$
  ||f(x) - f(y)||_K <= L ||x-y||_K
$

对于 FFN，由仿射变换和逐点非线性组成的函数，这些非线性是光滑的 Lipschitz 函数（例如 sigmoid, tanh, elu, softplus 等）arjovsky2017wasserstein. 对于线性矩阵计算函数，通过以下方式判断：
$
  ||A x_1 - A x_2||_2 <= L||x_1 - x_2||_2\
  sigma_max = sup_x (||A x||_2)/(||x||_2) <= L
$
其中 $sup_x (||A x||_2)/(||x||_2)$ 刚好等价于矩阵的 spectral norm, 即矩阵的最大奇异值。因此，为了使矩阵 $A$ 为满足 Lipschitz 函数，只需要使
$
  A := A / sigma_max
$
可以使用 power iteration 来计算 $sigma_max$ (#link("https://en.wikipedia.org/wiki/Power_iteration")[power iteration wiki])  #index("Power Iteration")

#hd4("Gan Algorithm")
$
  &min_theta min_phi cal(L)(theta, phi) =\
  &min_theta min_phi bb(E)_(p(x)) log D(x;phi) + bb(E)_p(z) log(1-D(G(z;theta);phi))
$
1. $
     phi^(t+1) = phi^t + alpha nabla_phi cal(L)(theta^t, phi^t)
   $
2. $
      theta^(t+1) = theta^t - alpha nabla_theta cal(L)(theta^t, phi^(t+1))
   $
3. 重复1,2直至收敛

#pagebreak()
