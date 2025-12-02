#import "../../prelude.typ": *

#hd3("EM Algorithm")  #index("EM Algorithm")
#hd4("EM Algorithm")

在最大化 $log p(X|theta)$ 时，我们不仅要最大化参数 $theta$，还要最大化潜变量 $Z$ 的分布，即
$
  cal(L)(q, theta) = integral q(Z) (p(X,Z|theta))/q(Z) d Z arrow max_(q,theta)
$
E-step，设置一个初始点 $theta_0$，类似 BaysianVariation1：
$
  q(Z) &= arg max_q cal(L)(q, theta_0) = arg min_q  K L(q||p)\
  &= p(Z|X, theta_0) = (p(X,Z|theta_0))/p(X|theta_0) =  (p(X,Z|theta_0))/ (integral_i p(X,z_i|theta_0) d z_i)\
$ <EM-E-step>

当 $p(Z|X, theta_0)$ 无法获得解析解时，可以采取variational inference的方法。

M-step，考虑到 $Z$ 的具体值不明确但知道其分布 $q(Z)$，我们采用其期望：
$
  theta_* = arg max_theta cal(L)(q, theta) = arg max_theta bb(E)_Z log p(X,Z|theta)
$
将新的到的 $theta_*$ 传入 E-step，重复直到收敛。在更新过程中 variational lower bound 是单调递增的，因此可以保证收敛

#figure(
  image(
    "../../assets/EM-algorithm.png",
    width: 100%
    )
)

#hd4("Categorical latent variables")  #index("Categorical Latent Variables")

对于离散型潜变量，假设 $z_i in {1,dots,K}$，则
$
  p(x_i|theta) = sum_k p(x_i, z_i = k|theta) = sum_k p(x_i|z_i = k, theta) p(z_i = k|theta)
$
进行EM，E-step：
$
  q(z_i = k) &= p(z_i = k|x_i, theta) \
  &= (p(x_i|z_i = k, theta) p(z_i = k|theta)) / p(x_i|theta)\
  &= (p(x_i|z_i = k, theta) p(z_i = k|theta)) / (sum_l p(x_i|z_i = l, theta) p(z_i = l|theta))
$
M-step：
$
  theta_* &= arg max_theta bb(E)_Z log p(X,Z|theta)\
  &= arg max_theta sum_i sum_k q(z_i = k) log p(x_i, z_i = k|theta)
$
而对于连续型潜变量：
$
  p(x_i|theta) = integral p(x_i, z_i|theta) d z_i = integral p(x_i|z_i, theta) p(z_i|theta) d z_i
$
E-step：
$
  q(z_i) &= p(z_i|x_i, theta) = p(z_i|x_i, theta) / p(x_i|theta)\
  & = (p(x_i|z_i, theta) p(z_i|theta)) / (integral p(x_i|z_i, theta) p(z_i|theta) d z_i)
$
根据 conjugate，只有当 $p(X|Z,theta)$ 与 $p(Z|theta)$ 为共轭分布时，E-step才能获得解析解；否则，需要使用stochastic variational inference的方法

对于连续型潜变量，其重要应用之一在于representation learning：
1. 表示学习的目标 ：Representation learning 的核心目标是生成有效的数据表示，而连续潜变量提供了一个强大的工具来建模数据的内在结构。
2. 潜变量在表示学习中的作用 ：通过引入连续潜变量，模型能够更灵活地捕捉数据的连续变化和模式，形成有效的表示。这些潜变量通常在隐藏层中起作用，影响最终输出的生成。
3. 生成模型中的应用 ：许多现代的生成模型，如GAN（生成对抗网络）和VAE，都利用连续潜变量来生成新样本，通过学习数据的潜在结构来提高生成能力。
4. 优化和推断 ：在representation learning的上下文中，涉及到从观测数据中推断潜变量的分布，并优化这些潜变量以获得更好的数据表示。连续潜变量可以利用梯度下降等优化方法进行推断。
