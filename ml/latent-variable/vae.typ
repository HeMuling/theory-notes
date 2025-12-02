#import "../../prelude.typ": *

#hd3("VAE")

#hd4("Mixture PCA")
在线性代数下视角的PCA一般涉及特征值分解与主成分投影，对于 $n$ 个具有 $p$ 维特征的数据 $bold(X)in bb(R)^(n times p)$，将其中心化后计算协方差矩阵： #index("PCA")
$
  bold(Sigma) = 1/n bold(X)^tack.b bold(X)
$
然后对协方差矩阵进行特征值分解：
$
  bold(Sigma) bold(v)_j = lambda_j bold(v)_j, space.quad j = 1,2,dots, p
$
其中 $lambda_1>=lambda_2>=dots>=lambda_p>=0$ 为特征值，$bold(v)_j$ 是对应的特征向量。选择前 $k$ 个特征，将中心化后的数据投影到前 $k$ 个特征向量上，得到降维的表示：
$
  bold(Z) = bold(X) bold(V)_k in bb(R)^(n times k)\
  bold(V)_k = [bold(v)_1, bold(v)_2, dots, bold(v)_k] in bb(R)^(p times k)
$
在latent variable model视角下，PCA可以被视为一个潜变量模型，其中潜变量为降维后的数据。假设 $x in bb(R)^D, z in bb(R)^d, D >> d$，则：
$
  p(X,Z|theta) &= product_i p(x_i|z_i, theta) p(z_i|theta)\
  &= product_i cal(N)(x_i|bold(V)z_i+mu, sigma^2 I) cal(N)(z_i|0, I)
$ <PCA-latent1>
$theta$作为参数，包含 $bold(V) in bb(R)^(D times d), mu, sigma in bb(R)^D$
我们可以利用 EM 算法求解 latent variable model 视角下的PCA。以 PCA-latent1 中假设为gaussian分布为例，考虑到gaussian和gaussian互为共轭，因此可以获得解析解。使用EM而不是直接求解PAC的好处在于：
- EM算法每一个迭代的的复杂度为 $O(n D d)$，而直接求解PCA的复杂度为 $O(n D^2)$；因此当 $D>>d$ 时EM算法更加高效
- 可以处理缺失的 $x_i$ 或者多观察的 $z_i$
- 可以通过确定 $p(theta)$ 自动确定 $d$ 的值，而不需要想PCA一样提前确定
- 可扩展至混合PCA

现在考虑混合PCA(Mixture PCA)，即降维后的数据存在于多个子空间中，假设 $x in bb(R)^D, t in {1, dots, K}, z in bb(R)^d$，其中 $t$ 是每个子空间的索引，则有： #index("PCA", "Mixture PCA")
$
  p(X,Z,T|theta) &= product_i p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta)\
  & = product_i cal(N)(x_i|bold(V)_(t_i)z_i+mu_(t_i), sigma^2 I) cal(N)(z_i|0, I) pi_(t_i)
$
其中参数 $theta$ 包含 $bold(V)_k in bb(R)^(D times d), mu_k, sigma_k in bb(R)^D, pi_k in bb(R)^K$，并有 $p(t_i = k) = pi_k$

E-step:
$
  q(Z,T) &= p(Z,T|X, theta) = product_i p(z_i, t_i|x_i, theta)\
  &= product_i (p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta)) / (sum_i integral p(x_i|z_i, t_i, theta) p(z_i|theta) p(t_i|theta) d z_i)
$
M-step:
#set math.equation(number-align: top)
$
  theta_* &= arg max_theta bb(E)_(Z,T) log p(X,Z,T|theta)\
  &= arg max_theta sum_i bb(E)_(Z,T) [log p(x_i|z_i, t_i, theta) + log p(z_i|theta) + log p(t_i|theta)]
$
#set math.equation(number-align: horizon)

通常来说，PCA仅构造线性子空间，即只能捕捉数据中的线性关系。然而，在许多实际应用中，数据常常分布在非线性流形上。例如，在图像处理、自然语言处理等领域，数据往往具有复杂的模式和结构，这些模式和结构不能用平面或超平面来描述。

为了更好地处理非线性数据分布，可以使用一些其他的降维技术，例如：
- t-SNE：一种有效的降维方法，特别适合于处理高维数据的非线性结构，能够保留局部邻域的相似性。
- UMAP：类似于t-SNE，但更注重全局结构的同时保持局部结构。
- 自编码器 ：基于神经网络的方法，可以学习复杂的非线性映射，例如VAE。

#hd4("VAE") #index("VAE")

假设 $X in bb(R)^(n times D), Z in bb(R)^(n times d)$，则 latent variable model 告诉我们：
$
  p(X,Z|theta) &= product_i p(x_i|z_i, theta) p(z_i|theta)\
  &= product_(i=1)^n (product_(j=1)^D cal(N)(x_(i j)|mu_j (z_i), sigma^2_j (z_i)))cal(N)(z_i|0, I)
$
在这里，我们不需要局限 $cal(N)(dot)$ 的 $mu$ 为 $z_j$ 的线性函数，而是可以使用神经网络来表示非线性的 $mu_j (z_i)$ 和 $sigma^2_j (z_i)$，这样我们可以学习到更复杂的非线性关系

但使用非线性的 $mu_j (z_i)$ 和 $sigma^2_j (z_i)$ 会导致 $p(x_i|z_i, theta)$ 与 $p(x_i|theta)$ 不再共轭，因此无法获得解析解。同样，在EM算法的E-step中，我们也无法获得后验的解析解。即，无法计算：
$
  q(Z) = p(Z|X, theta) = (p(X,Z|theta))/p(X|theta)
$
从 EM-E-step 可知，我们从 $K L(q||p)$ 的定义直接推导出可以通过求解 $p(Z|X, theta)$ 来替代求解 $q(Z)$. 而在不能求解 $p(Z|X, theta)$ 的情况下，我们则利用 variational inference 的方法求解 $q(Z)$. 例如我们可以利用mean field approximation 的方法，即：
$
  q(Z) = product_i q_i (z_i)
$
但是在神经网络的视角下，我们完全可以另外#margin-note([训练一个神经网络用于拟合$q(Z)$])[回归神经网络的本质：拟合]，用参数 $phi$ 表示：
$
  q(z_i|x_i, phi) &approx p(z_i|x_i, theta)\
  q(z_i|x_i, phi) &= product_(j=1)^d cal(N)(z_(i j)|mu_j (x_i), sigma^2_j (x_i))
$
因此：
$
  &"encoder:" phi: x arrow.bar q(z|x, theta), bb(R)^D arrow bb(R)^(2d)\
  &"decoder:" theta: z arrow.bar p(x|z, theta), bb(R)^d arrow bb(R)^(2D)
$
其中 $2d$ 与 $2D$ 都是包括了 $mu$ 和 $sigma^2$ 的两个参数

优化神经网络 $phi$ 等价于：
$
  q(Z|X, phi) = arg min_phi K L(q(Z|X, phi)||p(Z|X, theta))
$ <VAE-opt>
根据 BaysianVariation1，VAE-opt 等价于最大化 ELBO：
$
  q(Z|X, phi) &= arg max_(phi, theta) cal(L)(phi, theta)\
  &= arg max_(phi, theta) integral q(Z|X,phi) log p(X,Z|theta)/q(Z|X,phi) d Z\
$
鉴于在非共轭情况下我们无法使用EM对 $q(Z|X, phi)$ 进行求解，我们因此使用 stochastic gradient 的方法

#hd4("Stochastic gradient") #index("Stochastic Gradient") #index("Mini-batch")

Stochastic gradient 通常使用mini-batch与Monte-Carlo estimation 来优化 ELBO，对于
$
  cal(L)(phi, theta) &= integral q(Z|X,phi) log p(X,Z|theta)/q(Z|X,phi) d Z\
  &=integral q(Z|X,phi) log (p(X|Z,theta)P(Z))/q(Z|X,phi) d Z
$
对 $theta$ 求导有：
#set math.equation(number-align: top)
$
  nabla_theta cal(L)(phi, theta) &= nabla_theta integral q(Z|X,phi) log (p(X|Z,theta)P(Z))/q(Z|X,phi) d Z\
  &= sum_(i=1)^n integral q(z_i|x_i, phi) nabla_theta log (p(x_i|z_i, theta)P(z_i))/(q(z_i|x_i, phi))d z_i\
  &= sum_(i=1)^n integral q(z_i|x_i,phi) nabla_theta log p(x_i|z_i, theta) d z_i\
  &approx n integral q(z_i|x_i, phi) nabla_theta log p(x_i|z_i,theta) d z_i, i tilde cal(U){1,dots, n} space.quad "(mini-batch)"\
  &approx n/abs(cal(U)) sum_(i in cal(U)) nabla_theta log p(x_i|z^*_i,theta), z^*_i tilde q(z_i|x_i, phi) space.quad "(Monte-Carlo estimation)"\
  &= n nabla_theta log p(x_i|z^*_i,theta), space.quad i tilde cal(U){1,dots, n}, abs(cal(U))=1, z^*_i tilde q(z_i|x_i, phi)
$
#set math.equation(number-align: horizon)

其中 $cal(U)$ 为数据集的子集，其大小为 $abs(cal(U))$，包含随机选择的$({x}_i, {z}_i)^abs(cal(U))$ ；$z^*_i$ 为从 mini-batch 中的 $q(z_i|x_i, phi)$ 采样的样本。通常来说，先对数据集进行 mini-batch 选择，然后在选定的 mini-batch 中进行 Monte-Carlo 采样，这样可以提高计算效率

对 $phi$ 求导有：
$
  nabla_phi cal(L)(phi, theta) = nabla_phi [&integral q(Z|X,phi) log p(X|Z,theta) d Z \
  - &integral q(Z|X,phi) (log q(Z|X,phi))/P(Z) d Z]\
$ <VAE-phi-dri>
第一项：
#set math.equation(number-align: top)
$
  nabla_phi integral q(Z|X,phi) log p(X,Z|theta) &= integral log p(X|Z,theta) nabla_phi q(Z|X,phi) d Z\
  &= integral q(Z|X,phi) log p(X|Z,theta) nabla_phi log q(Z|X,phi) d Z space.quad "(log-derivative trick)"\
  &= sum_(i=1)^n integral q(z_i|x_i,phi) log p(x_i|z_i,theta) nabla_phi log q(z_i|x_i,phi) d z_i\
  &= n integral q(z_i|x_i,phi) log p(x_i|z_i,theta) nabla_phi log q(z_i|x_i,phi) d z_i, i tilde cal(U){1,dots, n} space.quad "(mini-batch)"\
  &= n/abs(cal(U)) sum_(i in cal(U)) log p(x_i|z^*_i,theta) nabla_phi log q(z_i|x_i,phi), z^*_i tilde q(z^*_i|x_i, phi) space.quad "(Monte-Carlo estimation)"\
  &= n log p(x_i|z^*_i,theta) nabla_phi log q(z^*_i|x_i,phi), space.quad i tilde cal(U){1,dots, n}, abs(cal(U))=1, z^*_i tilde q(z_i|x_i, phi)
$
#set math.equation(number-align: horizon)

注意到 $nabla_phi log q(z^*_i|x_i,phi)$ 为分数函数 (score function)，具有以下性质： #index("Score Function")
- $
    bb(E)[nabla_theta log p(x|theta)] &= integral p(x|theta) nabla_theta log p(x|theta) d x\
    &= integral p(x|theta) 1/p(x|theta) nabla_theta p(x|theta) d x\
    &= nabla_theta integral p(x|theta) d x = 0
  $
  考虑到 $z^*_i$ 为我们抽样从 $q(z_i|x_i,phi)$ 的样本，因此可以认为 $nabla_phi q(z^*_i|x_i,phi)$ 在 $0$ 附近震荡，且其质量严格与抽样情况相关。除非 $n$ 具有较大值，否则几乎为0，因此可以认为梯度下降时速度较慢
- $
    "Var"(nabla_theta log p(x|theta)) &= bb(E)[nabla_theta log p(x|theta)^2]\
    &= I(theta)
  $
  可知 score function 的方差为 fisher information. Fisher information 反应了估计参数 $theta$ 的方差下界，即 Cramer-Rao Lower Bound，它越大，代表估计的 $theta$ 约精确。因此可以认为在梯度下降过程中，随着参数 $theta$ 的逼近精确值，方差逐渐增加，导致收敛速度变慢

综上所述 $nabla_phi log q(z^*_i|x_i,phi)$ 的存在会导致梯度下降效率降低，因此一般不这么做，而使用 reparameterization trick 来避免这个问题。

#hd4("Reparameterization trick")  #index("Reparameterization Trick")

考虑复杂期望的求导：
$
  partial / partial_x integral p(y|x) h(x, y) d y
$
假设 $y$ 可以被表达为一个随机变量 $epsilon$ 与 $x$ 的函数，即 $y = g(x, epsilon)$，利用Monte-Carlo estimation，我们可以将上式改写为：
$
  integral p(y|x) h(x, y) d y &= integral r(epsilon) h(x, g(x, epsilon)) d epsilon\
  &approx d/(d x) h(x, g(x, epsilon^*)), space.quad epsilon^* tilde r(epsilon)\
  &= partial/(partial x) h(x, g(x, epsilon^*)) + partial/(partial y) h(x, g(x, epsilon^*)) partial/(partial x) g(x, epsilon^*)\
$
常见的 reparameterization trick 有：
#figure(
  table(
    columns: (auto, auto, auto),
    rows: (2em, 3em, 3em, 3em),
    [$p(y|x)$], [$r(epsilon)$], [$g(epsilon,x)$],
    [$cal(N)(y|mu, sigma^2)$], [$cal(N)(epsilon|0, 1)$], [$x = mu + sigma epsilon$],
    [$cal(G)(y|1,beta)$], [$cal(G)(epsilon|1, 1)$], [$x = beta epsilon$],
    [$epsilon(y|lambda)$], [$cal(U)(epsilon|0,1)$], [$x = -log(epsilon)/lambda$],
    [$cal(N)(y|mu, Sigma)$], [$cal(N)(epsilon|0,I)$],[$x=A epsilon + mu$, where  $A A^tack.b = Sigma$]
  )
)

// 变分RNN阅读笔记
#pagebreak()
