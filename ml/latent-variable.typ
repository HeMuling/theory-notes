#import "../prelude.typ": *

#hd2("Latent variable model")

#hd3("Laten variable model")  #index("Latent Variable Model")

潜变量模型（Latent variable model）是一种统计模型，其中包含了一些未观测的变量，这些变量通常被称为潜变量（latent variable）。潜变量模型通常用于描述数据背后的潜在结构，以及数据生成的机制。潜变量模型可以用于多种任务，如聚类、降维、异常检测等。

对于观测数据 $X$ 与其模型参数 $theta$，要估计模型参数 $theta$，通常采用MLE方法，即
$
theta_("MLE") = arg max_theta log p(X|theta)
$

我们假设存在某种潜在变量 $Z$，其与观测数据 $X$ 之间存在关系，此时我们可以对 $log p(X|theta)$ 进行分解，类似 BaysianVariation1：
$
  log p(X|theta) &= integral q(Z) (p(X,Z|theta))/q(Z) d Z  + integral q(Z) log q(Z)/p(Z|X,theta) d Z\
  &= cal(L)(q, theta) + K L(q||p) >= cal(L)(q, theta)
$
因此，我们可以通过最大化 ELBO $cal(L)(q, theta)$ 来使 $log p(X|theta)$ 尽可能大。对于ELOB，给出其广泛定义 variational lower bound:

#showybox()[  #index("Variational Lower Bound")
  函数 $g(xi, x)$ 是另一函数 $f(x)$ 的 variational lower bound，当且仅当：

  - $forall xi, f(x) >= g(xi, x)$
  - #margin-note($forall x_0, exists xi(x_0) arrow.double f(x_0) = g(xi(x_0), x_0)$)[例如过二次函数最低点的切线]
  这样，对于：
  $
    x = arg max_x f(x)
  $
  我们可以通过对 $g(xi, x)$ 区块坐标更新（Block-coordinate updates）来获得 $x$ 的近似解，即：
  $
    x_n &= arg max_x g(xi_(n-1), x)\
    xi_n &= xi(x_n) = arg max_xi g(xi, x_n)
  $
]

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
    "../assets/EM-algorithm.png",
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
#hd3("Variational RNN") #index("RNN", "Variational RNN")

#hd4("Motivation")

对于通常RNN来说，其训练为：
$
  h_t = f_theta (h_(t-1), x_t)
$
而推理过程则使用训练好的神经网络 $theta$，这导致RNN网络的变异性较低，即：输出的任何变化或波动都仅仅取决于RNN经过训练所学习到的 $theta$。

这意味着，RNN生成的输出受学习到的模式和规律影响，而不是受隐藏状态之间的直接转移过程的影响。换句话说，内部状态改变（即隐藏状态的变化）并不会直接引入新的变异性，所有变化都通过输出层的概率机制来实现。

因此，作者提出：在每一个时间步加入一个VAE机制，隐变量 $z_t$ 从 $cal(N)(theta_t)$ 采样，但 $theta_t$ 又由 $h_(t-1)$ 决定。这样，$z_t$ 的变化将直接影响到输出的变异性。

#hd4("数学公式")

#figure(
  image("../assets/VariationalRNN.png")
)

在生成过程中：

对于每个时间步 $t$ 的VAE，其隐变量 $z_t$ 采样于：
$
  z_t tilde cal(N)(mu_(0,t), "diag"(sigma_(0,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $h_(t-1)$ 生成：
$
  mu_(0,t), sigma_(0,t) = phi_tau^("prior") (h_(t-1))
$
$x_t$ 的生成则为：
$
  x_t|z_t tilde cal(N)(mu_(x,t), "diag"(sigma_(x,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $z_t$ 和 $h_(t-1)$ 生成：
$
  mu_(x,t), sigma_(x,t) = phi_tau^("dec") (phi_tau^z (z_t), h_(t-1))
$
对于RNN：
$
  h_t = f_theta (phi_tau^x (x_t), phi_tau^z (z_t), h_(t-1))\
  p(x_(<=T), z_(<=T)) = product_(t=1)^T p(x_t|z_(<=t),x_(<t))p(z_t|x_(<t),z_(<t))
$
在推理过程中：

隐变量的后验采样于：
$
  z_t|x_t tilde cal(N)(mu_(z,t), "diag"(sigma_(z,t)^2))
$
其中，$cal(N)(dot)$ 的参数由 $x_t$ 和 $h_(t-1)$ 生成：
$
  mu_(z,t), sigma_(z,t) = phi_tau^("enc") (phi_tau^x (x_t), h_(t-1))
$
进而：
$
  q(z_(<=T)|x_(<=T)) = product_(t=1)^T q(z_t|x_(<=t),z_(<t))
$

我们的目标函数为：
$
  bb(E)_(q(z<=T|x<=T))[sum_(t=1)^T (-"KL"(q(z_t|x_(<=t),z_(<t))||p(z_t|x_(<t),z_(<t)))& \
  + log p(x_t|z_(<=t),x_(<t)))]&
$

reparameterization trick 并不适用于所有连续分布，且不适用于离散分布。对于离散分布，我们可以使用 Gumbel-Softmax trick 来进行 reparameterization

对于ELBO VAE-phi-dri 第一项，我们首先进行mini-batch，然后对其使用 reparameterization trick，最后使用Monte-Carlo estimation：
$
  nabla_phi integral q(Z|X,phi) log p(X|Z,theta) &approx n nabla_phi integral q(z_i|x_i, phi) log p(x_i|z_i,theta) d z_i\
  &=n nabla_phi integral r(epsilon) log p(x_i|g(epsilon,x_i,phi)z_i, theta) d epsilon\
  &approx n nabla_phi log p(x_i|g(epsilon^*,x_i,phi)z_i, theta)\
  &i tilde cal(U){1,dots, n}, abs(cal(U))=1, z_i=g(epsilon,x_i,phi), epsilon^* tilde r(epsilon)
$

#hd4("VAE Algorithm")
我们的目标函数则为：
$
  cal(L)(phi,theta) = bb(E)_(q(Z|X,phi)) log p(X|Z, theta) - K L(q(Z|X,phi)||p(Z))
$
其中， $q(Z|X,phi)$ 为 encoder 网络，$p(X|Z,theta)$ 为decoder

接着更新 $phi, theta$，迭代直至收敛。因为两个都是无偏估计且存在ELBO的下界保证，因此可以保证收敛

对于 $X in bb(R)^(n times d)$，随机选取 mini-batch $cal(U) = ({x}_i, {z}_i)^abs(cal(U))$，计算：
$
  "stoch.grad"_theta cal(L)(phi, theta) = n/abs(cal(U)) sum_(i in cal(U)) nabla_theta log p(x_i|z^*_i,theta)
$
其中 $z_i^* tilde q(z_i|x_i,phi)$
$
  "stoch.grad"_phi cal(L)(phi, theta) = n/abs(cal(U)) sum_(i in cal(U)) nabla_phi log p(x_i|g(epsilon^*,x_i,phi), theta)&\
  - nabla_phi K L(q(z_i|x_i,phi)||p(z_i))&
$
其中 $z_i=g(epsilon,x_i,phi), epsilon^* tilde r(epsilon)$

#hd4("VAE Code")

对于目标函数：
$
  cal(L)(phi,theta) = bb(E)_(q(Z|X,phi)) log p(X|Z, theta) - K L(q(Z|X,phi)||p(Z))
$
对于第一项期望，可以使用 Monte-Carlo estimation，这样只剩下 $log p(X|Z,theta)$，根据数据类型一般考虑 $p(X|Z,theta)$ 为 Bernoulli 或者 Gaussian 分布，在 Bernoulli 分布下，我们可以使用交叉熵损失函数:
$
  "loss" = -sum_i sum_j [x_(i j) log y_(i j) + (1-x_(i j)) log (1-y_(i j))]
$
考虑到 $x_(i j)$ 与 $y_(i j)$ 范围取值为 $[0,1]$，因此通常来说都可以使用交叉熵损失函数

对于KL-divergence，考虑到 $q(Z|X,phi)$ 为高斯分布，$p(Z)$ 为标准高斯分布，因此 KL-divergence 可以直接计算：
$
  K L(q(Z|X,phi)||p(Z)) = 1/2 sum_(i=1)^d [1 + log(sigma_i^2) - mu_i^2 - sigma_i^2]
$
#showybox(breakable: true)[
  proof：

  对于两高斯分布的 KL-divergence 的：
  #set math.equation(number-align: top)
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(mu', Sigma')) =\
    &1/2 [tr(Sigma'^(-1) Sigma) + (mu'-mu)^tack.b Sigma'^(-1) (mu'-mu) - k + log(det(Sigma'^(-1) Sigma))]
  $
  其中 $Sigma$ 为协方差矩阵，$k$ 为维度

  特殊的，当一个高斯分布为标准高斯分布时，其 KL-divergence 为：
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(0, I)) = 1/2 [tr(Sigma) + mu^tack.b mu - k - log(det(Sigma))]
  $
  更特殊的，对于对角协方差矩阵$Sigma = "diag"(sigma_1^2, sigma_2^2, dots, sigma_k^2)$
  有：
  $
    "tr"(Sigma) = sum_(i=1)^k sigma_i^2, space.quad "det"(Sigma) = product_(i=1)^k sigma_i^2
  $
  因此：
  $
    "KL" (cal(N)(mu, Sigma)&||cal(N)(0, I)) = 1/2 [sum_(i=1)^k (1 + log(sigma_i^2) - mu_i^2 - sigma_i^2)]
  $
]
因此，目标函数可以改写为：
$
  cal(L)(phi, theta) tilde.eq 1/2 sum_(i=1)^d [1 + log(sigma_i^2) - mu_i^2 - sigma_i^2] + 1/L sum_(l=1)^L log p(x|z_l, theta)
$
其中 $z_l = mu + sigma dot.o epsilon, epsilon tilde cal(N)(0,I)$

```python
def loss_function(recon_x, x, mu, logvar):
    """
    计算VAE的损失函数，包括重构损失和KL散度
    """
    # 重构损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL散度计算
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```
对于 encoder $P(Z|X,phi)$，要计算 $mu$ 和 $log sigma^2$，这样我们可以根据 $mu$ 和 $log sigma^2$ 采样 $z$，这样我们可以保证 $z$ 为高斯分布
```python
def encode(self, x):
    """
    编码器前向传播，输出潜在变量的均值和对数方差
    """
    h1 = F.relu(self.fc1(x))
    mu = self.fc_mu(h1)
    logvar = self.fc_logvar(h1)
    return mu, logvar
def reparameterize(self, mu, logvar):
    """
    重参数化技巧，从 Q(Z|X) 中采样潜在变量 Z
    """
    std = torch.exp(0.5 * logvar)  # 标准差 σ
    eps = torch.randn_like(std)    # 从标准正态分布中采样 ε
    return mu + eps * std          # 潜在变量 Z
```
对于 decoder $P(X|Z,theta)$，我们可以直接计算重构的 $x$：
```python
def decode(self, z):
    """
    解码器前向传播，重构输入
    """
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))  # 输出范围 [0,1]
```

#pagebreak()
#hd3("Discrete Latent Variables")

#hd4("Reinforce estimator") #index("Reinforce Estimator")

在连续潜变量模型中，我们利用 reparameterization trick 对 VAE-phi-dri 的第一项进行了求导。而对于离散潜变量模型，我们无法使用 reparameterization trick，因此我们需要使用 Reinforce estimator. 考虑：
$
  cal(L)(phi) = sum_Z q(Z;phi) f(Z) = bb(E)_(q(Z;phi)) f(Z)
$ <reinforce-estimator-eg>
对其求导:
$
  nabla_phi cal(L)(phi) &= sum_Z nabla_phi q(Z;phi) f(Z)\
  &= sum_Z q(Z;phi) f(Z) nabla_phi log q(Z;phi), space.quad "(log-derivative trick)"\
  &= 1/M sum_(m=1)^M f(z_m) nabla_phi log q(z_m;phi), space.quad "(Monte-Carlo estimation)"
$
因此 reinforce estimator 为：
$
  g(z_(1:M), phi) = 1/M sum_(m=1)^M f(z_m) nabla_phi log q(z_m;phi), space.quad z_m tilde q(z_m;phi)
$ <reinforce-estimator>
注意到，reinforce estimator 不仅允许离散潜变量，甚至允许不可微函数 $f(dot)$ 的存在。但同时存在以下缺点：
1. 方差较大，通过增大 $M$ 只会使std以 $1\/sqrt(M)$ 的速率减小
2. 其梯度方向由向量 $nabla_phi log q(z_m;phi)$ 决定，步长由标量 $f(z_m)$ 决定，因此可以认为梯度方向指向 $Z$ 概率增加的方向。
3. 不像 reparameterization trick, reinforce estimator 缺少 $nabla_phi f(Z)$ 的信息（例如VAE中decoder的梯度信息），而只使用了值；因此还对函数 $f(Z)$ 的移动 (e.g. $f(x)+c$) 敏感

#hd4("Gumbel-SoftMax trick")
考虑到 reinforce-estimator 这么多的缺点，一个合理的想法是将离散潜变量转化为连续潜变量，然后使用 reparameterization trick，即：
$
  bb(E)_(q(Z;phi)) f(Z) = bb(E)_(q(tilde(Z)|phi)) f(tilde(Z)) = bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))
$
其中 $gamma$ 为噪声，$tilde(Z)(gamma, phi)$ 为 $Z$ 的连续估计，此时要确保 $f(dot)$ 连续可微

一种方式是使用 Gumbel-Max trick，假设 $z$ 为 $K$ 类离散变量，各自概率为 ${pi_i}_(i=1)^K, sum_i pi_i = 1$，则
$
  z &= arg min_i zeta_i/pi_i, space.quad zeta_i tilde "Exp"(1)\
  &= arg max_i [log pi_i - log zeta_i]\
  &= arg max_i [log pi_i + gamma_i], space.quad gamma_i tilde "Gumbel"(0,1)
$ #index("Gumbel-Max trick")
唯一的问题在于 $arg max(dot)$ 不可导
#showybox()[
  proof:

  对于：
  $
    Y_i = zeta_i/pi_i = g(zeta_i), space.quad zeta_i tilde "Exp"(1)
  $
  计算其pdf：
  $
    f_(Y_i)(y) &= f_(zeta_i)(g^(-1)(zeta_i))abs(d/(d y) g^(-1)(zeta_i))\
    &= f_(zeta_i)(pi_i y) pi_i\
    &= pi_i e^(-pi_i y)
  $
  因此：
  $
    Y_i  tilde "Exp"(pi_i)
  $
  且：
  $
    P(Y_i = min {Y_j}_(j=1)^K) &= integral_0^(+infinity) P(Y_i=y) product_(i eq.not j) P(Y_j >= y) d y\
    &= integral_0^(+infinity) pi_i e^(-pi_i y) product_(i eq.not j) e^(- pi_j y) d y\
    &= integral_0^(+infinity) pi_i exp(-pi_i y + sum_(i eq.not j) -pi_j y)d y\
    &= integral_0^(+infinity) pi_i exp(-y) d y\
    &= pi_i
  $
]
因此使用 Gumbel-SoftMax trick，即使用带温度控制的 SoftMax 替代 argmax:  #index("Gumbel-SoftMax Trick")
$
  "softmax"(x;tau)_j = (exp(x_j\/tau))/(sum_i exp(x_i\/tau))
$
其中温度 $tau$ 控制与 argmax 的相似性：
- 当 $tau=0$ 时，$"softmax"="argmax"$
- 当 $tau=infinity$ 时，$"softmax"="Uniform"$
因此：
$
  tilde(z)(gamma,pi) = "softmax"(log pi_i+gamma_i;tau), space.quad i = 1,dots,K
$
其中 $gamma_i tilde "Gumbel"(0,1)$，等价于
$
  gamma_i = -log(log u_i), space.quad u_i tilde "Uniform"(0,1)
$
此时，reinforce-estimator-eg 可改写为：
$
  cal(L)(phi) = bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))
$
此时
$
  nabla_phi cal(L)(phi) &= nabla_phi bb(E)_(p(gamma)) f(tilde(Z)(gamma, phi))\
  &= nabla_phi f(tilde(Z)(gamma^*, phi)), space.quad gamma^* tilde "Gumbel"(0,1)
$
引入噪声 $gamma$ 的好处有：
- 提升泛化能力
- 正确的噪声类型（例如 $"Gumbel"(0,1)$）可以使 $tilde(z)$ 类似于 one-hot vector，提升训练集与测试集的相似度

对于温度 $tau$，通常使 $tau <= 1\/(K-1)$，并且使用grid search搜索。小 $tau$ 会导致高方差，但更能表示离散值；大 $tau$ 反之。

#hd4("Control Variates")  #index("Control Variates")

对于 reinforce-estimator，另一个合理的想法是控制 reinforce estimator的方差，利用 variant control，通过减去已知量的估计误差来降低未知量的估计误差. 假设对于未知随机变量 $V(X)$，以及已知随机变量 $W(X)$ 与 其计算出的期望 $mu = bb(E)[W(X)]$，我们可以构造一个新的随机变量，与 $V(X)$ 具有相同期望，但方差更小：
$
  Z = V(X) - alpha(W(X) - mu)
$
此时方差为：
$
  "Var"(Z) = "Var"(V(X)) - 2 alpha "Cov"(V(X), W(X)) + alpha^2 "Var"(W(X))
$
注意到上式为关于 $alpha$ 的二次方程，最小值为：
$
  "Var"_min (Z) &= "Var"(V(X)) - ("Cov"^2(V(X), W(X)))/("Var"^2(W(X)))\
  alpha^* &=  "Cov"(V(X), W(X)) / "Var"(W(X))
$
除此以外，我们还可以利用多个已知变量 $W_i (X)$ 来降低方差（查看 goodman2005montecarlo）

因此，假设 $mu = bb(E)_(q(Z;phi))[b(Z)]$，则 reinforce-estimator 可以改写为：
$
  cal(L) &= bb(E)_(q(Z;phi))[f(Z) - b(Z) + mu]\
  &= bb(E)_(q(Z;phi))[f(Z) - b(Z)] + mu(phi)
$
利用 Monte-Carlo estimation，我们可以得到：
$
  cal(L) &= bb(E)_(q(z_(1:M)|phi))[1/M sum_(m=1)^M f(z_i)-b(z_i)] + mu(phi)
$
求导得到 reinforce estimator：
$
  g(z_(1:M), phi) = 1/M sum_(m=1)^M [f(z_m) - b(z_m)] nabla_phi log q(z_m;phi) +nabla_phi mu(phi)
$
其中 $z_i tilde q(Z;phi)$，$b(Z)$ 被称为 baseline, $b(Z) nabla_phi log q(Z;phi)$ 被称为 control variate. 接下来讨论 baseline 的选择： #index("Baseline")

1. 选择 baseline 为常数 $b(Z) = c$，有 $nabla_phi mu(phi)=0$，因此：
  $
    "Var"(g) &= "Var" (1/M sum_(m=1)^M [f(z_m) - b(z_m)] nabla_phi log q(z_m;phi))\
    &= 1/M^2 sum_(m=1)^M "Var"([f(z_m) - c] nabla_phi log q(z_m;phi)), space.quad "i.i.d. "z_m\
    &= 1/M^2 sum_(m=1)^M ["Var"(f dot nabla_phi)+c^2 "Var"(nabla_phi)-2 c "Cov" (f dot nabla_phi,nabla_phi)]
  $
  因此，$c$ 的最佳选择为：
  $
    c^* = "Cov"(f(Z) nabla_phi log q(Z;phi), nabla_phi log q(Z;phi))/"Var"(nabla_phi log q(Z;phi))
  $
  但如果有额外的观察项 (例如VAE中的 $log q(Z|X,phi)$)，则最佳 baseline 的选择应该与 $x$ 有关，因此不再适用于上式
2. NVIL: 针对上面的问题，mnih2014neural 提出了使用 MSE 来估计 baseline：
  $
    b(X) = arg min_b bb(E)_p(X) bb(E)_(q(Z|X,phi))[f(Z) - b(X)]^2
  $
3. MuProp: gu2015muprop 提出了将 $f(z)$ 的一阶泰勒展开作为 baseline：
  $
    b(Z) = f(mu) + nabla_Z f(mu)^tack.b dot (Z-mu)
  $
  对于 $mu$ 取 $mu = mu(phi)=bb(E)_(q(Z;phi))Z$，有
  $
    g(Z,phi) = (f(Z)-b(Z)) nabla_phi log q(Z;phi) + nabla_phi f(mu(phi))
  $
除此以外，还有其他不同的方法，如 maddison2016concrete, tucker2017rebar, etc.

#pagebreak()
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
  image("../assets/NormalizingFlowsCP.png", width: 80%)
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
