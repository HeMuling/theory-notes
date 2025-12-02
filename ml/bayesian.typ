#import "../prelude.typ": *

#hd2("贝叶斯")

贝叶斯涉及以下组件：

似然函数（likelihoo）：表示观测数据在参数 $theta$ 给定情况下的概率，通常记作 $p(D|theta)$，其中 $D$ 为观测数据

先验分布（prior distribution）：表示在没有观测数据时对参数 $theta$ 的信念，记作 $p(theta)$.

后验分布（posterior distribution）：表示在观测数据更新后参数分布，记作 $p(theta|D)$，通常由贝叶斯定理进行计算

#hd3("基础")

#hd4("贝叶斯优点")

贝叶斯的基础形式为：
$
p(y|x) = p(x,y)/p(x) = (p(x|y)p(y))/(p(x)) = (p(x|y)p(y))/(integral p(x|y)p(y) d y)
$
即：
$
text("Posterior") = (text("Likelihood") times text("Prior"))/text("Evidence")
$
考虑一系列观测数据 $X=(x_1,x_2,dots,x_n)$，i.i.d.来自某个分布 $p(x|theta)$，其中 $theta$ 为参数。我们希期通过观测数据来估计参数 $theta$，即获得 $p(theta|X)$.通常情况下我们可以利用MLE进行处理：
$
theta_(text("MLE")) = arg max_theta p(X|theta)=arg max_theta sum_i log p(x_i|theta)
$
如果利用贝叶斯方法，我们可以得到：
$
p(theta|X)=(p(X|theta)p(theta))/p(X) = (p(X|theta)p(theta))/(integral p(X|theta)p(theta) d theta) op("=", limits: #true)^(i i d) (product_i p(x_i|theta)p(theta))/(integral product_i p(x_i|theta)p(theta) d theta)
$
这里的性质在于，使用贝叶斯方法得到的后验概率分布 $p(theta|X)$ 包括了观测数据的信息，这样当我们有新的观测数据时，可以直接利用后验概率分布来估计参数，例如：
$
p(theta|X,x_(n+1)) = (p(x_(n+1)|theta)p(theta|X))/p(x_(n+1)|X) op("=", limits: #true)^(i i d) (p(x_(n+1)|theta)p(theta|X))/(p(x_(n+1)))
$
贝叶斯的优点在于：无论数据大小，都可以得到后验概率分布，这样可以避免过拟合问题。

#hd4("Probabilistic ML model")

判别式概率模型，Discriminative probabilistic ML model，用于分类和回归等任务。其特点是根据条件概率 $p(y|x,theta)$ 进行建模，而不是通过联合概率分布 $p(x,y)$. 即，根据 $x$ 预测 $y$。通常假设 $theta$ 的先验分布与 $x$ 无关，因此有：
$
p(y,theta|x) = p(y|x, theta) p(theta)
$
在这里，$p(y|x,theta)$ 是对与模型的选择，即函数 $y = f(x, theta)$.

生成式概率模型，Generative probabilistic ML model，则是可以根据联合概率分布 $p(x,y,theta)$ 进行建模，最终要获得的是 $p(x,y|theta)$，即
$
p(x,y,theta) = p(x,y|theta)p(theta)
$
贝叶斯模型，假设训练数据 $(X_(t r), Y_(t r))$ 和一个判别式模型 $p(y,theta|x)$，我们可以通过贝叶斯方法来估计参数 $theta$，在训练阶段，我们的 $theta$ 是由训练数据 $(X_(t r), Y_(t r))$ 估计得到的，即 $p(theta|X_(t r), Y_(t r))$. 根据贝叶斯定理：

$
p(theta|X_(t r), Y_(t r)) &= (p(X_(t r), Y_(t r),theta))/(p(X_(t r), Y_(t r)))\
&= (p(Y_(t r)|X_(t r),theta)p(X_(t r)|theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(X_(t r)|theta)p(theta) d theta)\
text("given: ") p(X_(t r)|theta) = P(X_(t r)) &=(p(Y_(t r)|X_(t r),theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(theta) d theta)
$ <BaysianBasic1>
通过训练，我们获得了后验分布 $p(theta|X_(t r), Y_(t r))$. 在测试阶段，加入新数据点 $x$，此时我们可以通过后验分布 $p(theta|X_(t r), Y_(t r))$ 来估计 $y$ 的概率分布：
$
p(y|x,X_(t r),Y_(t r)) = integral p(y|x,theta)p(theta|X_(t r),Y_(t r)) d theta
$ <BaysianBasic2>

这是对所有的模型 $theta$ 进行平均，其中 $p(y|x,theta)$ 代表每个模型（由 $theta$ 表示）的预测，而 $p(theta|X_(t r),Y_(t r))$ 代表这些模型的不确定性，衡量我们对不同参数的信心。

#hd4("Conjugate distribution")

在贝叶斯模型中， BaysianBasic1 和 BaysianBasic2 都存在积分计算，在大部分情况下是难以直接获得数值解的。但共轭分布（Conjugate distribution）可以简化这种计算。#index("Conjugate Distribution")

共轭分布是指：对于先验分布 $p(theta)$ 、似然函数 $p(X|theta)$和后验分布 $p(theta|X)$，若先验分布和后验分布属于同一分布族（distribution family），则称 $p(theta)$ 和 $p(X|theta)$ 为共轭分布。即：
$
p(theta) in cal(A)(alpha), p(X|theta) in cal(B)(beta) arrow.double p(theta|X) in cal(A)(alpha^prime)
$
这样的好处在于，我们可以直接获得后验分布 $p(theta|X)$ 的形式，从而可以忽略积分的过程，例如：
$
p(theta|X) = (p(theta) p(X|theta)) / (integral p(theta) p(X|theta) d theta)
$ <conjugate>
我们知道 $p(theta|X)$ 的函数形式是与 $p(theta)$ 相同的，即确保了 $integral p(theta|X) d theta = 1$. 因此我们可以忽略积分，得到：
$
p(theta|X) prop p(theta) p(X|theta)
$
接着只需要计算参数即可。

常见的共轭分布：

#set table(stroke: (x, y) => (
  bottom: if y == 0 {1pt},
  right: if x == 0 or x == 1 {1pt},
  ))
#align(center)[#figure(
  table(
    align: horizon + center,
    columns: (40%, 20%, 40%),
    table.header[Likelihood $p(x|theta)$][$theta$][Conjugate prior $p(y)$],
    [Gaussian], [$mu$], [Gaussian],
    [Gaussian], [$sigma^(-2)$], [Gamma],
    [Gaussian], [$(mu, sigma^(-2))$], [Gaussian-Gamma],
    [Multivariate Gaussian], [$Sigma^(-1)$], [Wishart],
    [Bernoulli], [$p$], [Beta],
    [Multinomial], [$(p_1,dots,p_m)$], [Dirichlet],
    [Poisson], [$lambda$], [Gamma],
    [Uniform], [$theta$], [Pareto]
  ),
)]

共轭分布通常只适用于简单概率模型。

#hd4("Maximum a posterior estimation") #index("Maximum a Posterior Estimation")

当共轭分布不可用时，一种简单的方法是使用最大后验估计（maximum a posteriori probability estimate, MAP）.其思想是将分布估计转变为点估计，将参数取为后验分布的最大值，即：
$
theta_(M A P) &= arg max_(theta) p(theta|X_(t r), Y_(t r))\
&= arg max_(theta) (p(Y_(t r)|X_(t r),theta)p(theta))/(integral p(Y_(t r)|X_(t r),theta)p(theta) d theta)\
&integral p(Y_(t r)|X_(t r),theta)p(theta) d theta text("does not depend on ") theta\
&= arg max_(theta) p(Y_(t r)|X_(t r),theta)p(theta)
$
鉴于 $theta_(M A P)$ 为点估计值，此时测试阶段则转变为：
$
p(y|x, X_(t r), Y_(t r)) = p(y|x, theta_(M A P))
$

#pagebreak()
#hd3("Variational inference")

#hd4("KL-divergence")

变分推断（Variational inference）是一种近似推断方法，旨在处理复杂的后验分布。其基本思想是，将后验分布 $p(theta|X)$ 近似为一个简单的分布 $q(theta) in cal(Q)$，使得 $q(theta)$ 尽可能接近 $p(theta|X)$. 通常情况下，我们可以通过最小化两个分布的KL-divergence来获得 $q(theta)$，即：

$
q(theta) &= arg min_(q in cal(Q)) F(q) := K L(q(theta)||p(theta|X))
$

其中，KL-divergence为： #index("Divergence Mesure", "KL-divergence")
$
K L(q(theta)||p(theta|X)) = integral q(theta) log(q(theta)/p(theta|X)) d theta
$
注意：
1. KL-divergence是非负的，当且仅当 $q(theta) eq.triple p(theta|X)$ 时，KL-divergence为0
2. KL-divergence不对称，即 $K L(q(theta)||p(theta|X)) eq.not K L(p(theta|X)||q(theta))$
3. KL-divergence包含的两个分布必须是相同的支持集，即 $q(theta)$ 和 $p(theta|X)$ 必须在#margin-note()[
  两者必须拥有相同的信息量
]相同的空间上定义
  #showybox()[
    支撑集(support)：它是集合$X$的一个子集，要求对给定的$X$上定义的实值函数$f$在这个子集上恰好非$0$. 特别地，在概率论中，一个概率分布是随机变量的所有可能值组成的集合的闭包。
  ]

存在两个问题：
1. 未知的后验分布 $p(theta|X)$ 导致 KL-divergence 无法计算
2. KL-divergence的优化空间为分布空间，通常情况下是无法直接优化的

#hd4("Evidence lower bound")

证据下界（Evidece lower bound, ELOB）用于解决问题1. 我们知道后验分布的贝叶斯形式为：
$
p(theta|X) = (p(X|theta) p(theta))/ p(X) = (p(X|theta) p(theta))/ (integral p(X|theta) p(theta) d theta) = (text("Likelihood") times text("Prior"))/text("Evidence")
$
考虑 $log p(X)$，我们可以做出以下变形：

$
log p(X) &= integral q(theta) log(p(X)) d theta = integral q(theta) log p(X,theta)/(p(theta|X)) d theta\
&= integral q(theta) log (p(X,theta)q(theta))/((p(theta|X))q(theta)) d theta\
&= integral q(theta) log p(X,theta)/q(theta) d theta + integral q(theta) log q(theta)/p(theta|X) d theta\
&= cal(L)(q(theta)) + K L(q(theta)||p(theta|X))
$ <BaysianVariation1>

其中，$cal(L)(q(theta))$ 为证据下界（Evidence lower bound, ELBO），即： #index("Evidence Lower Bound")
$
log p(X) >= cal(L)(q(theta))
$

对于 BaysianVariation1，我们注意到：$log p(x)$ 与 $q(theta)$ 无关，而 $cal(L)(p(theta))$ 和 $K L(q(theta)||p(theta|X))$ 与 $q(theta)$ 有关。因此，我们可以通过最大化 $cal(L)(q(theta))$ 来最小化 $K L(q(theta)||p(theta|X))$，即：
$
q(theta) &= arg min_(q in cal(Q))  K L(q(theta)||p(theta|X))\
&= arg max_(q in cal(Q)) cal(L)(q(theta))\
&= arg max_(q in cal(Q)) integral q(theta) log p(X,theta)/q(theta) d theta\
$
其中 $cal(L)(q(theta))$ 中分布都是已知可以计算的，进一步我们可以得到：
$
cal(L)(q(theta)) &= integral q(theta) log p(X,theta)/q(theta) d theta = integral q(theta) log (p(X|theta)p(theta))/q(theta) d theta\
& = integral q(theta) log p(X|theta) d theta + integral q(theta) log p(theta)/q(theta) d theta\
& = bb(E)_(q(theta)) log p(X|theta) - K L (q(theta)||p(theta))
$

对于第一项 $bb(E)_(q(theta)) log p(X|theta)$，我们需要其最大化。即使取 $log p(X|theta)$ 的加权平均值，其收敛点仍然与MLE一致：将 $q(theta)$ 设置在集中于MLE点估计的位置，有 $hat(theta) = arg max_theta p(X|theta)$，此时参数对应的似然函数 $p(X|theta)$ 取最大值，有：
$
bb(E)_(q(theta)) log p(X|theta) arrow log p(X|theta)
$
这告诉我们最大化第一项会导致参数 $theta$ 逐渐收敛到 $theta_(M L E)$. 同时这一项被叫作 data term，当模型的对数似然性很高时，意味着模型在给定的参数下生成观察数据 $X$ 的概率大于在其他参数下生成数据的概率。因此，可以说模型与观察数据的拟合程度较好

第二项被称为 regularizer，可以防止模型的过拟合。

#hd4("Mean field approximation")

对于 $q(theta)$ 的选择，我们可以使用均场近似（mean field approximation）来简化问题。均场近似假设 $q(theta)$ 可以分解为一系列独立的分布，即：  #index("Mean Field Approximation")
$
q(theta) = product_(i=1)^(m) q_i (theta_i)
$
均场近似的思想是在每一步中，固定参数 ${q_i (theta_i)}_(i eq.not j)$，只对单一参数 $q_j (theta_j)$ 做优化，其中参数 $theta_i$ 仍然可以为向量，因此我们的目标转化为：
$
q_j (theta_j) = arg max_(q_j (theta_j)) cal(L)(q(theta))
$
#pagebreak()
此时将固定参数视为常量，对 $q_j (theta_j)$ 做解析解，可以得到：
#set math.equation(number-align: top)
$
cal(L)(q(theta)) &= integral q(theta) log p(X,theta)/q(theta) d theta = integral q(theta) log  p(X, theta) d theta - integral q(theta) log q (theta) d theta\
&=integral product_i q_i (theta_i) log p(X, theta) product_i d theta_i - 
integral product_i q_i (theta_i) log product_i q_i (theta_i) product_i d theta_i\
&=integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) log p(X, theta) product_i d theta_i  - integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) log product_i q_i (theta_i) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) integral product_(i eq.not j) q_i (theta_i) (log q_j (theta_j)+log product_(i eq.not j) q_i (theta_i)) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) log q_j (theta_j)  underbrace(integral product_(i eq.not j) q_i (theta_i) product_i d theta_i, "=1") - underbrace(integral q_j (theta_j),"=1") integral product_(i eq.not j) q_i (theta_i) log product_(i eq.not j) q_i (theta_i) product_i d theta_i\
&=bb(E)_(q_j (theta_j)) [ bb(E)_(q_(i eq.not j)) log p(X, theta)] - integral q_j (theta_j) log q_j (theta_j) d theta_j - integral product_(i eq.not j) q_i (theta_i) sum_(i eq.not j) log q_i (theta_i) product_(i eq.not j) d theta_i\
&=bb(E)_(q_j (theta_j))[bb(E)_(q_(i eq.not j)) log p(X, theta) - log q_j (theta_j)] - text("constant")
$
引入Lagrangian 函数计算其最小值：
$
partial_(q_j (theta_j)) cal(L) (q(theta)) &+ sum_i lambda_i (integral p_i (theta_i) d theta_i - 1) = 0 \
0&= bb(E)_(q_(i eq.not j)) log p(X, theta) - partial_(q_j (theta_j)) bb(E)_(q_j (theta_j)) [log q_j (theta_j)] + partial_(q_j (theta_j)) lambda_j integral p_j (theta_j) d theta_j
$ <BaysianVariation2>
#set math.equation(number-align: horizon)
利用变分法解决求导：
$
partial_(q_j (theta_j)) bb(E)_(q_j (theta_j)) [log q_j (theta_j)] &= partial_(q_j (theta_j)) integral q_j (theta_j) log q_j (theta_j) d theta_j\
&= partial_(q_j (theta_j)) q_j (theta_j) log q_j (theta_j) \
&= log q_j (theta_j) + 1\
partial_(q_j (theta_j)) lambda_j integral p_j (theta_j) d theta_j &= lambda_j
$
因此， BaysianVariation2 改写为：
$
0&=bb(E)_(q_(i eq.not j)) log p(X, theta)- log q_j (theta_j) - 1 + lambda_j\
&=bb(E)_(q_(i eq.not j)) log p(X, theta) - log q_j (theta_j) + text("constant")\
$
得到：
$
q_j (theta_j) = 1/(Z_j) exp(bb(E)_(q_(i eq.not j)) log p(X, theta))
$
其中 $Z_j$ 为归一化常数，使得 $q_j (theta_j)$ 满足概率分布的性质。
#pagebreak()
因此， Mean field approximation 的算法为：
1. 初始化：
$
q(theta) = product_i q_i (theta_i)
$
2. 重复，直到ELBO收敛：
  - 对于每一个 $q_i (theta_i)$，做如下计算更新：
  $
  q_j (theta_j) = 1/(Z_j) exp(bb(E)_(q_(i eq.not j)) log p(X, theta))
  $
  或
  $
    log q_j(theta_j) = bb(E)_(q_(i eq.not j)) log p(X, theta) + text("constant")
  $
  - 重新计算 ELBO:
  $
  cal(L)(q(theta))
  $
问题在于如何计算 $Z_j$ 与期望 $bb(E)_(q_(i eq.not j)) log p(X, theta))$ 是否能够被解析解计算出来。如果要确保其能够被计算出来，需要假设 $theta arrow [theta_1,dots, theta_m]$ 的过程具有共轭性，即：
$
  forall theta_j,space.quad p(theta_j|theta_(i eq.not j)) in cal(A)(alpha_j), &space.quad p(x| theta_j, theta_(i eq.not j)) in cal(B)(beta_j) \
  &arrow p(theta_j|X, theta_(i eq.not j)) in cal(A)(alpha^prime)
$
在实际操作中，可以这样对共轭性进行检验：

对于每一个 $theta_j$:
- 固定 ${theta_i}_(i eq.not j)$ （将其视为常数）
- 检查 $p(X|theta)$ 与 $p(theta)$ 是否对于 $theta_j$ 是共轭的

#pagebreak()
#hd3("Dirichlet process mixture model")

#hd4("Dirichlet process") #index("Dirichlet Process")

Dirichlet process 是一种用于非参数贝叶斯统计的随机过程，可以创建无限个连续分布 $H$ 的离散副本，即 $H arrow.bar G$. 记作 $G tilde D P(alpha, H)$，其中 $alpha$ 为浓度参数，决定了聚类的程度。浓度参数越大，生成的簇的数量通常越多。

Dirichlet process 可以由 stick-breaking process 来描述。假设我们有一个长度为1的棍子，我们从棍子的一端开始，每次从棍子的长度中折断一部分，折断的长度服从beta分布，折断的位置服从beta分布。这样我们可以得到一个无限个分布的序列：

#index("Stick-breaking Process")

1. 生成一个无限长序列 $V_k tilde text("Beta")(1, alpha) in (0,1)$
2. 生成一个无限长序列权重 $pi_k$:
$
pi_k = V_k product_(j=1)^(k-1) (1-V_j), sum_(k=1)^infinity pi_k = 1
$
3. 从连续分布 $H$ 中抽取无限多个样本 $theta_k$，构成新的分布 $G$:
$
G = sum_(k=1)^infinity pi_k delta_(theta_k), delta_(theta_k) = delta(theta-theta_k)
$

Dirichlet process 有以下性质：

1. 期望不变：
$
bb(E)_(D P (alpha, H)) [x] = bb(E)_(H) [x]
$
2.
$
alpha arrow infinity arrow.double D P(alpha, H) = H
$
3. 序列为无限长，无法完全在计算机中表达

#hd4("Mixture model") #index("Mixture Model")

混合模型（Mixture Model）是一种统计模型，它假设数据来自多个不同的分布，每个分布被称为一个"成分"（component）。混合模型的主旨在于通过这些成分的组合来更好地描述数据的总体分布。每个成分可以用不同的概率分布函数来定义，如高斯分布、伯努利分布等。

对于给定的观察值 $x$ ，混合模型的概率密度函数可以表示为:
$
p(x) = sum_(k=1)^K pi_k p(x|theta_k), sum_(k=1)^K pi_k = 1
$

混合模型广泛应用于许多领域，包括但不限于：

- 聚类：通过对数据进行分组，帮助识别数据中的模式和结构。例如，K均值聚类可以看作是高斯混合模型的一个特例。
- 密度估计：能够捕捉到复杂的分布形状，比单一的概率分布（如正态分布）更具灵活性。
- 信号处理：在音频和图像处理等领域中，用于建模混叠信号。
- 生物信息学：在基因表达数据分析中识别不同的生物状态。

考虑利用核混合模型（Kernel Mixture Model）估计pdf，我们可以初步写为：
$
f(y|P) = integral cal(K)(y|theta) d P(theta)
$ <DPMM1>
其中，$cal(K)(dot|theta)$ 为核函数，以 $theta$ 作为其参数，可能包括每个核的中心位置、宽度等；$P$ 是混合测度。

#showybox[  #index("Measure")
  测度（measure）：测度是一个函数，它将集合映射到实数。常见的测度包括：长度、面积、概率测度等。假设样本空间 $Omega$，一个概率测度 $P$ 满足以下条件：
  - $P(A) >= 0, forall A in Omega$
  - $P(Omega) = 1$
  - 对于不相交事件 $A_1, A_2, dots$，有 $P(union.big_i A_i) = sum_i P(A_i) $

  混合测度（mixture measure）：混合测度是指在混合模型中，表示由多种不同的分布组合而成的分布特征。混合测度可以写作：
  $
  P = integral P_(theta) d H(theta)
  $ <DPMM2>
  其中：$P_theta$ 是给定参数的成分分布，例如可以是正态分布、指数分布等；$H$ 是先验测度，描述参数 $theta$ 的分布。

]

因此，DPMM1 计算了在参数空间中，对于每个 $theta$ 的核函数的加权求和。

#hd4("DP for mixutre model")

DP适合用作为未知混合分布的先验。例如在 DPMM1 中，考虑其为无限核混合模型，混合测度 $P$ 视为未知。此时，我们可以令 $P tilde pi_(cal(P))$，其中 $cal(P)$ 代表在样本空间中所有可能的概率测度，$pi_(cal(P))$ 则代表样本空间中的先验分布。考虑将 $pi_(cal(P))$ 选做 DP 的先验，这样我们可以获得一个离散的 DP 混合模型:
$
f(y) = sum_(k=1)^infinity pi_k cal(K)(y|theta_k)
$
其中 $pi_i$ 来自 DP 的权重（参数为 $alpha$），且：
$
y_i tilde cal(K)(theta_i), space.quad theta_i tilde P, space.quad P tilde D P(alpha, P_0)
$
采用DP作为 $P$ 的先验，会导致后验计算变得复杂。根据 DPMM2 我们可知，以 DP 作为先验的混合模型，其混合测度拥有无限个参数，因此在拥有样本 $y^n=(y_1,y_2,dots,y_n)$ 的条件下无法直接获得 $P$ 的后验分布。解决方法是通过边缘化 $P$ 来获得参数 $theta^n=(theta_1, theta_2,dots,theta_n)$ 的先验分布。具体的，我们可以用Polya urn 预测规则来描述这种情形。即：
$
p(theta_i|theta_1,dots,theta_(i-1)) tilde (alpha / (alpha+i-1)) P_0(theta_i) + sum_(j=1)^(i-1) delta_(theta_j)
$ <DPMM3>

从 DPMM3 开始进行聚类，假设有 $n$ 个样本，$k$ 个簇，$n_k$ 代表第 $k$ 个簇的样本数量，$theta_k$ 代表第 $k$ 个簇的参数，我们有：

$
p(theta_i|theta_(-i)) = 
underbrace(
  (alpha / (alpha+n-1)) P_0(theta_i),
  "新建聚类"
) + 
underbrace(
  sum_(h=1)^(k^((-i))) 
  overbrace(
    ( n_h^((-i))/(alpha+n-1)),
    "DP中聚类h的权重/占比")
    delta_(theta_h^(*(-i))),
  "选择已有聚类"
)
$ <DPMM4>
其中 $theta_h^*, h=1,dots,k^((-i))$ 是 $theta_(-i)$ 中的唯一值，代表了在移除第 $i$ 个样本后剩下的唯一聚类参数；$n_h^((-i)) = sum_(j eq.not i) 1_(theta_j = theta_h^*)$ 代表除去第 $i$ 个样本后，第 $h$ 个簇的样本数量。

#hd4("DPMM with Gibbs sampling")
Gibbs sampling 允许我们从多维分布中抽样，通过迭代更新每个维度的样本。对于 DPMM，我们可以通过 Gibbs sampling 来优化 DPMM4:

令 $theta^* = (theta_1^*, dots, theta_k^*)$ 为参数 $theta$ 的唯一值，且令 $S_i$ 为第 $i$ 个样本的聚类分配，即若 $theta_i = theta_c^*$ 则 $S_i = c$. Gibbs sampler 的步骤如下：
#set math.cases(gap: 1em)
1. 通过从多项式条件后验中抽样来更新分配 $S$:
$
P(S_i = c|S_(i-1),theta^*,alpha,P_0) prop cases(
  &n_c^((-i)) cal(K)(y_i|theta_c^*)\, &c=1\,dots\,k^((-i)),

  &alpha integral cal(K)(y_i|theta) d P_0(theta)\, space.quad &c=k^((-i))+1
)
$
2. 通过从条件后验中抽样来更新参数 $theta^*$:
$
p(theta_c^*|-) prop P_0(theta_c^*) product_(i:S_i=c) cal(K)(y_i|theta_c^*)
$
其中 $product_(i:S_i=c) cal(K)(y_i|theta_c^*)$ 是聚类 $c$ 中所有样本在参数 $theta_c^*$ 下的似然函数的乘积，反应了在当前聚类分配下，样本数据对于参数的支持程度。

聚类行为的控制因素：

1. 浓度参数 $alpha$：$alpha$ 的大小直接影响聚类的数量。当 $alpha$ 接近于0时，获取的聚类会趋向于集中，从而展现出一种共同参数 $y_i tilde cal(K)(theta)$ 的行为。相反，增大 $alpha$ 则有更高的可能性去接受新聚类。
2. 先验 $P_0$ 的方差：高方差的先验 $P_0$ 表示对聚类位置的不确定性，阻碍新聚类的形成。

#pagebreak()
#hd3("Deep Gaussian Process")

#hd4("Bayesian Regression")

普通线性归回模型中，我们假设数据集 $cal(D) = {x_i, y_i}_(i=1)^N, x_i in bb(R)^d$ 的 $x_i$ 与 $y_i$ 存在线性关系，即： #index("Linear Regression")
$
  y = w^tack.b x + epsilon
$
其中 $w in bb(R)^d$ 为权重，$epsilon$ 为噪声，通常假设 $epsilon tilde cal(N)(0, sigma^2)$. 因此：
$
  p(y|w, X, sigma^2) = cal(N)(y|X w, sigma^2 I)
$ <linear-regression-model>

linear-regression-model 这种模型的缺点在于缺少对于非线性关系的拟合能力，因此，我们可以引入基函数 $phi(x)$，将输入特征映射到更高维的空间：
$
  phi(x) = mat(phi_1 (x), phi_2 (x), dots.c, phi_M (x))
$
其中，$phi_i (dot)$ 通常为非线性函数，例如径向基函数等。此时，我们的模型变为：
$
  y = w^tack.b phi(x) + epsilon
$
因此：
$
  p(y|w, X, sigma^2) = cal(N)(y|Phi w, sigma^2 I)
$ <nonlinear-regression-model>  #index("Nonlinear Regression")
其中：
$
  Phi := Phi(x) = mat(
    phi_1 (x_1), dots.c, phi_M (x_1);
    dots.v, dots.down, dots.v;
    phi_1 (x_N), dots.c, phi_M (x_N)
  )
$
nonlinear-regression-model 这种模型的缺点在于没有对权重的不确定性进行建模。而对权重的不确定性进行建模，引出了贝叶斯回归（Bayesian Regression）模型。我们给予权重一个先验分布：
$
  w tilde cal(N)(0, S)
$
其中，$w in bb(R)^M$，$S in bb(R)^(M times M)$ 为协方差矩阵。此时，$Phi w$ 的分布为：
$
  p(Phi w) &= cal(N)(bb(E)[Phi w], "Cov"(Phi w))\
  &= cal(N)(0, Phi S Phi^tack.b) = cal(N) (0, K)
$ <bayesian-phi-w>
其中 $Phi in bb(R)^(N times M), K = Phi S Phi^tack.b$ 为协方差矩阵。因此，我们可以得到权重的后验分布：
$
  p(w|X, y) = cal(N)(w|mu_w, Sigma_w)\
  Sigma_w = (sigma^(-2) Phi^tack.b Phi + S^(-1))^(-1), space.quad mu_w = sigma^(-2) Sigma_N Phi^tack.b y
$ <bayesian-nonlinear-regression> #index("Bayesian Nonlinear Regression")

bayesian-nonlinear-regression 这种模型的缺点在于需要手动选择基函数的类型与基函数的数量，数量较少会导致无法充分捕捉数据的非线性特征，而基函数数量过多则容易导致过拟合，并增加计算复杂度。

但是，我们观察到 bayesian-phi-w 中 $K = Phi S Phi^tack.b$ 等价于 $K(dot, dot)$ 的内积计算方式，因此我们可以引入一个函数 $K(dot, dot)$ 来代替显式构造 $phi(x)$，这就是高斯过程以及高斯过程回归的思想。

#hd4("Gaussian Process")

Gaussian distribution 具有良好的性质，对于多元高斯分布，其联合分布、条件分布、边缘分布都是高斯分布。例如，假设多元高斯分布：
$
  p(f_1, f_2) tilde cal(N)(f_1,f_2|mu, Sigma)
$
则：
$
  mat(delim: "(", f_1;f_2) tilde cal(N)(mat(mu_1;mu_2), mat(Sigma_(11), Sigma_(12);Sigma_(21),Sigma_(22)))\
  p(f_1) tilde cal(N)(f_1|mu_1, Sigma_(11))\
  p(f_1|f_2) tilde cal(N)(f_1|mu_1+Sigma_(12)Sigma_(22)^(-1)(f_2-mu_2), Sigma_(11)-Sigma_(12)Sigma_(22)^(-1)Sigma_(21))
$ <multivariate-g>  #index("Multivariate Gaussian")

Gaussian process (GP) 假设我们观测的样本来自一个连续随机过程。对于每一个观测的样本与其对应的输出，我们假设输入 $X = {x}_(i=1)^m, x_i in bb(R)^d$，对应观测值 $f = {f}_(i=1)^m$，且满足：
$
  p(f|X) = cal(N)(mu, K)
$ #index("Gaussian Process")
其中 $f_i$ 为随机函数，其值为 $f(x_i)$，是我们观察到的、来自随机过程的值。$mu$ 为均值函数，$K$ 为协方差函数，因此还有：
$
  mu = {mu_i} = {mu(x_i)}
$
对于协方差函数 $K$，其本质是量化两个点之间距离的函数，例如核函数中的高斯核。在GP过程中有多种选择，例如：  #index("Covariance Function")
$
  K &= {K_(i,j)} = {K(x_i, x_j)}\
  &= 2/pi sin^(-1)(2 (x_i^tack.b Sigma x_j)/(sqrt((1+2 x_i^tack.b Sigma x_i)(1+2 x_j^tack.b Sigma x_j))))
$
初次以外，协方差函数 $K$ 还可以组合使用：
- 相加：$K(x, x') = K_1(x, x') + K_2(x,x')$
- 相乘：$K(x, x') = K_1(x, x') dot K_2(x,x')$
- 卷积：$K(x, x') = integral K_1(x, z)K_2(z,x') d z$
此时，记作：
#let GP = $cal(G P)$
$
  f(x) tilde GP(dot|mu(x), K(x,x'))
$

#hd4("GP Regression")

假设观测到数据点 $S_m = {X, y } = {(x_i, y_i)}_(i=1)^m$，$y$ 为噪音观测值：
$
  y_i = f(x_i) + epsilon_i
$
其中 $f(dot) tilde GP(dot|0,K)$ 为我们希望拟合的随机过程函数，$epsilon tilde cal(N)(0,sigma^2)$ 为高斯噪声。因此，先验为：
$
  p(f) tilde cal(N)(0,K)
$
根据我们的假设，$y$ 服从多元正态分布：
$
  p(y) tilde cal(N)(0, K + sigma^2 I)
$
因此，似然函数为：
$
  p(y|f) = cal(N)(y|f, sigma^2 I)
$
假设测试输入值 $x_*$，我们希望预测其输出值 $y_*$，则：
$
  y_* = f_* + epsilon_*, space.quad f_* = f(x_*)
$
我们需要根据观测的 $y$ 获得估计的 $f_*$，因此考虑此时联合分布，根据 multivariate-g：
$
  p(y, f_*) tilde cal(N)(mat(0;0), mat(K+sigma^2 I, K_*; K_*^tack.b, K_(**)))
$
其中 $K_* = {K(x_*, x)}_(i=1)^m, K_(**) = K(x_*, x_*)$. 同理，根据 multivariate-g，我们可以得到 $f_*$ 的条件分布，这就是我们的 GP Regression：
$
  p(f_*|y) = cal(N)(f_*|mu_*, Sigma_*)\
  mu_* = K_*^tack.b (K+sigma^2 I)^(-1)y, space Sigma_* = K_(**) - K_*^tack.b (K+sigma^2 I)^(-1)K_*
$ #index("Gaussian Process Regression")
其中对于 $mu_*$，有 $K_* in bb(R)^m, (K+sigma^2 I)^(-1) in bb(R)^(m times m), y in bb(R)^m$，因此 $mu_* in bb(R)$. 同理，$Sigma_* in bb(R)$. 因此，$mu_*$ 可改写为：
$
  mu_* = sum_(i=1)^m alpha_i K(x_*, x_i), space.quad alpha = (K+sigma^2 I)^(-1)y
$
GP Regression 的超参数在于协方差函数 $K$ 的选择，以及噪声方差 $sigma^2$ 的选择。通常我们可以通过最大化似然函数来估计这些参数。考虑：
$
  p(y) = cal(N)(0, K+sigma^2 I_m)
$
对数似然函数为：
$
  cal(L) &= log p(y) = -1/2 y^tack.b (K+sigma^2 I_m)^(-1)y - 1/2 log det(K+sigma^2 I_m) - m/2 log 2pi\
  &= log p(y|theta) = -1/2 y^tack.b C^(-1)(theta) y - 1/2 log det(C(theta)) - m/2 log 2pi
$
其中 $C(theta) = K+sigma^2 I_m$

但使用GP同样会有以下问题：
1. 需要选择合适的协方差函数 $K$
2. 协方差矩阵 $K in bb(R)^(m times m)$，其中 $m$ 是数据集大小，且需要计算其逆矩阵，计算复杂度为 $O(m^3)$. 因此，协方差矩阵的空间复杂度与时间复杂度都很高
3. 计算 $p(y|theta)$ 时，需要保证高斯分布与噪声的分布为共轭，否则无法直接计算

#hd4("Deep Gaussian Process") #index("Deep Gaussian Process")

DGP 是一种多层次的高斯过程，可以用于建模复杂的非线性关系。DGP 由多个 GP 组成，每个 GP 代表一个层次。每个 GP 的输出作为下一层的输入，因此 DGP 可以看作是多个 GP 的堆叠。

对于观测数据 $cal(D) = {(x_i, y_i)}_(i=1)^N$，假设 $L$ 层的DGP，则第 $l, l=1,dots,L$ 层为：
$
  f^((l))|f^((l-1)) &= GP(mu^((l))(f^((l-1))), K^((l))(f^((l-1)), f^((l-1))))\
  &= GP(dot|f^((l-1)), theta^((l)))
$
而 $f^((0)) = X$，输出时：
$
  y = f^((L)) + epsilon, space.quad epsilon tilde cal(N)(0, sigma^2 I)
$
观测数据的边际似然为：
$
  p(Y|X, theta) = integral p(Y|F, theta) p(F|X, theta) d F\
  F = {f^((l))}_(l=1)^L, theta = {theta^((l))}_(l=1)^L
$
难以直接计算，可以考虑使用 gaussian posterior approximation： #index("Gaussian Posterior Approximation")
$
  p(f|X,y,theta) approx q(f)
$

#pagebreak()
#hd3("Bayesian optimization") #index("Bayesian Optimization")

有些时候，我们不仅需要根据数据预计结果，还期望获得预计结果的不确定性。

在回归中，我们的目标是基于来自未知函数的观察数据点对函数进行建模。传统的非线性回归方法通常给出一个被认为最适合数据集的单一函数。然而，可能存在多个函数同样适合观察到的数据点。我们观察到，当多元正态分布的维度为无限时，我们可以使用这些无限数量的函数在任何点进行预测。当我们开始有观察数据时，我们不再保留无限数量的函数，而只保留适合观察数据的函数，形成后验分布。
当我们有新的观察数据时，我们将当前的后验作为先验，并使用新的观察数据点来获得一个新的后验。wang2023intuitive

Bayesian optimization 是一种用于优化黑盒函数的方法，即无法直接观察其函数形式，只能通过输入输出的数据点来估计其函数形式。

考虑 $f: cal(X) arrow bb(R), cal(X) subset.eq bb(R)^d$，我们的任务是找到：
$
  x = arg min_(x in cal(X)) f(x)
$
其中，$f(x)$ 通常是难以计算的、计算成本较高的或是无法计算导数。例如，在神经网络优化中，$x$ 通常指的是神经网络的超参数，例如学习率、网络结构等。

Bayesian optimization 首先对目标函数 $f(dot)$ 进行先验假设；根据观测数据获得后验分布；根据后验分布计算采集函数 (acquisition function)，选择下一个采样点；重复上述步骤直到收敛。 #index("Acquisition Function")

通常来说，假设 $f(dot) tilde GP(dot|mu(x), K(x,x'))$，并利用已知形式的后验 $mu_*$ 与 $sigma_*^2$ 计算采集函数。 采集函数平衡以下两个因素：
1. 探索（Exploration） ：在未充分了解的（方差大的）区域内尝试新的或未观察过的选项，以获取更多的信息。这种策略的目的是为了发现新的潜在的更优解。
2. 利用（Exploitation） ：指的是在已知的表现较好的（均值低的）区域内进行更深入的尝试，以最大化当前已知的收益。利用的目标是充分利用已有的信息，以获得最优结果。
因此，BO将问题转化为：
$
  x_(t+1) = arg max_(x in cal(X)) alpha(x|S_t)
$
例如，直接对于 $mu_*$ 与 $sigma_*^2$ 进行平衡的 GP lower confidence bound：
$
  alpha_("LCB")(x) = -mu_*(x) + kappa sigma_*^2(x)
$ #index("Gaussian Process Lower Confidence Bound")
其中 $kappa$ 为超参数，LCB可以看作是 $mu_*$ 的置信区间。

除此以外，我们可以使用Expected Improvement (EI) 。考虑新采样点可以提升 $f$ 表现力（即降低 $f$ 值）的概率，令 $f_("best") = min_i f_i$ 为观测值中的最小值，我们可以定义 reward function: #index("Expected Improvement")
$
  u(x) = max(0, f_("best") - f(x))
$ #index("Reward Function")
因此，采集函数为：
$
  alpha_("EI")(x) &= bb(E)[u(x)]\
  &= integral_(-infinity)^(+infinity) max(0, f_("best") - f(x)) p(f(x)|x) d f(x)\
  &= integral_(-infinity)^(f_("best")) (f_("best") - f(x)) cal(N)(f(x)|mu_*(x), sigma_*^2(x)) d f(x)\
  &= Delta(x) Phi(Delta(x)/(sigma_*(x))) + sigma_*(x) phi(Delta(x)/(sigma_*(x)))
$
除此以外，还有其他方法：
$
  gamma(x) &= (mu(x) - f_("best"))/sigma(x)\
  alpha_("MPI")(x) &= P(f(x)<f_("best")) =  Phi(gamma(x))\
$

#pagebreak()
