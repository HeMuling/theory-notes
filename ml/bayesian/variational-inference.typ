#import "../../prelude.typ": *

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
