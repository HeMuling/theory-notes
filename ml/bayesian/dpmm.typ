#import "../../prelude.typ": *

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
