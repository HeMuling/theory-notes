#import "../../prelude.typ": *

#hd3("Noise-contrastive estimation")

#link("https://proceedings.mlr.press/v9/gutmann10a")

#hd4("数学推导")

假设观察到数据的分布（概率密度函数pdf）为 $p_d (dot)$，考虑pdf由参数 $bold(alpha)$ 决定，因此可以认为pdf属于参数家族 ${p_m (dot;bold(alpha))}_bold(alpha)$.其中 $bold(alpha)$ 为参数向量，且存在某个 $bold(alpha)^*$ 使得 $p_d (dot) = p_m (dot;bold(alpha)^*)$。

这里的问题是，如何在观察数据的基础上，通过最大化目标函数去估计参数 $bold(alpha)^*$. 我们知道的是，不管参数估计结果如何，都一定满足：

$
integral p_m (bold(u);hat(bold(alpha))) d bold(u) = 1
$ <NCE1>

为了绕过这个限制，并且保证为一，我们可以考虑#margin-note[使用归一化避免积分限制]使用归一化的方法，即：

$
p_m (dot;bold(alpha)) = (p_m^0 (dot;bold(alpha)))/(Z(alpha)) d bold(u), Z(bold(alpha)) = integral p_m^0 (bold(u);bold(alpha)) d bold(u)
$

这里，$p_m^0 (dot;bold(alpha))$ 不一定要满足 NCE1 的限制，可以是一个形式类似pdf的函数。但是，#margin-note[要尽量避免积分，积分计算困难]归一化系数 $Z(bold(alpha))$ 包括积分，通常是难以计算的。为了避免直接计算 $Z(bold(alpha))$ 的积分，我们可以将其考虑为一个新参数，通过优化的方法估计 $Z(bold(alpha))$.即：

$
p_m (dot;bold(theta)) = (p_m^0 (dot;bold(alpha)))/(C),theta = {bold(alpha), c}, c = Z(bold(alpha))
$

问题出现在使用最大似然估计MLE时：

$
hat(bold(theta))&= arg max_(bold(theta)) sum_(bold(u)) log p_m (bold(u);bold(theta))\
&= arg max_(bold(theta)) sum_(bold(u)) log (p_m^0 (bold(u);bold(alpha)))/(c)\
&arrow.double c arrow 0
$

会导致 $c$ 趋近于无穷小，这样的结果显然无效。

为了解决这样的问题，作者提出了一种新的估计方法，即Noise-contrastive estimation。其基本思想是，引入噪声分布 $p_n (dot)$，通过比较 $p_m (dot;bold(alpha))$ 和 $p_n (dot)$ 的相似性来估计参数 $bold(alpha)$.

从本质上来说，区分噪声与观测数据属于二分类问题。而二分类问题我们可以使用logistic回归解决，即：Logistic回归通过使用sigmoid函数将线性组合的输入映射到0到1之间的输出，以估计某个事件发生的概率。其模型形式为：

$
p(C=1|bold(u)) &= sigma(bold(u)^tack.b bold(theta)) = 1/(1+exp(-bold(u)^tack.b bold(theta)))\
p(C=0|bold(u)) &= 1 - p(C=1|bold(u)) = 1 - sigma(bold(u)^tack.b bold(theta))
$

其中，$C=1$ 代表为观测数据，$C=0$ 代表为噪声数据。

当进行回归时，我们的目标函数为：

$
L(bold(theta)) &= sum_(t=1)^T log p(C_t|bold(u)_t;bold(theta))\
&= sum_(t=1)^T C_t log p(C=1|bold(u)_t) +(1-C_t) log p(C=0|bold(u)_t)
$ <NCE2>

对于观测的数据：$X={bold(x)_1,bold(x)_2,dots,bold(x)_T}$，我们引入相同大小的噪声数据 $Y={bold(y)_1,bold(y)_2,dots,bold(y)_T}$. 根据logistic回归，我们令 $U=X union Y = {bold(u)_1, bold(u)_2, dots, bold(u)_(2T)}$.对于每一个 $bold(u)_t arrow.bar C_t = {0,1}$，当 $bold(u)_t$ 为观测数据时，$C_t=1$；当 $bold(u)_t$ 为噪声数据时，$C_t=0$.

#margin-note("直接类比logistic回归的结果即可") $p_m (bold(u);bold(theta))$，即通过优化获得参数 $bold(theta)$.因此，现在需要做的是获得目标函数 $L(bold(theta))$，且根据logistic回归的 NCE2，我们需要求出 $p(C=1|bold(u);bold(theta))$

我们知道，观测数据的pdf为等价于 $C_t=1$ 时对应的条件分布；噪声数据的pdf为等价于 $C_t=0$ 时对应的条件分布：
$
p_m (bold(u);bold(theta)) = P(bold(u)|C=1;bold(theta)), p_n (bold(u);bold(theta)) = P(bold(u)|C=0;bold(theta))
$

我们同时知道，$C$ 的分布为bernoulli分布，有 $P(C=1)=P(C=0)=1\/2$.因此，我们可以使用贝叶斯来计算：
$
P(C=1|bold(u);bold(theta)) &= (P(C=1 union bold(u);bold(theta)))/(P(bold(u);bold(theta)))\
&= (P(bold(u)|C=1;bold(theta)) dot P(C=1;bold(theta))) / (P(bold(u)|C=1;bold(theta)) dot P(C=1;bold(theta)) + P(bold(u)|C=0;bold(theta)) dot P(C=0;bold(theta)) )\
&= (p_m (bold(u);bold(theta)))/(p_m (bold(u);bold(theta)) + p_n (bold(u);bold(theta)))\
&=h(bold(u);bold(theta))\
P(C=0|bold(u);bold(theta)) &= 1 - h(bold(u);bold(theta))
$
代入 NCE2，我们可以得到
$
L(bold(theta)) = sum_(t) C_t log h(bold(u)_t;bold(theta)) +(1-C_t) log(1-h(bold(u)_t;bold(theta)))
$
考虑到 $bold(u)_t arrow.bar C_t = {0,1}$，我们可以化简上述公式为：
$
L(bold(theta)) = sum_t log h(bold(x)_t;bold(theta)) + log(1-h(bold(y)_t;bold(theta)))
$
我们有 $2T$ 个样本，#margin-note[
  解决：
  1. 损失值不可比\
  2. 梯度下降快\
  3. 数值不稳定
  ]因此对目标函数平均化：
$
L(bold(theta)) = 1/(2T)sum_t log h(bold(x)_t;bold(theta)) + log(1-h(bold(y)_t;bold(theta)))
$
