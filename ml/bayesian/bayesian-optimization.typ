#import "../../prelude.typ": *

#let GP = $cal(G P)$

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
