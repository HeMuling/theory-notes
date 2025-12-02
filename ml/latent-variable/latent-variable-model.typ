#import "../../prelude.typ": *

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

