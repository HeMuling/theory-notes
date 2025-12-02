#import "../../prelude.typ": *

#hd3("Basics")

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
