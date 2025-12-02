#import "../../prelude.typ": *

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
