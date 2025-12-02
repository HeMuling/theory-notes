#import "../../prelude.typ": *

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
