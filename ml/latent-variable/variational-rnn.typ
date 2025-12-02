#import "../../prelude.typ": *

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
  image("../../assets/VariationalRNN.png")
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
#showybox()[
  $
    "KL" (cal(N)(mu, Sigma)||cal(N)(0, I)) = bb(E)_p [log (p(Z)/q(Z))] = bb(E)_p [log (q(Z)/p(Z))]\
    = integral q(Z) log (q(Z)/p(Z)) d Z
  $
  记 $Z = (Z_(1:k)), z_i tilde cal(N)(z|mu, sigma) => z_i^2/sigma_i^2 tilde chi_k^2$，此时有：
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
