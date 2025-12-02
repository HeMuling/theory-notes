#import "../prelude.typ": *

#hd2("Conditional Probabilistic Graphical Model") #index("Conditional Probabilistic Graphical Model")

#hd3("Graphical Model")

#hd4("图")

图 (Graph)：  #index("Graph")

图由节点 (Node) 和边 (Edge) 组成，一个图包含两部分：
1. 节点集合 $V$，节点 $v in V$，通常代表一个实体 (entity) 、变量或状态
2. 边集合 $E$，边 $e in E$，通常代表节点之间的关系

无向图 (Undirected Graph)： #index("Graph", "Undirected Graph")

图中的边没有方向，即 $(v,w) = (w,v)$. 以马尔可夫随机场 (Markov Random Field, MRF) 为例，图中节点代表随机变量，边表示随机变量直接的依赖关系

有向图 (Directed Graph)： #index("Graph", "Directed Graph")

图中的边有方向，即 $(v,w) eq.not (w,v)$. 以贝叶斯网络 (Bayesian Network) 为例，有向边 $(v,w)$ 表示 $v$ 是 $w$ 的父节点 (parent)，即 $v$ 对 $w$ 有直接的因果影响或概率上的依赖。因此：
$
  P(W) = product_(w in W) P(w|"Parents"(w))
$

#hd4("Graphical Model") #index("Graphical Model")

图模型是一个概率分布（probability distributions） 家族，这些概率分布能够根据一个底层图（underlying graph）进行因子分解（factorize）。因子分解通过将复杂的全局分布分解为一系列局部函数（local functions）的乘积。每个局部函数仅仅依赖于变量集合中的一小部分变量（这些变量子集通常对应于图中的某些结构，如边或团）：
$
  p(x,y) = 1/Z product_A Psi_A (x_A, y_A)
$ <graphical-model>
其中：$Psi_A$ 为概率分布经过因式分解后获得的因子 (factor)，也叫势函数 (potential function)、局部函数 (local function)或兼容性函数 (compatibility function). $Psi_A$ 定义在变量子集 $A$ 上（即 $A subset X$），$x_A$ 代表变量子集 $A$ 的赋值（映射方式）。$Z$ 为归一化函数：
$
  Z = sum_(x,y) product_A Psi_A (x_A, y_A)
$

#figure(
  image("../assets/GraphicalModel.png", width: 70%)
) <graphical-model-img>

如 graphical-model-img 右所示，因子图模型（一种特殊的无向图模型）的随机变量节点被边连接，这些边的具体关系又由 $Psi_A$ 决定。

局部函数通常可以定义为：
$
  Psi_A (x_A, y_B) = exp{sum_k theta_(A k) f_(A k) (x_A, y_B)}
$
其中 
#margin-note($f_(A k)$)[
  exp的形式\
  方便计算log p
] 为特征函数 (feature function)，$theta_(A k)$ 为特征函数的权重。

有向图模型（也叫生成式模型）如 graphical-model 左所示，随机变量节点被有向边连接。给定图 $G = (V,E)$，有向图模型可以表示为：
$
  p(y, x) = product_(v in V) p(v|pi(v))
$
其中 $pi(v)$ 为 $v$ 的父节点。

#hd3("Conditional Probabilistic Graphical Model") #index("Conditional Probabilistic Graphical Model") 

#hd4("Linear CRF")

线性条件随机场 (Linear Chain Conditional Random Field, Linear CRF) 是条件随机场 (Conditional Random Field, CRF) 的一种特殊形式，用于处理序列数据。

处理序列数据的时，通常会用到隐马尔可夫模型 (Hidden Markov Model, HMM). #index("HMM")

对于观测序列 $X = {x}_(t=1)^T$，隐状态序列 $Y = {y}_(t=1)^T$，HMM 的联合概率可以表示为：
$
  p(y,x) = product_(t=1)^T p(y_t|y_(t-1))p(x_t|y_t)
$ <HHM>

对 HHM 进行因子分解，两边取对数：
$
  log p(y,x) = sum_(t=1)^T (log p(y_t|y_(t-1)) + log p(x_t|y_t))
$
对于转移项：$lambda_(i j) = log p(y_t=i|y_(t-1)=j) $，因此
$
  log p(y_t|y_(t-1)) = sum_(i,j in S) lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j)
$
对于发射项：$mu_(o i) = log p(x_t=o|y_t=i)$，因此，若对于时间 $t$，状态 $y_t=i$，观测 $x_t=o$，有
$
  log p(x_t|y_t) = sum_(o in O) sum_(i in S) mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i)
$
因此，可以得到：
$
  log p(y,x) = sum_(t=1)^T (sum_(i,j in S) lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j) + sum_(o in O) sum_(i in S) mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i))\
  p(y,x) = exp{sum_(t=1)^T (sum_(i,j in S) lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j) + sum_(o in O) sum_(i in S) mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i))}
$
考虑到我们引入的参数 $lambda_(i j)$ 和 $mu_(o i)$ 并非严格的$log$概率，因此需要引入归一化常数：
$
  p(y,x) = 1/Z exp{sum_(t=1)^T (sum_(i,j in S) lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j) + sum_(o in O) sum_(i in S) mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i))}\
  Z = sum_(y,x) exp{sum_(t=1)^T (sum_(i,j in S) lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j) + sum_(o in O) sum_(i in S) mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i))}
$
为了转化到 graphical-model，我们令：
$
  f_(i j) (y_t, y_(t-1), x_t) &= lambda_(i j) bb(I)(y_t=i) bb(I)(y_(t-1)=j)\
  f_(o i) (x_t, y_t) &= mu_(o i) bb(I)(x_t=o) bb(I)(y_t=i)
$
其中，状态数为 $N=abs(S)$，观测数为 $M=abs(O)$；共有 $N^2$ 个可能的转移特征，$N M$ 个可能的状态-观测特征；因此，共有 $N^2 + N M$ 个
#margin-note([总特征数量])[
  sum 带来的遍历要求衡量所有可能性
]。

可以创建索引 $k$，使其唯一地对应一个 $(i,j)$ 的转移特征或 $(o,i)$ 的观察特征。因此，可以改写为：
$
  p(y,z) = 1/Z exp{sum_(k=1)^K lambda_k f_k (y_t, y_(t-1), x_t)}
$ <linear-crf-HMM>
由于特征索引 $k$ 已经包含了转移特征 $(i,j)$ 和观察特征 $(o,i)$ 的信息，因此我们可以将时序 $t$ 的显式表示省略

根据 linear-crf-HMM，我们可以得到：
$
  p(y|x) = (p(y,x))/(sum_y' p(y'|x)) = (exp{sum_(k=1)^K lambda_k f_k (y_t, y_(t-1), x_t)})/(sum_y' exp{sum_(k=1)^K lambda_k f_k (y'_t, y'_(t-1), x_t)})
$ <linear-crf>
linear-crf 即为线性条件随机场，严格定义为：
$
  p(y|x) = 1/Z(x) exp{sum_(k=1)^K lambda_k f_k (y_t, y_(t-1), x_t)}
$
其中 $Y,X$ 为随机变量向量，$f_k$ 为特征函数，$lambda_k$ 为特征函数的权重；$Z(x)$ 为归一化常数：
$
  Z(x) = sum_(y in Y) exp{sum_(k=1)^K lambda_k f_k (y_t, y_(t-1), x_t)}
$

#pagebreak()
