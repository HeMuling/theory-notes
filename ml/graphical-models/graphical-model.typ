#import "../../prelude.typ": *

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
  image("../../assets/GraphicalModel.png", width: 70%)
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
