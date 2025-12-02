#import "../prelude.typ": *

#hd2("贝叶斯")

贝叶斯涉及以下组件：

似然函数（likelihoo）：表示观测数据在参数 $theta$ 给定情况下的概率，通常记作 $p(D|theta)$，其中 $D$ 为观测数据

先验分布（prior distribution）：表示在没有观测数据时对参数 $theta$ 的信念，记作 $p(theta)$.

后验分布（posterior distribution）：表示在观测数据更新后参数分布，记作 $p(theta|D)$，通常由贝叶斯定理进行计算

#include "bayesian/basics.typ"
#include "bayesian/variational-inference.typ"
#include "bayesian/dpmm.typ"
#include "bayesian/dgp.typ"
#include "bayesian/bayesian-optimization.typ"
