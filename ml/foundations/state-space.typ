#import "../../prelude.typ": *

#hd3("State-Space model") #index("State-Space model")

State-Space model (SSM) 是用于描述时间序列数据的模型。对于任意时间序列输入 $u(t)$，SSM首先将其映射到 hidden space $x(t)$，然后进一步映射为输出空间 $y(t)$：
$
  u(t) arrow.bar x(t) arrow.bar y(t)
$
SSM以以下形式表示：
$
  x'(t) = A x(t) + B u(t)\
  y(t) = C x(t) + D u(t)
$
解为：
$
  y(t) = sum_(n=0)^t (C A^(t - n) B + D delta(t-n))u(n)
$ <SSM-soultion>
其中，$delta(t-n)$ 为Kronecker delta函数。

