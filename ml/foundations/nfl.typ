#import "../../prelude.typ": *

#hd3("NFL定理")#index([NFL Theorem])

归纳偏好用于描述当特征相同时，哪些特征更为重要

假设样本空间 $cal(X)$ 和假设空间 $cal(H)$ 为离散。令 $P(h|X, xi_a)$ 代表算法 $xi_a$ 基于训练数据 $X$ 产生假设 $h$ 的概率；令 $f$ 代表希望学习的目标函数。因此，算法在训练集外产生的误差为：

$
E_(o t e) (xi_a|X,f) = sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)
$



其中 $bb(I)(dot)$ 为指示函数，当 $dot$ 为真时返回 1，否则返回 0。

若学习目标为二分类，则 $cal(X) arrow.bar {0,1}$ 且函数空间为 ${0, 1}^(|cal(X)|)$，其中 $|dot|$ 用于计算集合长度。

算法用于解决多个任务，则拥有多个学习的目标函数；假设这些目标函数均匀分布，则这些目标函数的误差总和为：

$
sum_f E_(o t e) (xi_a|X,f) &= sum_f sum_h sum_(bold(x) in cal(X) - X) P(X) bb(I)(h(bold(x)) eq.not f(bold(x)))P(h|X, xi_a)\
&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) sum_f bb(I)(h(bold(x)) eq.not f(bold(x)))\

&text(font: "STFangsong", "根据假设，总有一半是正确的，因此")\

&= sum_h sum_(bold(x) in cal(X) - X) P(X) P(h|X, xi_a) 1/2 2^(|cal(X|))\
&= 2^(|cal(X)|-1) sum_(x in cal(X)-X)P(X)
$

#h(2em) 因此可知，在多目标目标函数均匀分布的情况下，不同算法所得的误差总和相同。实际情况中，某一算法通常只用于解决单一问题，且其目标函数的分布不均匀（即目标函数重要性不同），因此不同算法所得的误差总和不同。

这告诉我们，在某一任务上表现好的算法在另一任务上表现不一定好。

#pagebreak()
