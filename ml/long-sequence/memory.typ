#import "../../prelude.typ": *

#hd3("记忆力")
一种记忆力的评估方法是评估模型，在第 $n$ 步时可以利用多远的信息计算输出poli2023hyena. 对于输入序列 $u(t)$，输出序列 $y(t)$，统计以下输出中不为零的数量：
$
(partial y(t)) / (partial u(t-n)), space.quad n=0,1,dots,t
          $
以 SSM 为例，有：
$
  (partial y(t)) / (partial u(t-n)) = C A^n B
$
除此以外，chen2024hope 提出了以下方式：
1. 生成序列：序列第一个位置为`[bos]`，后续的每一个位置随机来自于字典中的token
2. 计算 pre-softmax logits：在计算注意力时，会计算query和所有key的相似性(点积)，因此这里相当于计算了位置 $i$ 和所有位置的相似性
$
  text("logits")_(i,j) = q_i^tack.b k_j
$
3. 对 pre-softmax logit归一化：在所有注意力头上平均
