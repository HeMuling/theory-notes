#import "../../prelude.typ": *

#hd3("CNN")

#hd4("Convolution")
卷积通常是指：
$
Y_(i,j) = (X * W)_(i,j) = sum_(m=0)^(M-1) sum_(n=0)^(N-1) X_(i+m, j+n) W_(m,n)
$
卷积操作包括Conv1d和Conv2d#index("Convolution","Convolution layer")

Conv1d: 通常对于一维特征图，给定输入 $X in bb(R)^(N times L times C_text("in"))$，filter $W in bb(R)^(K times C_text("in") times C_text("out"))$，卷积操作为：
$
  Y_(n, C_text("out"),l) = sum_(c_text("in")=0)^(C_text("in")-1) sum_(k=0)^(K-1) X_(n,l+k, c_text("in")) W_(k, c_text("in"), c_text("out"))
$

Conv2d: 通常对于图像或二维特征图，给定输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")) times C_(text("out")))$，其中 $C_(text("in"))$ 为输入通道数，$C_(text("out"))$ 为输出通道数，$K_H, K_W$ 为卷积核的高度和宽度，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c_text("in")=0)^(C_text("in") - 1) sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c_text("in")) W_(m, n, c_text("in"), c_text("out"))
$
其中，$Y in bb(R)^(N times H_text("out") times W_text("out") times C_text("out"))$，$h, w$ 为空间位置索引，$o$ 为输出通道索引

池化层类型包括：最大池化、平均池化。#index("Convolution","Pooling Layer")
$
  A_(i, j) = max_(m,n) (a, Y_i,j)\
  A_(i, j) = 1/(M*N) sum_(m=0)^(M-1) sum_(n=0)^(N-1) Y_(i+m, j+n)
$

```python
import torch.nn as nn
conv = nn.Conv1d(in_channels, out_channels, kernel_size)
conv = nn.Conv2d(in_channels, out_channels, kernel_size)
pool = nn.MaxPool2d(kernel_size, stride)
pool = nn.AvgPool2d(kernel_size, stride)
```

#hd4("Depthwise Separable Convolution") #index("Convolution","Depthwise Separable Convolution")

Depthwise Separable Convolution @howard2017mobilenets 是一种轻量级卷积操作，其包括两个步骤：Depthwise Convolution 和 Pointwise Convolution.

Depthwise Convolution 对每一个通道单独使用卷积核 filter.对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(K_H times K_W times C_(text("in")))$，卷积操作为：
$
  Y_(n, h, w, c) = sum_(m=0)^(K_H - 1) sum_(n=0)^(K_W - 1) X_(n, h+m, w+n, c) W_(m, n, c)
$
Pointwise Convolution 使用 $1 times 1$ 的卷积核，对每一个通道进行卷积操作。对于输入 $X in bb(R)^(N times H times W times C_(text("in")))$，filter $W in bb(R)^(1 times 1 times C_(text("in")) times C_(text("out")))$，卷积操作为：
$
  Y_(n, h, w, o) = sum_(c=0)^(C_(text("in")) - 1) X_(n, h, w, c) W_(0, 0, c, o)
$
Depthwise Separable Convolution 结合以上两种卷积方式，首先使用 Depthwise Convolution ，然后使用 Pointwise Convolution. 

```py
import torch  
import torch.nn as nn  

depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                      groups=in_channels)
pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
deepwise_separable = pointwise(depthwise(input))
```

