import torch.nn as nn
from util import *


class Encoder(nn.Module):
    """_summary_

    编码器：由N个相同的Transformers Encoder Block 组成。

    Attrs:
        layers (nn.ModuleList): N个相同的网络层列表
        norm : 归一化层
    """

    def __init__(self, layer, N):
        """_summary_

        Args:
            layer (_type_): _description_
            N (_type_): _description_
        """
        super(Encoder, self).__init()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """_summary_

    编码器的单层结构，包含多头注意力层、残差连接、归一化、前馈层

    Attrs:
        nn (_type_): _description_
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(
            size, dropout), 2)  # 两层残差+归一化
        self.size = size

    def forward(self, x, mask):
        # lambda 参数序列 : 函数体
        # 等价于 ==>
        #   定义 def lambda_function(x): {self.self_attn(x,x,x,mask)}
        #   将lambda_function作为参数传入self.sublayer[0]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
