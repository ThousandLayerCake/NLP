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
            layer (EncoderLayer): 单层编码器结构
            N (int): 编码器层的层数
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
        size (int): 特征向量的维度
        self_attn (MultiHeadedAttention): 多头自注意力层
        feed_forward (PositionwiseFeedForward): 一个包含两个线性层的FFN
        sublayer (nn.ModuleList): 残差+归一化层
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """_summary_

        Args:
            size (int): 特征向量的维度 d_model
            self_attn (MultiHeadedAttention): 多头注意力层
            feed_forward (PositionwiseFeedForward): 一个包含两个线性层的FFN
            dropout (float): 归一化层的丢弃率
        """
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(
            size, dropout), 2)  # 两层残差+归一化


    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

    '''
    语法：lambda 参数序列 : 函数体

    lambda_function = lambda x: self.self_attn(x, x, x, mask)

    等价于 ===>

    def lambda_function(x):
        return self.self_attn(x, x, x, mask)

    sublayer[i] 参数接收一个输入和一个函数
    '''