import numpy as np
import torch.nn as nn
from util import *


class Decoder(nn.Module):
    """_summary_

    解码器：包含N层 解码器layer

    Attrs:
        layers (ModuleList): N层解码器列表
        norm (nn.Module): 归一化层
    """

    def __init__(self, layer, N):
        """_summary_
        
        Args:
            layer (DecoderLayer): 单层解码器
            N (int): 层数
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """_summary_

    Attrs:
        size (int): 特征向量的维度
        self_attn (MultiHeadedAttention): 多头注意力层
        src_attn (MultiHeadedAttention): 多头注意力层
        feed_forward (PositionwiseFeedForward): 一个包含两个线性层的FFN
        sublayer (nn.ModuleList): 残差+归一化层
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Args:
            size (int): 特征向量的维度
            self_attn (MultiHeadedAttention): 多头注意力层
            src_attn (MultiHeadedAttention): 多头注意力层
            feed_forward (PositionwiseFeedForward): 一个包含两个线性层的FFN
            dropout (float): 归一化层的丢弃率
        """
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(
            size, dropout), 3)  # 三层残差+归一化

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
