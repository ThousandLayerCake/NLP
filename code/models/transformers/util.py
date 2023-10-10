# -*-coding:utf-8-*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout=None):
    """_summary_

        计算缩放的点积注意力

    Args:
        query (torch.Tensor): _description_
        key (torch.Tensor): _description_
        value (torch.Tensor): _description_
        mask (_type_, optional): _description_. Defaults to None.
        dropout (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
        nn.Tensor : 加权后的Value
        nn.Tensor : 注意力权重
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N: int):
    """_summary_

    产生N个完全相同的网络层，要求输入与输出形状一致

    Args:
        module (nn.Module): 需要复制的神经网络层
        N (int): 神经网络层数
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """_summary_

    将输入token和输出token转换为d_model维的向量

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # TODO what is vocab?
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    """_summary_

    LayerNorm层，用于归一化

    Attrs:
        a_2 (nn.Parameter): _description_
        b_2 (nn.Parameter): _description_
        eps (float): _description_
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 特征的维度要分段放入每个头，所以要整除
        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),
                              4)  # 前三层是线性投影层，最后一层是线性输出
        self.attn = None  # 自注意力权重
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 对所有的头都使用同一个mask
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 做线性投影，变形为[h, d_k]
        query, key, value = [linear(qkv).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, qkv in zip(self.linears, (query, key, value))]

        # 在批处理中，对所有投影的向量应用attention
        x, self.attn = attention(query=query, key=key,
                                 value=value, mask=mask, dropout=self.dropout)

        # 输出使用 view 拼接起来，应用到最终的线性层输出
        x = x.transpose(1, 2).contiguous()\
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.l_1 = nn.Linear(d_model, d_ff)
        self.l_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l_2(self.dropout(F.relu(self.l_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间上计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    """_summary_

    层归一化和残差连接

    Attrs:
        norm (nn.Module): 归一化层
        dropout (float): 丢弃结果的概率
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """_summary_

        残差连接：将输入经过子层后，得到的中间结果与原始输入相加。

        Args:
            x (_type_): 未经过归一化的特征
            sublayer (function): 需要残差连接和归一化的前一层
        """
        return x + self.dropout(sublayer(self.norm(x)))
