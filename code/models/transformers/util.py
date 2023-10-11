# -*-coding:utf-8-*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout: nn.Dropout = None):
    """_summary_

        计算缩放点积注意力

    Args:
        query (torch.Tensor): 查询向量
        key (torch.Tensor): 键向量，一般与value相同
        value (torch.Tensor): 值向量，一般与key相同
        mask (torch.Tensor, optional): mask向量，他是一个与key形状相同的向量，主要作用是找到有0的地方，用-1e9 (负无穷) 来替代。
        dropout (nn.Dropout, optional): 丢弃层，在对象创建的时候就指定了丢弃率，调用时传入的参数为要随机丢弃结果的向量。

    Returns:
        res_1 (nn.Tensor): 加权后的Value
        res_2 (nn.Tensor): 注意力权重
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


def clones(module: nn.Module, N: int):
    """_summary_

    产生N个完全相同的网络层，要求输入与输出形状一致

    Args:
        module (nn.Module): 需要复制的神经网络层
        N (int): 神经网络层数

    Returns:
        res_1 (nn.ModuleList): N个相同的模型列表 [module1, module2, ..., modulen]
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """_summary_

    嵌入层，将词转为向量

    Attrs:
        d_model (int): 转换后输出向量的维度
        lut (nn.Embedding): 嵌入层
    """

    def __init__(self, d_model, vocab):
        """_summary_

        将词总数为vocab的词典转换为d_model维的向量

        Attrs:
            d_model (_type_): 词向量的维度，简单点说就是用多少个数字来表示一个向量
            vocab (_type_): 词典的大小，也就是单词的数量
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    """_summary_

    LayerNorm层，用于归一化

    Attrs:
        scale (nn.Parameter): scale可学习的缩放因子 shape = (d_model)
        beta (nn.Parameter): beta是可学习的偏移因子 shape = (d_model)
        epsilon (float): epsilon是很小的常数，用于数值稳定。
    """

    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.epsilon) + self.beta


class MultiHeadedAttention(nn.Module):
    """_summary_

        多头注意力层

    Args:
        h (int): 注意力层的头数量
        d_k (int): 每个注意力层处理的向量维度数
        linears (nn.ModuleList): 四个线性层的列表，分别负责 投影Q、K、V，以及最终结果输出
        attn (torch.Tensor)：自注意力权重
        dropout (nn.Dropout)：丢弃层
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 特征的维度要分段放入每个头，所以要整除
        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),
                              4)  # 前三层是线性投影层，最后一层是线性输出
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """_summary_

        对传入的Q、K、V做线性投影后，经过多头注意力层获得各种模式的向量输出，然后将结果使用一个线性层拼接后输出。
        
        Args:
            query (_type_): _description_
            key (_type_): _description_
            value (_type_): _description_
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
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
