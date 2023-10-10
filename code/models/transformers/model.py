# -*-coding:utf-8-*-
import copy
from decoder import *
from encoder import *
import torch.nn as nn
import torch.nn.functional as F
from util import *


class EncoderDecoder(nn.Module):
    """_summary_

    基础的Encoder-Decoder结构，可以用于其他模型。

    Attrs:
        encoder (_type_): _description_
        decoder (_type_): _description_
        src_embed (_type_): _description_
        tgt_embed (_type_): _description_
        generator (model.Generator): 特征生成器，用于将原始输入转为特征
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """_summary_

        将输入经过编码器，然后再经过解码器，得到特征输出
        Args:
            src (_type_): _description_
            tgt (_type_): _description_
            src_mask (_type_): _description_
            tgt_mask (_type_): _description_
        """
        return self.decode(
            memory=self.encode(src, src_mask),
            src_mask=src_mask,
            tgt=tgt,
            tgt_mask=tgt_mask
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """_summary_

    定义答案生成器。由 linear 和 softmax 组成。

    Attrs:
        proj (_nn.Linear_): 将特征向量映射到答案空间
    """

    def __init__(self, d_model, vocab):
        """_summary_

        构建特征向量到答案空间的线性映射层

        Args:
            d_model (_int_): 特征向量的维度
            vocab (_type_): 答案单词的数量
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048,
               h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # TODO 非常重要
    # 使用Glorot / fan_avg方法初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
