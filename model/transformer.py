import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, glu_activation='swish'):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.v = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)


        if glu_activation == 'swish':
            self.act = nn.SiLU()
        elif glu_activation == 'gelu':
            self.act = nn.GELU()
        elif glu_activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):

        gate = self.act(self.v(x))
        output = self.w2(self.w1(x) * gate)
        return output


class LongTermAttention(nn.Module):
    def __init__(self, input_size, attn_dropout=0.1):
        super(LongTermAttention, self).__init__()

        self.input_size = input_size
        self.W_q = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, input_size))

        self.dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(input_size)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.normal_(self.W_q, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_k, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_v, mean=0, std=np.sqrt(2.0 / (self.input_size)))


    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):


        _, k_len, _ = K.size()
        temperature = self.input_size ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K = Q_K.repeat(1, k_len, 1)

        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size(), device=Q.device), diagonal=1).bool()
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -1e10)

        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V, mask):


        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q)
        K_ = K.matmul(self.W_k)
        V_ = V.matmul(self.W_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)

        return V_att

    def forward(self, Q, K, V, mask=None):


        Q = Q.unsqueeze(dim=1)
        V_att = self.multi_head_attention(Q, K, V, mask)
        output = self.layer_norm(V+V_att)
        return output

class TransformerBlock(nn.Module):

    def __init__(self, d_q = 64, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=d_v)
            self.norm1 = nn.LayerNorm(normalized_shape=d_v)
            self.norm2 = nn.LayerNorm(normalized_shape=d_v)


        self.W_q = nn.Parameter(torch.Tensor(d_q, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(d_k, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(d_v, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, d_v))
        self.linear1 = nn.Linear(d_v, d_v)
        self.linear2 = nn.Linear(d_v, d_v)

        self.dropout = nn.Dropout(attn_dropout)

        self.glu_ffn = GLUFFN(d_v, d_v, glu_activation = 'gelu')

        self.__init_weights__()


    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.gelu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):


        temperature = self.d_k ** 0.5

        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size(), device=Q.device), diagonal=1).bool()
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -1e2)

        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):


        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)


        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))

        return output


    def forward(self, Q, K, V, S=0, mask=None):


        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:

            X = self.norm1(V_att + V)
            output = self.norm2(self.FFN(X) + X)


        else:
            X = V + V_att
            output = self.FFN(X) + X
        return output
