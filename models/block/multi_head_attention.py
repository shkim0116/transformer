import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        assert d_model % h == 0, 'd_model must equal h * d_k'
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: (n_batch, query_seq, d_k)  -> (n_batch, query_seq, d_model)    (1,1,8)
        # key: (n_batch, key_seq, d_k)  -> (n_batch, key_seq, d_model)          (1,8,8)
        # value: (n_batch, value_seq, d_k)  -> (n_batch, value_seq, d_model)    (1,8,8)
        n_batch = query.size(0)
        query_seq = query.size(1)
        key_seq = key.size(1)
        query = self.w_q(query)    # (n_batch, query_seq, d_model)  (1,1,8)
        key = self.w_k(key)        # (n_batch, key_seq, d_model)    (1,8,8)
        value = self.w_v(value)    # (n_batch, key_seq, d_model)    (1,8,8)
        query = query.view(n_batch, query_seq, self.h, -1).transpose(1, 2)   # (n_batch, h, query_seq, d_k)
        key = key.view(n_batch, key_seq, self.h, -1).transpose(1, 2)         # (n_batch, h, key_seq, d_k)
        value = value.view(n_batch, key_seq, self.h, -1).transpose(1, 2)     # (n_batch, h, key_seq, d_k)

        out = self.scaled_dot_product_attention(query, key, value, mask)          # (n_batch, h, query_seq, d_k)
        out = out.transpose(1, 2).contiguous().view(n_batch, query_seq, -1)
        out = self.out_fc(out)
        return out

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, query_seq, key_seq)
        out = torch.matmul(attention_prob, value)  # (n_batch, h, query_seq, d_k)
        return out


# attention = MultiHeadAttention(2, 2*4)
# src = torch.rand(2, 5, 2*4)  # (n_batch, query_seq, d_model=h*d_k)
# src = torch.tensor(src)
# print(src.size())
# result = attention(src, src, src)
# print(result)