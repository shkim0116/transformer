import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.requires_grad = False
        encoding = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)

    def forward(self, x):
        # x: (n_batch, seq_len, d_embed=d_model)
        _, seq_len, _ = x.size()
        out = x + self.encoding[:, :seq_len, :]         # (n_batch, seq_len, d_embed=d_model)
        return out

