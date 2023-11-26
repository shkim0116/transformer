import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        # x = (n_batch, seq_len)
        out = self.embedding(x) * math.sqrt(self.d_embed)       # (n_batch, seq_len, d_embed=d_model)
        return out

