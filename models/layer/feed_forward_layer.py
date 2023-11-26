import torch.nn as nn


class FeedForwardLayer(nn.Module):

    def __init__(self, d_embed, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_embed, d_ff)  # (d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_embed)  # (d_ff, d_model)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
