import torch.nn as nn


class AddNormLayer(nn.Module):

    def __init__(self, d_model):
        super(AddNormLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, new_output, old_output):
        out = new_output + old_output
        out = self.norm(out)
        return out
