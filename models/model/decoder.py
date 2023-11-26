import torch.nn as nn
import copy


class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(n_layer)])

    def forward(self, x, encoder_out, target_mask, source_target_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, target_mask, source_target_mask)      # (n_batch, target_len, d_model=h*d_k)
        return x



