import torch.nn as nn


class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, add_norm, ff_layer):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.add_norm1 = add_norm
        self.add_norm2 = add_norm
        self.add_norm3 = add_norm
        self.ff_layer = ff_layer

    def forward(self, target, encoder_out, target_mask, source_target_mask):
        # target : (1,1,8)
        # encoder_out : (1,8,8)
        out = self.self_attention(target, target, target, target_mask)
        # target_out : (1,1,8)
        out = self.add_norm1(out, target)
        cross_attention_out = self.cross_attention(out, encoder_out, encoder_out, source_target_mask)
        out = self.add_norm2(cross_attention_out, out)
        ff_out = self.ff_layer(out)
        decoder_out = self.add_norm3(ff_out, out)
        return decoder_out
