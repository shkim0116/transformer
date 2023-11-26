import torch.nn as nn
import copy


class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(n_layer)])

    def forward(self, x, source_mask):
        for layer in self.layers:
            x = layer(x, source_mask)       # (n_batch, source_len, d_model=h*d_k)
        return x



# from models.block.encoder_block import EncoderBlock
# from models.block.multi_head_attention import MultiHeadAttention
# from models.layer.feed_forward_layer import FeedForwardLayer
# from models.layer.add_norm_layer import AddNormLayer
# import torch
#
# h = 2
# d_model = 8
# d_embed = d_model
# d_ff = 4
# attention = MultiHeadAttention(
#         h=h,
#         d_model=d_model)
# position_ff = FeedForwardLayer(
#     d_embed=d_embed,
#     d_ff=d_ff)
# add_norm_layer = AddNormLayer(
#     d_model=d_model)
# encoder_block = EncoderBlock(
#         self_attention=copy.deepcopy(attention),
#         add_norm=copy.deepcopy(add_norm_layer),
#         ff_layer=copy.deepcopy(position_ff))
#
# src = torch.rand(2, 4, 2*4)  # (n_batch, query_seq, d_model=h*d_k)
# src_mask = torch.zeros(2, 1, 4, 4)   # (n_batch, 1, query_seq_len, key_seq_len)
# encoder = Encoder(encoder_block, 2)
# encoder_out = encoder(src, src_mask)
# print(encoder_out)
# print(encoder_out.size())