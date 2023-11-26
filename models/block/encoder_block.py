import torch.nn as nn


class EncoderBlock(nn.Module):

    def __init__(self, self_attention, add_norm, ff_layer):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.add_norm1 = add_norm
        self.ff_layer = ff_layer
        self.add_norm2 = add_norm

    def forward(self, source, source_mask):
        out = self.self_attention(source, source, source, source_mask)
        out = self.add_norm1(out, source)
        ff_out = self.ff_layer(out)
        encoder_out = self.add_norm2(ff_out, out)
        return encoder_out



# from models.block.multi_head_attention import MultiHeadAttention
# from models.layer.feed_forward_layer import FeedForwardLayer
# from models.layer.add_norm_layer import AddNormLayer
# import torch
# import copy
# copy = copy.deepcopy
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
#         self_attention=copy(attention),
#         add_norm=copy(add_norm_layer),
#         ff_layer=copy(position_ff))
# # source = [1,2,3,-1,0]
# # source_mask = [0,0,0,1,1]
# src = torch.rand(2, 4, 2*4)  # (n_batch, query_seq, d_model=h*d_k)
# src_mask = torch.zeros(2, 1, 4, 4)   # (n_batch, 1, query_seq_len, key_seq_len)
# encoder_out = encoder_block(src, src_mask)
# print(encoder_out)
# print(encoder_out.size())