import torch
import torch.nn as nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.model.transformer import Transformer
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.block.multi_head_attention import MultiHeadAttention
from models.layer.feed_forward_layer import FeedForwardLayer
from models.layer.add_norm_layer import AddNormLayer
from models.embedding.embedding import Embedding
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.transformer_embedding import TransformerEmbedding


def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("cpu"),
                max_len=256,
                d_embed=512,
                n_encoder_layer=6,
                n_decoder_layer=6,
                d_model=512,
                h=8,
                d_ff=2048,
                pad_idx=1):
    import copy
    copy = copy.deepcopy

    src_token_embed = Embedding(
        d_embed=d_embed,
        vocab_size=src_vocab_size)

    tgt_token_embed = Embedding(
        d_embed=d_embed,
        vocab_size=tgt_vocab_size)

    pos_embed = PositionalEncoding(
        d_embed=d_embed,
        max_len=max_len)

    src_embed = TransformerEmbedding(
        token_embed=src_token_embed,
        pos_embed=copy(pos_embed))

    tgt_embed = TransformerEmbedding(
        token_embed=tgt_token_embed,
        pos_embed=copy(pos_embed))

    attention = MultiHeadAttention(
        h=h,
        d_model=d_model)

    position_ff = FeedForwardLayer(
        d_embed=d_embed,
        d_ff=d_ff)

    add_norm_layer = AddNormLayer(
        d_model=d_model)

    encoder_block = EncoderBlock(
        self_attention=copy(attention),
        add_norm=copy(add_norm_layer),
        ff_layer=copy(position_ff))

    decoder_block = DecoderBlock(
        self_attention=copy(attention),
        cross_attention=copy(attention),
        add_norm=copy(add_norm_layer),
        ff_layer=copy(position_ff))

    encoder = Encoder(
        encoder_block=encoder_block,
        n_layer=n_encoder_layer)

    decoder = Decoder(
        decoder_block=decoder_block,
        n_layer=n_decoder_layer)

    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
        src_embedding=src_embed,
        trg_embedding=tgt_embed,
        encoder=encoder,
        decoder=decoder,
        device=device,
        generator=generator,
        pad_idx=pad_idx).to(device)
    model.device = device

    return model

# src_vocab_size = 5
# tgt_vocab_size = 5
# model = build_model(src_vocab_size, tgt_vocab_size, device=torch.device("cpu"), max_len=256,
#                 d_embed=8, n_encoder_layer=2, n_decoder_layer=2, d_model=8, h=2, d_ff=4, pad_idx=1)
#
# src = torch.tensor([[2,3,4,1]])
# trg = torch.tensor([[4,3,2,1]])
#
# out = model(src, trg)
# print(out)
# print(out.size())