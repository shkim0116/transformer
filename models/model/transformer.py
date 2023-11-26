import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, src_embedding, trg_embedding, encoder, decoder, device, generator, pad_idx):
        super(Transformer, self).__init__()
        self.src_embed = src_embedding
        self.tgt_embed = trg_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.generator = generator
        self.pad_idx = pad_idx

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, source, target):
        # source: (n_batch, source_len)
        # target: (n_batch, target_len)
        src_mask = self.make_pad_mask(source, source, self.pad_idx)       # (n_batch, 1, source_len, source_len)
        trg_mask = self.make_trg_mask(target)               # (n_batch, 1, target_len, target_len)
        src_trg_mask = self.make_pad_mask(target, source, self.pad_idx)   # (n_batch, 1, target_len, source_len)
        src_embed = self.src_embed(source)                  # (n_batch, source_len, d_embed=d_model)
        trg_embed = self.tgt_embed(target)                  # (n_batch, target_len, d_embed=d_model)

        encoder_out = self.encoder(src_embed, src_mask)     # (n_batch, source_len, d_model=h*d_k)
        decoder_out = self.decoder(trg_embed, encoder_out, trg_mask, src_trg_mask)  # (n_batch, target_len, d_model)
        out = self.generator(decoder_out)       # (n_batch, target_len, tgt_vocab_size)
        out = F.log_softmax(out, dim=-1)        # (n_batch, target_len, tgt_vocab_size)

        return out

    def make_pad_mask(self, query, key, pad_idx=1):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask  # True: both not padded / False: any padded parts are masked
        mask.requires_grad = False
        return mask

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        # lower triangle without diagonal
        tri_lower = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tri_lower, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask

    def make_trg_mask(self, trg):
        out = self.make_pad_mask(trg, trg) & self.make_subsequent_mask(trg, trg)
        return out
