from models.build_model import build_model
from transformers import BertTokenizer
from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
trg_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

src_vocab_size = src_tokenizer.vocab_size + 10
tgt_vocab_size = trg_tokenizer.vocab_size + 10
sos_idx = 101
eos_idx = 102

max_len = 32
pad_idx = src_tokenizer.pad_token_id


model = build_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, device=device,
                        max_len=max_len, d_embed=128, n_encoder_layer=4, n_decoder_layer=4,
                        d_model=128, h=4, d_ff=256, pad_idx=pad_idx).to(device)

model.load_state_dict(torch.load('/Users/shkim/PycharmProjects/Transformer/data/checkpoints/adamepochbig30.pth'))

model.eval()

def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(model.device)
    src_mask = model.make_pad_mask(src, src).to(model.device)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len-1):
        memory = memory.to(model.device)
        tgt_mask = model.make_trg_mask(ys).to(model.device)
        src_tgt_mask = model.make_pad_mask(ys, src).to(model.device)
        out = model.decode(ys, memory, tgt_mask, src_tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys

src_sentence = "동의해요. 사실 사람에 비해 공간이 너무 부족해요."

src = src_tokenizer.encode(src_sentence, max_length=max_len, truncation=True)
num_tokens = len(src)
rest = max_len - len(src)
src = torch.tensor([src + [0] * rest])

tgt_tokens = greedy_decode(model,
                         src,
                         max_len=max_len,
                         start_symbol=sos_idx,
                         end_symbol=eos_idx).flatten().cpu().numpy()

print(src_tokenizer.decode(tgt_tokens))

