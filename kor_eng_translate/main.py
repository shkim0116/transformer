import torch
import torch.nn as nn
import pandas as pd
from models.build_model import build_model
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from dataset import *


def build_dataloaders(dataset, batch_size, train_test_split=0.2, train_eval_split=0.2,
                      train_shuffle=True, eval_shuffle=True, test_shuffle=True):
    dataset_len = len(dataset)
    test_len = int(dataset_len * train_test_split)
    train_len = dataset_len - test_len
    train_dataset, test_dataset = random_split(dataset, (train_len, test_len))
    eval_len = int(train_len * train_eval_split)
    train_len = train_len - eval_len
    print(train_len, eval_len, test_len)
    train_dataset, eval_dataset = random_split(train_dataset, (train_len, eval_len))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=eval_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    return train_loader, eval_loader, test_loader


def train(model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for idx, dic in enumerate(data_loader):
        src = dic['source']
        tgt = dic['target']
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(src, tgt_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for idx, dic in enumerate(data_loader):
            src = dic['source']
            tgt = dic['target']
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    return loss_avr



if __name__ == '__main__':
    # data
    kor_eng = pd.read_excel("/Users/shkim/PycharmProjects/Transformer/data/2_대화체.xlsx")
    data = kor_eng[["원문", "번역문"]][:1000]
    data = data.values.tolist()
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    trg_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    max_len = 32

    dataset = CustomDataset(src_tokenizer, trg_tokenizer, data, max_length=max_len)
    train_loader, eval_loader, test_loader = build_dataloaders(dataset=dataset, batch_size=4)
    pad_idx = src_tokenizer.pad_token_id
    src_vocab_size = src_tokenizer.vocab_size + 10
    print(src_vocab_size)
    tgt_vocab_size = trg_tokenizer.vocab_size + 10
    print(src_vocab_size)

    # train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.manual_seed(0)
    model = build_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, device=device,
                        max_len=max_len, d_embed=128, n_encoder_layer=4, n_decoder_layer=4,
                        d_model=128, h=4, d_ff=256, pad_idx=pad_idx).to(device)

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4
    ADAM_EPS = 5e-9
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    N_EPOCH = 30

    for epoch in range(N_EPOCH):
        print(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"train_loss: {train_loss:.5f}")
        valid_loss = evaluate(model, eval_loader, criterion)
        print(f"valid_loss: {valid_loss:.5f}")

    test_loss = evaluate(model, test_loader, criterion)
    print(f"test_loss: {test_loss:.5f}")
    torch.save(model.state_dict(), f"/Users/shkim/PycharmProjects/Transformer/data/checkpoints/adamepochbig{N_EPOCH}.pth")