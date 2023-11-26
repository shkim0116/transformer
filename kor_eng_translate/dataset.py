import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, src_tokenizer, trg_tokenizer, data, max_length):
        self.docs = []
        src_pad_idx = src_tokenizer.pad_token_id
        trg_pad_idx = trg_tokenizer.pad_token_id
        for line in tqdm(data):
            # source = '[SOS]' + line[0] + '[EOS]'
            source = src_tokenizer.encode(line[0], max_length=max_length, truncation=True)
            rest = max_length - len(source)
            source = torch.tensor(source + [src_pad_idx] * rest)
            # target = '[SOS]' + line[1] + '[EOS]'
            target = trg_tokenizer.encode(line[1], max_length=max_length, truncation=True)
            rest = max_length - len(target)
            target = torch.tensor(target + [trg_pad_idx] * rest)

            doc = {
                'source_str': src_tokenizer.convert_ids_to_tokens(source),
                'source': source,
                'target_str': trg_tokenizer.convert_ids_to_tokens(target),
                'target': target,
                'token_num': (target[..., 1:] != trg_pad_idx).data.sum()
            }
            self.docs.append(doc)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        item = self.docs[idx]
        return item


# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
# tokenizer.add_tokens(['[SOS]', '[EOS]'])


