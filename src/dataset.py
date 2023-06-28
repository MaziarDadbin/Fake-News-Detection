from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import pandas as pd


class FeatureDataset(Dataset):
    def __init__(self, file_name, is_train):
        self.is_train = is_train

        file_out = pd.read_csv(file_name)
        body_text = file_out.loc[:, "text"]

        if is_train:
            is_fake = file_out.loc[:, "is_fake"].values
            self.is_fake = torch.tensor(is_fake).float()

        self.ID = file_out.loc[:, "id"].values

        # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        tokens = list(map(lambda t: ["[CLS]"] + tokenizer.tokenize(t)[:511], body_text))
        tokens_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))
        tokens_ids = pad_sequences(
            tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int"
        )

        masks = [[float(i > 0) for i in ii] for ii in tokens_ids]
        self.masks = torch.tensor(masks)

        self.body_text = torch.tensor(tokens_ids)

    def __len__(self):
        return len(self.body_text)

    def __getitem__(self, idx):
        if self.is_train:
            return self.body_text[idx], self.masks[idx], self.is_fake[idx], self.ID[idx]
        else:
            return self.body_text[idx], self.masks[idx], self.ID[idx]
