import os
from typing import Optional, Union, List, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from label_augmented import io


class YahooAnswersDataset(Dataset):

    def __init__(self, data_path: str, shuffle: bool = True):
        super().__init__()

        self.data_path = data_path

        data = pd.read_csv(data_path, header=None)
        data[0] -= 1

        if shuffle:
            data = data.sample(frac=1)

        self.texts = data[1].tolist()
        self.target = data[0].tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:

        sample = {
            'text': self.texts[index],
            'target': self.target[index]
        }

        return sample


class Preparer:

    def __init__(self, model_name: str = 'distilbert-base-uncased', max_length: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.pad_index = self.tokenizer.pad_token_id

    def __call__(self, texts: List[str]) -> Tensor:
        tokenized = self.tokenizer(texts,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)['input_ids']

        return tokenized

    def collate(self, batch: Dict[str, Union[str, int]]) -> io.Batch:
        texts, targets = list(), list()

        for sample in batch:
            texts.append(sample['text'])
            targets.append(sample['target'])

        sequence_indices = self(texts)

        output_batch = {
            'sequence_indices': sequence_indices,
            'pad_mask': (sequence_indices != self.pad_index).long(),
            'target': torch.Tensor(targets).long()
        }

        return output_batch

    def decoding(self, batch: io.Batch) -> List[str]:
        return self.tokenizer.batch_decode(sequences=batch['sequence_indices'].detach().cpu().tolist(),
                                           skip_special_tokens=True)


class YahooAnswersDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path: str = './data/',
                 batch_size: int = 128,
                 pretrained_model_name: str = 'distilbert-base-uncased',
                 max_length: int = 32):
        super().__init__()

        self.data_path = data_path
        self.train_data_path = os.path.join(self.data_path, 'train.csv')
        self.valid_data_path = os.path.join(self.data_path, 'test.csv')

        self.batch_size = batch_size

        self.preparer = Preparer(model_name=pretrained_model_name, max_length=max_length)

        self.train_data = ...
        self.valid_data = ...

    def prepare_data(self, *args, **kwargs):
        ...

    def setup(self, stage: Optional[str] = None):
        self.train_data = YahooAnswersDataset(data_path=self.train_data_path, shuffle=True)
        self.valid_data = YahooAnswersDataset(data_path=self.valid_data_path, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size,
                            collate_fn=self.preparer.collate, shuffle=True)
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(dataset=self.valid_data, batch_size=self.batch_size,
                            collate_fn=self.preparer.collate, shuffle=False)
        return loader
