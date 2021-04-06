import torch

from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from src import io
from typing import Dict, Tuple, Optional, Union, List


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

    def __getitem__(self, index: int) -> io.RawText:

        text = self.texts[index]
        target = self.target[index]

        output = io.RawText(text=text, target=target)

        return output


class Preparer:

    def __init__(self, model_name: str = 'distilbert-base-uncased', max_length: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.pad_index = self.tokenizer.pad_token_id

    def __call__(self, texts: List[str]) -> torch.Tensor:

        tokenized = self.tokenizer(texts,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)['input_ids']

        return tokenized

    def collate(self, batch: Tuple[io.RawText]) -> io.ModelIO:

        texts, targets = list(), list()

        for sample in batch:
            texts.append(sample.text)
            targets.append(sample.target)

        tokenized_texts = self(texts)
        targets = torch.Tensor(targets)

        model_input = io.ModelInput(sequence_indices=tokenized_texts)

        model_input.set_pad_mask(pad_index=self.pad_index)

        model_io = io.ModelIO(input=model_input, target=targets.long())

        return model_io
