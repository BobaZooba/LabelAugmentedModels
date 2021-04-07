from typing import List, Any, Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch import nn

from label_augmented import utils, io


class LightningClassifier(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer_config: DictConfig,
                 f1_type: str ='macro'):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.optimizer_config = optimizer_config
        self.f1_type = f1_type

    def configure_optimizers(self):

        optimizer_object = utils.import_object_from_path(object_path=self.optimizer_config.class_path)
        optimizer = optimizer_object(self.model.parameters(), **self.optimizer_config.parameters)

        return optimizer

    def forward(self, sample: io.ModelIO) -> io.ModelIO:
        return self.model(sample)

    def step(self, batch: io.ModelIO) -> io.ModelIO:
        batch = self.forward(batch)
        batch = self.criterion(batch)
        return batch

    def calculate_f1_score(self,
                           predictions: List[int],
                           targets: List[int]) -> float:
        return f1_score(y_true=targets, y_pred=predictions, average=self.f1_type)

    def calculate_f1_score_from_batch(self, batch: io.ModelIO) -> float:
        return self.calculate_f1_score(predictions=batch.prediction, targets=batch.raw_target)

    def training_step(self,
                      batch: io.ModelIO,
                      batch_idx: int) -> Dict[str, Union[torch.Tensor, io.ModelIO]]:

        batch = self.step(batch)

        self.log(name='train_loss', value=batch.item(),
                 prog_bar=False, on_step=True, on_epoch=False)

        self.log(name=f'train_batch_f1_{self.f1_type}',
                 value=self.calculate_f1_score_from_batch(batch=batch),
                 prog_bar=False, on_step=True, on_epoch=False)

        output = {
            'loss': batch.loss.value,
            'batch': batch
        }

        return output

    def validation_step(self,
                        batch: io.ModelIO,
                        batch_idx: int) -> io.ModelIO:

        batch = self.step(batch)

        return batch

    def epoch_end(self, outputs: List[Any], stage: str = 'train') -> None:

        losses: List[float] = list()
        predictions: List[int] = list()
        targets: List[int] = list()

        for batch in outputs:
            if stage == 'train':
                batch = batch['batch']
            losses.extend(batch.item())
            predictions.extend(batch.prediction)
            targets.extend(batch.raw_target)

        self.log(name=f'{stage}_epoch_loss',
                 value=np.mean(losses),
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True)

        self.log(name=f'{stage}_epoch_f1_{self.f1_type}',
                 value=self.calculate_f1_score(predictions=predictions, targets=targets),
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end(outputs=outputs, stage='train')

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end(outputs=outputs, stage='valid')
