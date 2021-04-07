from typing import List, Any, Dict, Union, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch import nn, Tensor

from label_augmented.utils import import_object_from_path, prediction, batch_to_cpu, Batch


class LightningClassifier(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer_config: DictConfig,
                 f1_type: str = 'macro'):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.optimizer_config = optimizer_config
        self.f1_type = f1_type

    def configure_optimizers(self):

        optimizer_object = import_object_from_path(object_path=self.optimizer_config.class_path)
        optimizer = optimizer_object(self.model.parameters(), **self.optimizer_config.parameters)

        return optimizer

    def forward(self, batch: Batch) -> Batch:
        return self.model(batch)

    def step(self, batch: Batch) -> Batch:
        batch = self.forward(batch)
        batch = self.criterion(batch)
        batch = batch_to_cpu(batch, except_list=['loss'])
        return batch

    def calculate_f1_score(self,
                           predictions: List[int],
                           targets: List[int]) -> float:
        return f1_score(y_true=targets, y_pred=predictions, average=self.f1_type)

    def calculate_f1_score_from_batch(self, batch: Batch) -> float:
        return self.calculate_f1_score(predictions=prediction(batch=batch),
                                       targets=batch['target'].detach().cpu().tolist())

    def training_step(self,
                      batch: Batch,
                      batch_idx: int) -> Batch:

        batch = self.step(batch)

        self.log(name='train_loss', value=batch['loss'].item(),
                 prog_bar=False, on_step=True, on_epoch=False)

        self.log(name=f'train_batch_f1_{self.f1_type}',
                 value=self.calculate_f1_score_from_batch(batch=batch),
                 prog_bar=False, on_step=True, on_epoch=False)

        return batch

    def validation_step(self,
                        batch: Batch,
                        batch_idx: int) -> Batch:

        batch = self.step(batch)

        return batch

    def epoch_end(self, outputs: List[Batch], stage: str = 'train') -> None:

        losses: List[float] = list()
        predictions: List[int] = list()
        targets: List[int] = list()

        for batch in outputs:
            losses.append(batch['loss'].item())
            predictions.extend(prediction(batch=batch))
            targets.extend(batch['target'].detach().cpu().tolist())

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

        del outputs

    def training_epoch_end(self, outputs: List[Batch]) -> None:
        self.epoch_end(outputs=outputs, stage='train')

    def validation_epoch_end(self, outputs: List[Batch]) -> None:
        self.epoch_end(outputs=outputs, stage='valid')
