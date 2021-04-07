from typing import List

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch import nn

from label_augmented import io, utils


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

        optimizer_object = utils.import_object_from_path(object_path=self.optimizer_config.class_path)
        optimizer = optimizer_object(self.model.parameters(), **self.optimizer_config.parameters)

        return optimizer

    def forward(self, batch: io.Batch) -> io.Batch:
        return self.model(batch)

    def step(self, batch: io.Batch) -> io.Batch:
        batch = self.forward(batch)
        batch = self.criterion(batch)
        batch = io.batch_to_device(batch, except_list=['loss'])
        return batch

    def calculate_f1_score(self,
                           predictions: List[int],
                           targets: List[int]) -> float:
        return f1_score(y_true=targets, y_pred=predictions, average=self.f1_type)

    def training_step(self,
                      batch: io.Batch,
                      batch_idx: int) -> io.Batch:

        batch = self.step(batch)

        batch_output = {
            'loss': batch['loss'],
            'predictions': io.prediction(batch),
            'targets': io.raw_target(batch)
        }

        self.log(name='train_loss', value=batch['loss'].item(),
                 prog_bar=False, on_step=True, on_epoch=False)

        self.log(name=f'train_batch_f1_{self.f1_type}',
                 value=self.calculate_f1_score(predictions=batch_output['predictions'],
                                               targets=batch_output['targets']),
                 prog_bar=False, on_step=True, on_epoch=False)

        return batch_output

    def validation_step(self,
                        batch: io.Batch,
                        batch_idx: int) -> io.Batch:

        batch = self.step(batch)

        batch_output = {
            'loss': batch['loss'],
            'predictions': io.prediction(batch),
            'targets': io.raw_target(batch)
        }

        return batch_output

    def epoch_end(self, outputs: List[io.Batch], stage: str = 'train') -> None:

        losses: List[float] = list()
        predictions: List[int] = list()
        targets: List[int] = list()

        for batch in outputs:
            losses.append(batch['loss'].item())
            predictions.extend(batch['predictions'])
            targets.extend(batch['targets'])

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

    def training_epoch_end(self, outputs: List[io.Batch]) -> None:
        self.epoch_end(outputs=outputs, stage='train')

    def validation_epoch_end(self, outputs: List[io.Batch]) -> None:
        self.epoch_end(outputs=outputs, stage='valid')
