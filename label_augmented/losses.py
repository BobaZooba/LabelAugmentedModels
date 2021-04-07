from typing import Optional

import torch
from torch import nn

from label_augmented import io


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean'):
        super().__init__(weight=weight,
                         size_average=size_average,
                         ignore_index=ignore_index,
                         reduce=reduce,
                         reduction=reduction)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        loss = super().forward(sample.logits, sample.target)

        sample.loss.value = loss

        return sample
