from typing import Dict, List, Union, Optional
import torch

from typing import Dict, List, Union, Optional

import torch
from torch import Tensor

Batch = Dict[str, Tensor]


def batch_to_device(batch: Batch,
                    device: Optional[torch.device] = None,
                    except_list: Optional[List[str]] = None) -> Batch:

    if device is None:
        device = torch.device('cpu')

    if except_list is None:
        except_list = list()

    for key, value in batch.items():
        if key not in except_list:
            if hasattr(value, 'to'):
                batch[key] = value.to(device)

    return batch


def prediction(batch: Batch, distribution: bool = False) -> Union[List[int], Tensor]:
    if distribution:
        return torch.softmax(batch['logits'], dim=-1)
    else:
        return batch['logits'].argmax(dim=-1).detach().cpu().tolist()


def raw_target(batch: Batch) -> List[int]:
    return batch['target'].detach().cpu().tolist()
