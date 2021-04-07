import importlib
from typing import Any, Dict, List, Union, Optional
import torch

from omegaconf import DictConfig
from torch import nn, Tensor

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


def import_object(module_path: str, object_name: str) -> Any:
    module = importlib.import_module(module_path)
    if not hasattr(module, object_name):
        raise AttributeError(f'Object `{object_name}` cannot be loaded from `{module_path}`.')
    return getattr(module, object_name)


def import_object_from_path(object_path: str, default_object_path: str = '') -> Any:
    object_path_list = object_path.rsplit('.', 1)
    module_path = object_path_list.pop(0) if len(object_path_list) > 1 else default_object_path
    object_name = object_path_list[0]
    return import_object(module_path=module_path, object_name=object_name)


def load_object(config: DictConfig) -> Any:

    _class = import_object_from_path(object_path=config.class_path)

    if not config.parameters:
        _object = _class()
    else:
        _object = _class(**config.parameters)

    return _object


def model_assembly(backbone: nn.Module, aggregation: nn.Module, head: nn.Module) -> nn.Module:

    model = nn.Sequential()
    model.add_module('backbone', backbone)
    model.add_module('aggregation', aggregation)
    model.add_module('head', head)

    return model
