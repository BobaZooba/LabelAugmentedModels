from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, fields, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn import functional as F


class Container(OrderedDict):

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        # assert all(
        #     field.default is None for field in class_fields[1:]
        # ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, torch.Tensor):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

    def _replace_value_to_device(self,
                                 value: Any,
                                 device: Optional[torch.device] = None,
                                 to_cuda: bool = False,
                                 to_cpu: bool = False) -> Any:

        if isinstance(value, torch.Tensor):
            if to_cuda:
                value = value.cuda(device=device)
            elif to_cpu:
                value = value.cpu()
            else:
                value = value.to(device=device)
        elif hasattr(value, 'replace_to_device'):
            value = value.replace_to_device(device=device, to_cuda=to_cuda, to_cpu=to_cpu)
        elif isinstance(value, (list, tuple)):
            value_type = type(value)
            value = value_type([self._replace_value_to_device(value=piece,
                                                              device=device,
                                                              to_cuda=to_cuda,
                                                              to_cpu=to_cpu) for piece in value])
        elif isinstance(value, dict):
            value = {key: self._replace_value_to_device(value=piece,
                                                        device=device,
                                                        to_cuda=to_cuda,
                                                        to_cpu=to_cpu)
                     for key, piece in value.items()}

        return value

    def _get_device_from_value(self, value: Any):

        devices = set()

        if isinstance(value, torch.Tensor):
            devices.add(value.device)
        elif hasattr(value, 'device'):
            tmp_device = value.device
            if isinstance(tmp_device, list):
                devices.update(tmp_device)
            else:
                devices.add(value.device)
        elif isinstance(value, (list, tuple)):
            for piece in value:
                device = self._get_device_from_value(value=piece)
                devices.update(device)
        elif isinstance(value, dict):
            for piece in value.values():
                device = self._get_device_from_value(value=piece)
                devices.update(device)

        devices = list(devices)

        return devices

    @property
    def device(self):

        devices = set()

        for value in self.__dict__.values():
            devices.update(self._get_device_from_value(value=value))

        devices = list(devices)

        if len(devices) == 1:
            return devices[0]
        elif not devices:
            return torch.device('cpu')
        else:
            return devices

    def _detach_value(self, value: Any) -> Any:

        if hasattr(value, 'detach'):
            value = value.detach()
        elif isinstance(value, (list, tuple)):
            value_type = type(value)
            value = value_type([self._detach_value(value=piece) for piece in value])
        elif isinstance(value, dict):
            value = {key: self._detach_value(value=piece)
                     for key, piece in value.items()}

        return value

    def replace_to_device(self,
                          device: Optional[torch.device] = None,
                          to_cuda: bool = False,
                          to_cpu: bool = False) -> Dict[str, Any]:

        output = {key: self._replace_value_to_device(value=value,
                                                     device=device,
                                                     to_cuda=to_cuda,
                                                     to_cpu=to_cpu)
                  for key, value in self.__dict__.items()}

        output = self.__class__(**output)

        return output

    def cpu(self) -> Dict[str, Any]:
        return self.replace_to_device(to_cpu=True)

    def cuda(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        return self.replace_to_device(device=device, to_cuda=True)

    def to(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        return self.replace_to_device(device=device)

    def detach(self) -> Dict[str, Any]:

        output = {key: self._detach_value(value=value)
                  for key, value in self.__dict__.items()}

        output = self.__class__(**output)

        return output


@dataclass
class RawText:

    text: str
    target: int


@dataclass
class Sample(ABC, Container):
    ...


@dataclass
class ModelInput(Sample):

    sequence_indices: Optional[torch.Tensor] = None
    pad_mask: Optional[torch.Tensor] = None
    position_indices: Optional[torch.Tensor] = None
    segment_indices: Optional[torch.Tensor] = None

    def set_pad_mask(self, pad_index: int = 0):
        if self.pad_mask is None:
            self.pad_mask = (self.sequence_indices != pad_index).long()


@dataclass
class Encodings(Sample):

    backbone: Optional[torch.Tensor] = None
    aggregation: Optional[torch.Tensor] = None
    head: Optional[torch.Tensor] = None


@dataclass
class ModelOutput(Sample):

    encodings: Encodings = field(default_factory=Encodings)
    embeddings: Optional[torch.Tensor] = None
    embeddings_is_normalized: bool = False


@dataclass
class LossOutput(Sample):

    value: Optional[torch.Tensor] = None

    def backward(self,
                 gradient: Optional[torch.Tensor] = None,
                 retain_graph: Optional[bool] = None,
                 create_graph: Optional[bool] = False):

        if self.value is None:
            raise RuntimeError('loss is None object')

        self.value.backward(gradient=gradient,
                            retain_graph=retain_graph,
                            create_graph=create_graph)

    def item(self):
        return self.value.item()


@dataclass
class ModelIO(Sample):

    input: ModelInput = field(default_factory=ModelInput)
    output: ModelOutput = field(default_factory=ModelOutput)
    target: Optional[torch.Tensor] = None
    loss: LossOutput = field(default_factory=LossOutput)

    def set_pad_mask(self, pad_index: int = 0):
        self.input.set_pad_mask(pad_index=pad_index)

    def backward(self,
                 gradient: Optional[torch.Tensor] = None,
                 retain_graph: Optional[bool] = None,
                 create_graph: Optional[bool] = False):
        self.loss.backward(gradient=gradient,
                           retain_graph=retain_graph,
                           create_graph=create_graph)

    def item(self):
        return self.loss.item()

    @property
    def logits(self):
        return self.output.encodings.head

    @property
    def probability(self):
        if self.logits is not None:
            return torch.softmax(self.logits, dim=-1)
        else:
            raise AttributeError('head encodings is None')

    @property
    def prediction(self):
        return self.logits.argmax(dim=-1).detach().cpu().tolist()

    @property
    def raw_target(self):
        return self.target.detach().cpu().tolist()

    def get_normalized_embeddings(self) -> Optional[torch.Tensor]:
        if self.output.embeddings_is_normalized:
            return self.output.embeddings
        elif isinstance(self.output.embeddings, torch.Tensor):
            return F.normalize(self.output.embeddings)

    def normalize_embeddings(self):
        if not self.output.embeddings_is_normalized:
            self.output.embeddings = self.get_normalized_embeddings()
            if self.output.embeddings is not None:
                self.output.embeddings_is_normalized = True

    @property
    def similarity_matrix(self) -> torch.Tensor:
        embeddings = self.get_normalized_embeddings()

        if embeddings is not None:
            similarity_matrix = embeddings @ embeddings.t()
            return similarity_matrix

