import random
from typing import Union, List

import numpy as np
import torch
from torch import nn


def get_label_candidates(embeddings, labels, num_labels, label_samples_ratio=0.7, min_samples=3):

    label_candidates = {}
    indices = torch.arange(embeddings.size(0))

    for n_label in range(num_labels):

        label_subset = indices[labels == n_label]

        if label_subset.size(0) < min_samples:
            continue

        label_candidates[n_label] = list()

        n_samples = int(label_subset.size(0) * label_samples_ratio)

        for i in range(label_subset.size(0) - n_samples + 1):
            label_candidates[n_label].append(embeddings[label_subset[i:i + n_samples]])

    return label_candidates


def update_memory(label_candidates,
                     memory, memory_mask,
                     memory_collected_flag,
                     memory_indices,
                     max_candidates=10):
    for n_label, candidates in label_candidates.items():

        candidates = random.sample(candidates, max_candidates) if len(candidates) > max_candidates else candidates
        candidates = torch.stack([candidate.mean(dim=0) for candidate in candidates])

        if memory_collected_flag[n_label]:
            ...
            # отправить на обновление центроид
        else:
            insert_indices = memory_indices[memory_mask[n_label] == False][:candidates.size(0)]
            if insert_indices.size(0) == 0:
                memory_collected_flag[n_label] = True
                # отправить на обновление центроид
            else:
                # добавление новых центроид
                candidates = candidates[:insert_indices.size(0)]
                memory[n_label][insert_indices] = candidates
                memory_mask[n_label][insert_indices] = torch.ones(candidates.size(0)).bool()

    return memory, memory_mask, memory_collected_flag


def update_exist_memory(candidates, n_label, memory, memory_mask, memory_n_no_updates):

    similarity = candidates @ memory[n_label][memory_mask[n_label]].t()

    top_indices = similarity.topk(candidates.size(0)).indices

    insert_indices = torch.zeros(candidates.size(0))

    for n_candidate, index_list in enumerate(top_indices):
        for index in index_list:
            if index not in insert_indices:
                insert_indices[n_candidate] = index
                break

    insert_indices = insert_indices.long()

    memory[n_label][insert_indices] = (candidates + memory[n_label][insert_indices]) / 2

    memory_n_no_updates[n_label] += 1
    memory_n_no_updates[n_label][insert_indices] = 0

    return memory, memory_n_no_updates


class OldBootstrapLabelMemoryStorage(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_labels: int,
                 memory_size_per_label: Union[int, List[int]] = 128,
                 label_samples_ratio: float = 0.7,
                 momentum: float = 0.5,
                 min_samples_per_label: int = 3,
                 max_candidates: int = 10,
                 max_no_updates: int = 32):
        super().__init__()

        self.model_dim = model_dim
        self.num_labels = num_labels

        if isinstance(memory_size_per_label, list):
            self.memory_size_per_label = memory_size_per_label
        else:
            self.memory_size_per_label = [memory_size_per_label for _ in range(self.num_labels)]

        self.momentum = momentum
        self.label_samples_ratio = label_samples_ratio
        self.min_samples_per_label = min_samples_per_label
        self.max_candidates = max_candidates
        self.max_no_updates = max_no_updates

        self.updating = True

        self.memory = [nn.Parameter(torch.zeros(self.memory_size_per_label[n], self.model_dim), requires_grad=False)
                       for n in range(self.num_labels)]
        self.memory_mask = [nn.Parameter(torch.zeros(self.memory_size_per_label[n]).bool(), requires_grad=False)
                            for n in range(self.num_labels)]
        self.memory_collected_flag = nn.Parameter(torch.zeros(self.num_labels).bool(), requires_grad=False)
        self.memory_norms = [nn.Parameter(torch.zeros(self.memory_size_per_label[n]), requires_grad=False)
                             for n in range(self.num_labels)]
        self.memory_n_no_updates = [nn.Parameter(torch.zeros(self.memory_size_per_label[n]), requires_grad=False)
                                    for n in range(self.num_labels)]
        self.memory_n_updates = [nn.Parameter(torch.zeros(self.memory_size_per_label[n]), requires_grad=False)
                                 for n in range(self.num_labels)]
        self.memory_indices = [nn.Parameter(torch.arange(self.memory_size_per_label[n]), requires_grad=False)
                               for n in range(self.num_labels)]

    def _get_label_candidates(self, embeddings, labels):

        label_candidates = {}
        indices = torch.arange(embeddings.size(0))

        for n_label in range(self.num_labels):

            label_subset = indices[labels == n_label]

            if label_subset.size(0) < self.min_samples_per_label:
                continue

            label_candidates[n_label] = list()

            n_samples = int(label_subset.size(0) * self.label_samples_ratio)

            if not n_samples:
                continue

            for i in range(label_subset.size(0) - n_samples + 1):
                label_candidates[n_label].append(embeddings[label_subset[i:i + n_samples]])

        return label_candidates

    def _set_memory(self, memory: torch.Tensor, n_label: int, insert_indices: torch.Tensor):

        self.memory[n_label][insert_indices] = memory
        self.memory_mask[n_label][insert_indices] = torch.ones(insert_indices.size(0)).bool()
        self.memory_n_updates[n_label][insert_indices] += 1
        self.memory_n_no_updates[n_label] += 1
        self.memory_n_no_updates[n_label][insert_indices] = 0

    def _update_exist_memory(self, candidates: torch.Tensor, n_label: int):

        similarity = candidates @ self.memory[n_label][self.memory_mask[n_label]].t()

        # TODO использовать скоры
        top_indices = similarity.topk(candidates.size(0)).indices

        insert_indices = torch.zeros(candidates.size(0))

        for n_candidate, index_list in enumerate(top_indices):
            for index in index_list:
                if index not in insert_indices:
                    insert_indices[n_candidate] = index
                    break

        insert_indices = insert_indices.long()

        updated_memory = self.momentum * candidates + (1 - self.momentum) * self.memory[n_label][insert_indices]

        self._set_memory(memory=updated_memory, n_label=n_label, insert_indices=insert_indices)

    def _replace_no_updates(self):
        ...

    def _update_memory(self, candidates, n_label):

        no_update_indices = self.memory_indices[n_label][self.memory_n_no_updates[n_label] >= self.max_no_updates]

        if no_update_indices.size(0) > 0:

            no_update_indices = no_update_indices[:candidates.size(0)]

            if candidates.size(0) > no_update_indices.size(0):

                replace_candidates = candidates[:no_update_indices.size(0)]
                update_candidates = candidates[no_update_indices.size(0):]

                # порядок важен, потому что эмбеддинги кандидатов могут быть похожими
                # потому что есть пересечения
                self._update_exist_memory(candidates=update_candidates,
                                          n_label=n_label)
                self._set_memory(memory=replace_candidates,
                                 n_label=n_label,
                                 insert_indices=no_update_indices)

            else:
                self._set_memory(memory=candidates, n_label=n_label, insert_indices=no_update_indices)

        else:
            self._update_exist_memory(candidates=candidates, n_label=n_label)

    def update_memory(self, embeddings, labels):

        label_candidates = self._get_label_candidates(embeddings=embeddings, labels=labels)

        for n_label, candidates in label_candidates.items():

            if len(candidates) > self.max_candidates:
                candidates = random.sample(candidates, self.max_candidates)

            candidates = torch.stack([candidate.mean(dim=0) for candidate in candidates])

            if self.memory_collected_flag[n_label]:
                # отправить на обновление центроид
                self._update_memory(candidates=candidates, n_label=n_label)
            else:
                insert_indices = self.memory_indices[n_label][self.memory_mask[n_label] == False][:candidates.size(0)]
                if insert_indices.size(0) == 0:
                    self.memory_collected_flag[n_label] = True
                    # отправить на обновление центроид
                    self._update_memory(candidates=candidates, n_label=n_label)
                else:
                    # добавление новых центроид
                    candidates = candidates[:insert_indices.size(0)]
                    self._set_memory(memory=candidates, n_label=n_label, insert_indices=insert_indices)

    def update(self, mode: bool = True):
        self.updating = mode

    def forward(self) -> torch.Tensor:

        if self.memory_collected_flag.sum() == self.memory_collected_flag.size(0):
            memory = torch.cat(self.memory)
        else:
            memory = torch.cat([self.memory[n_label][self.memory_mask[n_label]]
                                for n_label in range(self.num_labels)])

        return memory

    def extra_repr(self) -> str:

        description = [
            f'model_dim={self.model_dim}',
            f'num_labels={self.num_labels}',
            f'memory_size_per_label={self.memory_size_per_label}',
            f'momentum={self.momentum}',
            f'label_samples_ratio={self.label_samples_ratio}',
            f'min_samples_per_label={self.min_samples_per_label}',
            f'max_candidates={self.max_candidates}',
            f'max_no_updates={self.max_no_updates}',
        ]

        description = ',\n'.join(description)

        return description


class WtfBootstrapLabelMemoryStorage(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_labels: int,
                 memory_size_per_label: Union[int, List[int]] = 128,
                 label_samples_ratio: float = 0.7,
                 momentum: float = 0.5,
                 min_samples_per_label: int = 3,
                 max_candidates: int = 10,
                 max_no_updates: int = 32):
        super().__init__()

        self.model_dim = model_dim
        self.num_labels = num_labels
        self.memory_size_per_label = self._set_memory_size_per_label(memory_size_per_label)
        self.momentum = momentum
        self.label_samples_ratio = label_samples_ratio
        self.min_samples_per_label = min_samples_per_label
        self.max_candidates = max_candidates
        self.max_no_updates = max_no_updates

        self.updating = True

        self.bounds = self._set_bounds()

        self.memory = nn.Parameter(torch.zeros(sum(self.memory_size_per_label), self.model_dim),
                                   requires_grad=False)
        self.memory_mask = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)).bool(),
                                        requires_grad=False)
        self.memory_collected_flag = nn.Parameter(torch.zeros(self.num_labels).bool(), requires_grad=False)
        # self.memory_norms = nn.Parameter(torch.zeros(sum(self.memory_size_per_label), self.model_dim),
        #                                  requires_grad=False)
        self.memory_n_no_updates = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)),
                                                requires_grad=False)
        self.memory_n_updates = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)),
                                             requires_grad=False)
        self.memory_indices = nn.Parameter(torch.cat(
            [torch.arange(self.memory_size_per_label[n])
             for n in range(self.num_labels)]),
            requires_grad=False
        )

    def _set_memory_size_per_label(self, memory_size_per_label: Union[int, List[int]]):

        if isinstance(memory_size_per_label, list):
            memory_size_per_label = memory_size_per_label
        else:
            memory_size_per_label = [memory_size_per_label for _ in range(self.num_labels)]

        return memory_size_per_label

    def _set_bounds(self):

        bounds = list()

        ticks = [0] + np.cumsum(self.memory_size_per_label).tolist()

        for i in range(len(ticks) - 1):
            bounds.append((ticks[i], ticks[i + 1]))

        return bounds

    def _get_label_candidates(self, embeddings, labels):

        label_candidates = {}
        indices = torch.arange(embeddings.size(0))

        for n_label in range(self.num_labels):

            label_subset = indices[labels == n_label]

            if label_subset.size(0) < self.min_samples_per_label:
                continue

            label_candidates[n_label] = list()

            n_samples = int(label_subset.size(0) * self.label_samples_ratio)

            if not n_samples:
                continue

            for i in range(label_subset.size(0) - n_samples + 1):
                label_candidates[n_label].append(embeddings[label_subset[i:i + n_samples]])

        return label_candidates

    def _set_memory(self,
                    memory: torch.Tensor,
                    insert_indices: torch.Tensor,
                    update_flag: bool = True):

        self.memory[insert_indices] = memory
        self.memory_mask[insert_indices] = torch.ones(insert_indices.size(0)).bool().to(self.memory_mask.device)
        self.memory_n_updates[insert_indices] += 1
        # if update_flag:
        self.memory_n_no_updates += 1
        # self.memory_n_no_updates[insert_indices] = 0

    def _update_exist_memory(self, candidates: torch.Tensor, n_label: int):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_subset = self.memory[lower_bound:upper_bound]
        memory_mask_subset = self.memory_mask[lower_bound:upper_bound]

        similarity = candidates @ memory_subset[memory_mask_subset].t()

        # TODO использовать скоры
        top_indices = similarity.topk(candidates.size(0)).indices

        insert_indices = torch.zeros(candidates.size(0))

        for n_candidate, index_list in enumerate(top_indices):
            for index in index_list:
                if index not in insert_indices:
                    insert_indices[n_candidate] = index
                    break

        insert_indices = insert_indices.long().to(candidates.device)

        updated_memory = self.momentum * candidates + (1 - self.momentum) * memory_subset[insert_indices]

        insert_indices += lower_bound

        self._set_memory(memory=updated_memory, insert_indices=insert_indices)

    def _update_memory(self, candidates, n_label):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_indices_subset = self.memory_indices[lower_bound:upper_bound]
        memory_n_no_updates = self.memory_n_no_updates[lower_bound:upper_bound]

        no_update_indices = memory_indices_subset[memory_n_no_updates > self.max_no_updates] + lower_bound

        if no_update_indices.size(0) > 0:

            no_update_indices = no_update_indices[:candidates.size(0)]

            if candidates.size(0) > no_update_indices.size(0):

                replace_candidates = candidates[:no_update_indices.size(0)]
                update_candidates = candidates[no_update_indices.size(0):]

                # порядок важен, потому что эмбеддинги кандидатов могут быть похожими
                # потому что есть пересечения
                self._update_exist_memory(candidates=update_candidates, n_label=n_label)
                self._set_memory(memory=replace_candidates, insert_indices=no_update_indices, update_flag=False)

            else:
                self._set_memory(memory=candidates, insert_indices=no_update_indices)

        else:
            self._update_exist_memory(candidates=candidates, n_label=n_label)

    def update_memory(self, embeddings, labels):

        label_candidates = self._get_label_candidates(embeddings=embeddings, labels=labels)

        for n_label, candidates in label_candidates.items():

            lower_bound, upper_bound = self.bounds[n_label]

            if len(candidates) > self.max_candidates:
                candidates = random.sample(candidates, self.max_candidates)

            candidates = torch.stack([candidate.mean(dim=0) for candidate in candidates])

            if self.memory_collected_flag[n_label]:
                # отправить на обновление центроид
                self._update_memory(candidates=candidates, n_label=n_label)
            else:
                memory_indices_subset = self.memory_indices[lower_bound:upper_bound]
                memory_mask_subset = self.memory_mask[lower_bound:upper_bound]
                insert_indices = memory_indices_subset[memory_mask_subset == False][:candidates.size(0)]
                if insert_indices.size(0) == 0:
                    self.memory_collected_flag[n_label] = True
                    # отправить на обновление центроид
                    self._update_memory(candidates=candidates, n_label=n_label)
                else:
                    # добавление новых центроид
                    candidates = candidates[:insert_indices.size(0)]
                    insert_indices += lower_bound
                    self._set_memory(memory=candidates, insert_indices=insert_indices)

    def update(self, mode: bool = True):
        self.updating = mode

    def forward(self) -> torch.Tensor:
        return self.memory[self.memory_mask]

    def extra_repr(self) -> str:

        description = [
            f'model_dim={self.model_dim}',
            f'num_labels={self.num_labels}',
            f'memory_size_per_label={self.memory_size_per_label}',
            f'momentum={self.momentum}',
            f'label_samples_ratio={self.label_samples_ratio}',
            f'min_samples_per_label={self.min_samples_per_label}',
            f'max_candidates={self.max_candidates}',
            f'max_no_updates={self.max_no_updates}',
        ]

        description = ',\n'.join(description)

        return description


class BootstrapLabelMemoryStorage(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_labels: int,
                 memory_size_per_label: Union[int, List[int]] = 128,
                 label_samples_ratio: float = 0.7,
                 momentum: float = 0.5,
                 min_samples_per_label: int = 3,
                 max_candidates: int = 10,
                 max_no_updates: int = 32):
        super().__init__()

        self.model_dim = model_dim
        self.num_labels = num_labels
        self.memory_size_per_label = self._set_memory_size_per_label(memory_size_per_label)
        self.momentum = momentum
        self.label_samples_ratio = label_samples_ratio
        self.min_samples_per_label = min_samples_per_label
        self.max_candidates = max_candidates
        self.max_no_updates = max_no_updates

        self.updating = True

        self.bounds = self._set_bounds()

        self.memory = nn.Parameter(torch.zeros(sum(self.memory_size_per_label), self.model_dim),
                                   requires_grad=False)
        self.memory_mask = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)).bool(),
                                        requires_grad=False)
        self.memory_collected_flag = nn.Parameter(torch.zeros(self.num_labels).bool(), requires_grad=False)
        # self.memory_norms = nn.Parameter(torch.zeros(sum(self.memory_size_per_label), self.model_dim),
        #                                  requires_grad=False)
        self.memory_n_no_updates = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)),
                                                requires_grad=False)
        self.memory_n_updates = nn.Parameter(torch.zeros(sum(self.memory_size_per_label)),
                                             requires_grad=False)
        self.memory_indices = nn.Parameter(torch.cat(
            [torch.arange(self.memory_size_per_label[n])
             for n in range(self.num_labels)]),
            requires_grad=False
        )

    def _set_memory_size_per_label(self, memory_size_per_label: Union[int, List[int]]):

        if isinstance(memory_size_per_label, list):
            memory_size_per_label = memory_size_per_label
        else:
            memory_size_per_label = [memory_size_per_label for _ in range(self.num_labels)]

        return memory_size_per_label

    def _set_bounds(self):

        bounds = list()

        ticks = [0] + np.cumsum(self.memory_size_per_label).tolist()

        for i in range(len(ticks) - 1):
            bounds.append((ticks[i], ticks[i + 1]))

        return bounds

    def _get_label_candidates(self, embeddings, labels):

        label_candidates = {}
        indices = torch.arange(embeddings.size(0))

        for n_label in range(self.num_labels):

            label_subset = indices[labels == n_label]

            if label_subset.size(0) < self.min_samples_per_label:
                continue

            label_candidates[n_label] = list()

            n_samples = int(label_subset.size(0) * self.label_samples_ratio)

            if not n_samples:
                continue

            for i in range(label_subset.size(0) - n_samples + 1):
                label_candidates[n_label].append(embeddings[label_subset[i:i + n_samples]])

        return label_candidates

    def _set_memory(self, memory: torch.Tensor, insert_indices: torch.Tensor):

        self.memory[insert_indices] = memory
        self.memory_mask[insert_indices] = torch.ones(insert_indices.size(0)).bool().to(self.memory_mask.device)
        self.memory_n_updates[insert_indices] += 1
        self.memory_n_no_updates += 1
        # self.memory_n_no_updates[insert_indices] = 0

    def _update_exist_memory(self, candidates: torch.Tensor, n_label: int):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_subset = self.memory[lower_bound:upper_bound]
        memory_mask_subset = self.memory_mask[lower_bound:upper_bound]

        similarity = candidates @ memory_subset[memory_mask_subset].t()

        # TODO использовать скоры
        top_indices = similarity.topk(candidates.size(0)).indices

        insert_indices = torch.zeros(candidates.size(0))

        for n_candidate, index_list in enumerate(top_indices):
            for index in index_list:
                if index not in insert_indices:
                    insert_indices[n_candidate] = index
                    break

        insert_indices = insert_indices.long().to(candidates.device)

        updated_memory = self.momentum * candidates + (1 - self.momentum) * memory_subset[insert_indices]

        insert_indices += lower_bound

        self._set_memory(memory=updated_memory, insert_indices=insert_indices)

    def _update_memory(self, candidates, n_label):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_indices_subset = self.memory_indices[lower_bound:upper_bound]
        memory_n_no_updates = self.memory_n_no_updates[lower_bound:upper_bound]

        no_update_indices = memory_indices_subset[memory_n_no_updates >= self.max_no_updates] + lower_bound

        if no_update_indices.size(0) > 0:

            no_update_indices = no_update_indices[:candidates.size(0)]

            if candidates.size(0) > no_update_indices.size(0):

                replace_candidates = candidates[:no_update_indices.size(0)]
                update_candidates = candidates[no_update_indices.size(0):]

                # порядок важен, потому что эмбеддинги кандидатов могут быть похожими
                # потому что есть пересечения
                self._update_exist_memory(candidates=update_candidates, n_label=n_label)
                self._set_memory(memory=replace_candidates, insert_indices=no_update_indices)

            else:
                self._set_memory(memory=candidates, insert_indices=no_update_indices)

        else:
            self._update_exist_memory(candidates=candidates, n_label=n_label)

    def update_memory(self, embeddings, labels):

        label_candidates = self._get_label_candidates(embeddings=embeddings, labels=labels)

        for n_label, candidates in label_candidates.items():

            lower_bound, upper_bound = self.bounds[n_label]

            if len(candidates) > self.max_candidates:
                candidates = random.sample(candidates, self.max_candidates)

            candidates = torch.stack([candidate.mean(dim=0) for candidate in candidates])

            if self.memory_collected_flag[n_label]:
                # отправить на обновление центроид
                self._update_memory(candidates=candidates, n_label=n_label)
            else:
                memory_indices_subset = self.memory_indices[lower_bound:upper_bound]
                memory_mask_subset = self.memory_mask[lower_bound:upper_bound]
                insert_indices = memory_indices_subset[memory_mask_subset == False][:candidates.size(0)]
                if insert_indices.size(0) == 0:
                    self.memory_collected_flag[n_label] = True
                    # отправить на обновление центроид
                    self._update_memory(candidates=candidates, n_label=n_label)
                else:
                    # добавление новых центроид
                    candidates = candidates[:insert_indices.size(0)]
                    insert_indices += lower_bound
                    self._set_memory(memory=candidates, insert_indices=insert_indices)

    def update(self, mode: bool = True):
        self.updating = mode

    def forward(self) -> torch.Tensor:
        return self.memory[self.memory_mask]

    def extra_repr(self) -> str:

        description = [
            f'model_dim={self.model_dim}',
            f'num_labels={self.num_labels}',
            f'memory_size_per_label={self.memory_size_per_label}',
            f'momentum={self.momentum}',
            f'label_samples_ratio={self.label_samples_ratio}',
            f'min_samples_per_label={self.min_samples_per_label}',
            f'max_candidates={self.max_candidates}',
            f'max_no_updates={self.max_no_updates}',
        ]

        description = ',\n'.join(description)

        return description
