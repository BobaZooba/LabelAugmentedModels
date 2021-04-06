from typing import Optional, Tuple, Union, List, Sequence

import torch
from torch import nn
from abc import ABC

import copy
import math
from transformers import AutoModel
from src import io, layers

ENCODER_TYPES: List[str] = [
    'encoder',
    'transformer'
]


class TransformerEncoder(nn.Module):

    def __init__(self,
                 model_dim: int,
                 embedding_dim: int,
                 vocab_size: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 num_layers: int,
                 n_positions: int = 0,
                 n_segments: int = 0,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 zeroing_pad: bool = False,
                 head_dim: Optional[int] = None,
                 activation: str = 'geglu',
                 use_attention_bias: bool = False,
                 shared_relative_positions: bool = True,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 16,
                 use_bias_positions: bool = True,
                 use_fusion_gate: bool = False,
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

        self.embedding_layer = layers.TransformerEmbedding(embedding_dim=model_dim,
                                                           vocab_size=vocab_size,
                                                           n_positions=n_positions,
                                                           token_embedding_dim=embedding_dim,
                                                           n_segments=n_segments,
                                                           dropout=dropout,
                                                           norm_type=norm_type,
                                                           zeroing_pad=zeroing_pad,
                                                           pad_index=pad_index)

        self.layers = nn.ModuleList(modules=[
            layers.TransformerEncoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                head_dim=head_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation,
                use_attention_bias=use_attention_bias,
                use_relative_positions=use_relative_positions,
                max_relative_position=max_relative_position,
                use_bias_positions=use_bias_positions,
                use_fusion_gate=use_fusion_gate)
            for _ in range(num_layers)
        ])

        if shared_relative_positions:
            for n_layer in range(1, num_layers):
                self.layers[n_layer].self_attention.relative_positions \
                    = self.layers[0].self_attention.relative_positions

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        sample.set_pad_mask(pad_index=self.pad_index)

        x = self.embedding_layer(sequence_indices=sample.input.sequence_indices)

        for layer in self.layers:
            x = layer(x, pad_mask=sample.input.pad_mask)[-1]

        sample.output.encoded.append(x)

        return sample


class PreTrainedBert(nn.Module):

    def __init__(self, model_name: str = 'distilbert-base-uncased', num_layers: int = -1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pad_index = self.model.config.pad_token_id
        self.num_layers = num_layers

        if self.num_layers > 0:
            self._prune_layers()

    def _prune_layers(self):

        for encoder_type in ENCODER_TYPES:
            if hasattr(self.model, encoder_type):
                encoder_layers = getattr(self.model, encoder_type).layer[:self.num_layers]
                setattr(self.model, encoder_type, encoder_layers)
                break

        raise ValueError('Not specified encoder_type of model')

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        sample.set_pad_mask(pad_index=self.pad_index)

        output = self.model(input_ids=sample.input.sequence_indices, attention_mask=sample.input.pad_mask)

        sample.output.encoded.append(output)

        return output


class BaseMemoryAugmentedBackbone(ABC, nn.Module):

    def __init__(self):
        super().__init__()

        self.pad_index: int = ...
        self.memory_layer_each_n: int = ...

        self.embedding_layer: nn.Module = ...
        self.layers: nn.ModuleList = ...

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        for layer in self.layers:
            if hasattr(layer, 'storage'):
                layer.storage.extra_cuda(device)

        return self._apply(lambda t: t.cuda(device))

    def _apply_shared(self):

        first_ma_layer = None

        for n_layer in range(1, len(self.layers) + 1):
            if n_layer % self.memory_layer_each_n == 0:
                if first_ma_layer is None:
                    first_ma_layer = n_layer
                else:
                    if hasattr(self, 'shared_memory') and self.shared_memory:
                        self.layers.__dict__['_modules'][f'layer_{n_layer}'].memory_attention.storage \
                            = self.layers.__dict__['_modules'][f'layer_{first_ma_layer}'].memory_attention.storage
                    if hasattr(self, 'shared_encoder') and self.shared_encoder:
                        self.layers.__dict__['_modules'][f'layer_{n_layer}'].memory_attention.encoder \
                            = self.layers.__dict__['_modules'][f'layer_{first_ma_layer}'].memory_attention.encoder
            else:
                if hasattr(self, 'shared_relative_positions') and self.shared_relative_positions:
                    self.layers.__dict__['_modules'][f'layer_{n_layer}'].self_attention.relative_positions \
                        = self.layers.__dict__['_modules']['layer_1'].self_attention.relative_positions

    def updating(self, mode: bool = True):
        for layer in self.layers:
            if hasattr(layer, 'updating'):
                layer.updating(mode)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        sample.set_pad_mask(pad_index=self.pad_index)

        x = self.embedding_layer(sample.input.sequence_indices)

        all_embeddings = tuple()

        for n_layer, layer in enumerate(self.layers):
            n_layer += 1
            if n_layer % self.memory_layer_each_n == 0:
                x, embeddings = layer(x, labels=sample.target, pad_mask=sample.input.pad_mask)
                all_embeddings = all_embeddings + (embeddings,)
            else:
                x = layer(x, sample.input.pad_mask)[-1]

        if all_embeddings[0] is not None:
            all_embeddings = torch.stack(all_embeddings)

        sample.output.encoded.append(x)
        sample.output.embeddings = all_embeddings

        return sample


class MemoryAugmentedTransformerEncoder(BaseMemoryAugmentedBackbone):

    def __init__(self,
                 model_dim: int,
                 num_labels: int,
                 embedding_dim: int,
                 vocab_size: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 num_layers: int,
                 encoder_sizes: Sequence[int],
                 memory_layer_each_n: int = 3,
                 shared_memory: bool = False,
                 shared_encoder: bool = False,
                 bootstrap_storage: bool = True,
                 n_positions: int = 0,
                 n_segments: int = 0,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 zeroing_pad: bool = False,
                 head_dim: Optional[int] = None,
                 activation: str = 'geglu',
                 encoder_dropout: float = 0.15,
                 encoder_activation: str = 'gelu',
                 encoder_norm_type: str = 'bn',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 use_attention_bias: bool = False,
                 shared_relative_positions: bool = True,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 16,
                 use_bias_positions: bool = True,
                 momentum: float = 0.5,
                 scaled_momentum: bool = True,
                 min_samples_per_label: int = 3,
                 enough_samples_per_label: int = 32,
                 memory_size_per_label: Union[int, List[int]] = 128,
                 label_samples_ratio: float = 0.7,
                 max_candidates: int = 10,
                 max_no_updates: int = 32,
                 use_fusion_gate: bool = False,
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index
        self.memory_layer_each_n = memory_layer_each_n

        self.shared_memory = shared_memory
        self.shared_encoder = shared_encoder
        self.shared_relative_positions = shared_relative_positions

        self.embedding_layer = layers.TransformerEmbedding(embedding_dim=model_dim,
                                                           vocab_size=vocab_size,
                                                           n_positions=n_positions,
                                                           token_embedding_dim=embedding_dim,
                                                           n_segments=n_segments,
                                                           dropout=dropout,
                                                           norm_type=norm_type,
                                                           zeroing_pad=zeroing_pad,
                                                           pad_index=pad_index)

        self.layers = nn.ModuleList()

        for n_layer in range(1, num_layers + 1):

            if n_layer % self.memory_layer_each_n == 0:

                if bootstrap_storage:
                    storage = layers.BootstrapLabelMemoryStorage(
                        model_dim=model_dim,
                        num_labels=num_labels,
                        memory_size_per_label=memory_size_per_label,
                        label_samples_ratio=label_samples_ratio,
                        momentum=momentum,
                        scaled_momentum=scaled_momentum,
                        min_samples_per_label=min_samples_per_label,
                        max_candidates=max_candidates,
                        max_no_updates=max_no_updates
                    )
                else:
                    storage = layers.LabelMemoryStorage(
                        model_dim=model_dim,
                        num_labels=num_labels,
                        momentum=momentum,
                        min_samples_per_label=min_samples_per_label,
                        enough_samples_per_label=enough_samples_per_label
                    )

                layer = layers.MemoryAugmentedTransformerEncoderLayer(
                    model_dim=model_dim,
                    storage=storage,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    encoder_sizes=encoder_sizes,
                    head_dim=head_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                    activation=activation,
                    use_bias=use_attention_bias,
                    attention_dropout=dropout,
                    encoder_dropout=encoder_dropout,
                    encoder_activation=encoder_activation,
                    encoder_norm_type=encoder_norm_type,
                    pooling_type=pooling_type,
                    length_scaling=length_scaling,
                    scaling_square_root=scaling_square_root,
                    use_fusion_gate=use_fusion_gate
                )

            else:

                layer = layers.TransformerEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    head_dim=head_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                    activation=activation,
                    use_attention_bias=use_attention_bias,
                    use_relative_positions=use_relative_positions,
                    max_relative_position=max_relative_position,
                    use_bias_positions=use_bias_positions,
                    use_fusion_gate=use_fusion_gate
                )

            self.layers.add_module(f'layer_{n_layer}', layer)

        self._apply_shared()


class MemoryAugmentedPreTrainedBert(BaseMemoryAugmentedBackbone):

    def __init__(self,
                 num_labels: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 num_layers: int,
                 encoder_sizes: Sequence[int],
                 model_name: str = 'distilbert-base-uncased',
                 pre_trained_num_layers: int = -1,
                 memory_layer_each_n: int = 3,
                 shared_memory: bool = False,
                 shared_encoder: bool = False,
                 bootstrap_storage: bool = True,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 head_dim: Optional[int] = None,
                 activation: str = 'geglu',
                 encoder_dropout: float = 0.15,
                 encoder_activation: str = 'gelu',
                 encoder_norm_type: str = 'bn',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 use_attention_bias: bool = False,
                 momentum: float = 0.5,
                 scaled_momentum: bool = True,
                 min_samples_per_label: int = 3,
                 enough_samples_per_label: int = 32,
                 memory_size_per_label: Union[int, List[int]] = 128,
                 label_samples_ratio: float = 0.7,
                 max_candidates: int = 10,
                 max_no_updates: int = 32,
                 use_fusion_gate: bool = False,
                 pad_index: int = 0):
        super().__init__()

        self.model_name = model_name
        self.pre_trained_num_layers = pre_trained_num_layers
        self.memory_layer_each_n = memory_layer_each_n
        self.shared_memory = shared_memory
        self.shared_encoder = shared_encoder
        self.pad_index = pad_index

        model = AutoModel.from_pretrained(self.model_name)
        self.model_dim = copy.deepcopy(model.config.hidden_size)

        encoder_sizes = list(encoder_sizes)
        for i in range(2):
            encoder_sizes[-i] = self.model_dim

        self.embedding_layer = copy.deepcopy(model.embeddings)

        encoder_layers = self._get_encoder_layers(model=model)
        del model

        self.layers = nn.ModuleList()
        n_encoder_layer = 0

        for n_layer in range(1, num_layers + 1):

            if n_layer % self.memory_layer_each_n == 0:

                if bootstrap_storage:
                    storage = layers.BootstrapLabelMemoryStorage(
                        model_dim=self.model_dim,
                        num_labels=num_labels,
                        memory_size_per_label=memory_size_per_label,
                        label_samples_ratio=label_samples_ratio,
                        momentum=momentum,
                        scaled_momentum=scaled_momentum,
                        min_samples_per_label=min_samples_per_label,
                        max_candidates=max_candidates,
                        max_no_updates=max_no_updates
                    )
                else:
                    storage = layers.LabelMemoryStorage(
                        model_dim=self.model_dim,
                        num_labels=num_labels,
                        momentum=momentum,
                        min_samples_per_label=min_samples_per_label,
                        enough_samples_per_label=enough_samples_per_label
                    )

                layer = layers.MemoryAugmentedTransformerEncoderLayer(
                    model_dim=self.model_dim,
                    storage=storage,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    encoder_sizes=encoder_sizes,
                    head_dim=head_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                    activation=activation,
                    use_bias=use_attention_bias,
                    attention_dropout=dropout,
                    encoder_dropout=encoder_dropout,
                    encoder_activation=encoder_activation,
                    encoder_norm_type=encoder_norm_type,
                    pooling_type=pooling_type,
                    length_scaling=length_scaling,
                    scaling_square_root=scaling_square_root,
                    use_fusion_gate=use_fusion_gate
                )

            else:

                layer = encoder_layers[n_encoder_layer]
                n_encoder_layer += 1

            self.layers.add_module(f'layer_{n_layer}', layer)

        self._apply_shared()

    def _get_encoder_layers(self, model: nn.Module):

        for encoder_type in ENCODER_TYPES:

            if hasattr(model, encoder_type):

                encoder_layers = copy.deepcopy(getattr(model, encoder_type).layer)
                if self.pre_trained_num_layers > 0:
                    num_encoder_layers = math.ceil(
                        self.pre_trained_num_layers - self.pre_trained_num_layers / self.memory_layer_each_n
                    )
                else:
                    num_encoder_layers = len(encoder_layers)

                encoder_layers = encoder_layers[:num_encoder_layers]

                return encoder_layers

        raise ValueError('Not specified encoder_type of model')


class GlobalPooling(nn.Module):

    def __init__(self,
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

        self.pooling = layers.GlobalMaskedPooling(pooling_type=pooling_type,
                                                  length_scaling=length_scaling,
                                                  scaling_square_root=scaling_square_root)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        mean_pooled = self.pooling(sample.logits, sample.input.pad_mask)

        sample.output.encoded.append(mean_pooled)

        return sample


class AttentionAggregation(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

        self.pooling = layers.GlobalMaskedPooling(pooling_type=pooling_type,
                                                  length_scaling=length_scaling,
                                                  scaling_square_root=scaling_square_root)

        self.attention_pooling = layers.AttentionPooling(model_dim=model_dim,
                                                         num_heads=num_heads,
                                                         inner_dim=inner_dim,
                                                         dropout=dropout)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        mean_pooled = self.pooling(sample.logits, sample.input.pad_mask)
        attention_pooled = self.attention_pooling(sample.logits, sample.input.pad_mask).squeeze(dim=1)

        x = torch.cat((mean_pooled, attention_pooled), dim=-1)

        sample.output.encoded.append(x)

        return sample


class ResidualAttentionAggregation(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 pad_index: int = 0):
        super().__init__()

        self.pad_index = pad_index

        self.pooling = layers.GlobalMaskedPooling(pooling_type=pooling_type,
                                                  length_scaling=length_scaling,
                                                  scaling_square_root=scaling_square_root)

        self.normalization_layer = layers.get_norm(normalized_shape=model_dim, norm_type=norm_type)

        self.attention_pooling = layers.AttentionPooling(model_dim=model_dim,
                                                         num_heads=num_heads,
                                                         inner_dim=inner_dim,
                                                         dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.output_projection = nn.Linear(in_features=self.attention_pooling.output_dim,
                                           out_features=model_dim)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        mean_pooled = self.pooling(sample.logits, sample.input.pad_mask)

        attention_pooled = self.attention_pooling(self.normalization_layer(sample.logits),
                                                  sample.input.pad_mask)

        attention_pooled = self.output_projection(attention_pooled.squeeze(dim=1))

        attention_pooled = self.dropout(attention_pooled) + mean_pooled

        sample.output.encoded.append(attention_pooled)

        return sample


class Head(nn.Module):

    def __init__(self,
                 sizes: Sequence[int],
                 norm_type: Optional[str] = 'bn',
                 dropout: float = 0.1,
                 activation: Optional[str] = 'gelu',
                 residual_as_possible: bool = True):
        super().__init__()

        self.encoder = layers.MLP(sizes=sizes,
                                  norm_type=norm_type,
                                  dropout=dropout,
                                  activation=activation,
                                  residual_as_possible=residual_as_possible)

    def forward(self, sample: io.ModelIO) -> io.ModelIO:

        x = self.encoder(sample.logits)

        sample.output.encoded.append(x)

        return sample
