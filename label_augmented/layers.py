import math
import random
from abc import ABC
from typing import Optional, Tuple, Union, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def compute_n_parameters(module: nn.Module):
    return sum((p.numel() for p in module.parameters()))


def extra_cuda(module: nn.Module, device: torch.device):
    for children_module in module.children():
        if hasattr(children_module, 'extra_cuda'):
            children_module.extra_cuda(device)
        extra_cuda(children_module, device)


def update_memory(module: nn.Module, mode: bool = True):
    for children_module in module.children():
        if hasattr(children_module, 'update'):
            children_module.update(mode)
        update_memory(children_module, mode)


def embedding_masking(x: Tensor,
                      pad_mask: Tensor,
                      value: float = 0.) -> Tensor:
    x = x.masked_fill((~(pad_mask.bool())).unsqueeze(-1), value)
    return x


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return swish(x)


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)


ACTIVATIONS_MAPPER = {
    'relu': nn.ReLU(),
    're': nn.ReLU(),
    'tanh': nn.Tanh(),
    'tan': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'sigm': nn.Sigmoid(),
    'sig': nn.Sigmoid(),
    'swish': Swish(),
    'swi': Swish(),
    'gelu': GELU(),
    'ge': GELU()
}


class GLU(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation: str = 'relu', shared: bool = False):
        super().__init__()
        self.gate_activation = get_activation_function(activation=activation)
        self.gate = nn.Linear(in_features=in_features, out_features=out_features)
        self.projection = nn.Linear(in_features=in_features, out_features=out_features)
        if shared:
            self.projection = self.gate

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate_activation(self.gate(x)) * self.projection(x)
        return x


def get_activation_function(activation: Optional[str] = None,
                            in_features: Optional[int] = None,
                            out_features: Optional[int] = None):

    if activation is None:
        return nn.Identity()

    if activation.endswith('glu'):

        assert in_features is not None, 'set hidden_dim for activation'

        if out_features is None:
            out_features = in_features

        activation = activation[:-3]

        if not activation:
            activation = 'sigmoid'

        activation_module = GLU(in_features=in_features,
                                out_features=out_features,
                                activation=activation)

    else:

        activation_module = ACTIVATIONS_MAPPER.get(activation, nn.Identity())

    return activation_module


class MaskedLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = super().forward(x)

        x = embedding_masking(x, pad_mask)

        return x


class RMSNorm(nn.Module):
    def __init__(self,
                 normalized_shape: int,
                 partial: float = -1.,
                 eps: float = 1e-8,
                 bias: bool = False):
        """
            Root Mean Square Layer Normalization
        :param normalized_shape: model size
        :param partial: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.normalized_shape = normalized_shape
        self.partial = partial
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.register_parameter('scale', self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(normalized_shape))
            self.register_parameter('offset', self.offset)

    def forward(self, x: Tensor) -> Tensor:

        if self.partial < 0. or self.partial > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.normalized_shape
        else:
            partial_size = int(self.normalized_shape * self.partial)
            partial_x, _ = torch.split(x, [partial_size, self.normalized_shape - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

    def extra_repr(self):
        return f'{self.normalized_shape,}, partial={self.partial}, eps={self.eps}'


NORMS_MAPPER = {
    'batch_norm': nn.BatchNorm1d,
    'bn': nn.BatchNorm1d,
    'layer_norm': nn.LayerNorm,
    'ln': nn.LayerNorm,
    'masked_layer_norm': MaskedLayerNorm,
    'mln': MaskedLayerNorm,
    'rms_norm': RMSNorm,
    'rms': RMSNorm,
}


def get_norm(normalized_shape: int, norm_type: Optional[str] = None):

    normalization_module = NORMS_MAPPER.get(norm_type, nn.LayerNorm)
    normalization_layer = normalization_module(normalized_shape)

    return normalization_layer


class Residual(nn.Module):

    def __init__(self, layer: nn.Module):
        super().__init__()

        self.layer = layer

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:

        if pad_mask is not None:
            return x + self.layer(x, pad_mask)
        else:
            return x + self.layer(x)


class GlobalMaskedPooling(nn.Module):

    POOLING_TYPES = ('mean', 'max')

    def __init__(self,
                 pooling_type: str = 'mean',
                 dim: int = 1,
                 normalize: bool = False,
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__()

        self.pooling_type = pooling_type
        self.dim = dim

        self.normalize = normalize
        self.length_scaling = length_scaling
        self.scaling_square_root = scaling_square_root

        if self.pooling_type == 'max':
            self.mask_value = -float('inf')
        else:
            self.mask_value = 0.

        if self.pooling_type not in self.POOLING_TYPES:
            raise ValueError(f'Available types: {", ".join(self.POOLING_TYPES)}')

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        lengths = pad_mask.sum(self.dim).float()

        x = embedding_masking(x=x, pad_mask=pad_mask, value=self.mask_value)

        if self.pooling_type == 'mean':
            scaling = x.size(self.dim) / lengths
        else:
            scaling = torch.ones(x.size(0))

        if self.length_scaling:
            lengths_factor = lengths
            if self.scaling_square_root:
                lengths_factor = lengths_factor ** 0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.).unsqueeze(-1)

        if self.pooling_type == 'mean':
            x = x.mean(self.dim)
        else:
            x, _ = x.max(self.dim)

        x *= scaling

        if self.normalize:
            x = F.normalize(x)

        return x

    def extra_repr(self) -> str:

        description = [
            f'pooling_type="{self.pooling_type}"',
            f'normalize={self.normalize}',
            f'length_scaling={self.length_scaling}',
            f'scaling_square_root={self.scaling_square_root}',
        ]

        description = ',\n'.join(description)

        return description


class FusionGate(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()

        self.raw_linear = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.hidden_linear = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, raw: Tensor, hidden: Tensor) -> Tensor:
        gate = torch.sigmoid(self.raw_linear(raw) + self.hidden_linear(hidden))

        x = gate * raw + (1 - gate) * hidden

        return x


class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 norm_type: Optional[str] = 'bn',
                 dropout: float = 0.3,
                 activation: Optional[str] = None,
                 residual_as_possible: bool = True):
        super().__init__()

        if residual_as_possible and in_features == out_features:
            self.residual = True
        else:
            self.residual = False

        self.layers = nn.Sequential()

        if norm_type is not None:
            self.layers.add_module('normalization', get_norm(normalized_shape=in_features, norm_type=norm_type))

        if dropout:
            self.layers.add_module('dropout', nn.Dropout(p=dropout))

        if activation and 'glu' in activation:
            self.layers.add_module(activation, get_activation_function(activation=activation,
                                                                       in_features=in_features,
                                                                       out_features=out_features))
        else:
            self.layers.add_module('linear', nn.Linear(in_features=in_features, out_features=out_features))

        if activation and 'glu' not in activation:
            self.layers.add_module('activation', get_activation_function(activation=activation))

    def forward(self, x: Tensor) -> Tensor:

        x_projected = self.layers(x)

        if self.residual:
            x_projected = x_projected + x

        return x_projected

    def extra_repr(self) -> str:
        return f'(residual={self.residual})'


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 sizes: Sequence[int],
                 norm_type: Optional[str] = 'bn',
                 dropout: float = 0.15,
                 activation: Optional[str] = 'relu',
                 residual_as_possible: bool = True,
                 last_layer_activation: Optional[nn.Module] = None,
                 last_layer_dropout: Optional[float] = None,
                 last_layer_residual: bool = False):
        super().__init__()

        self.layers = nn.Sequential()

        for n, i_size in enumerate(range(len(sizes) - 1)):

            if i_size + 2 == len(sizes):
                current_activation = last_layer_activation
                residual = last_layer_residual
                layer_dropout = last_layer_dropout if last_layer_dropout is not None else dropout
            else:
                current_activation = activation
                residual = residual_as_possible
                layer_dropout = dropout

            self.layers.add_module(f'layer_{i_size + 1}',
                                   Linear(in_features=sizes[i_size],
                                          out_features=sizes[i_size + 1],
                                          norm_type=norm_type,
                                          dropout=layer_dropout,
                                          activation=current_activation,
                                          residual_as_possible=residual))

    def forward(self, x):

        x = self.layers(x)

        return x


class ResidualBidirectionalLSTM(nn.Module):

    def __init__(self, model_dim: int, dropout: float = 0.3, num_layers: int = 1):
        super().__init__()

        self.normalization_layer = get_norm(normalized_shape=model_dim, norm_type='mln')

        self.rnn_dropout = nn.Dropout2d(p=dropout)

        self.lstm = nn.LSTM(input_size=model_dim,
                            hidden_size=model_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.projection_dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(in_features=model_dim * 2, out_features=model_dim)

    def forward(self,
                x: Tensor,
                pad_mask: Tensor) -> Tensor:
        residual = x

        x = self.normalization_layer(x, pad_mask)

        x = self.rnn_dropout(x.transpose(1, 2)).transpose(1, 2)

        x_packed = pack_padded_sequence(x,
                                        pad_mask.sum(1),
                                        batch_first=True,
                                        enforce_sorted=False)

        x_packed, _ = self.lstm(x_packed)

        x, _ = pad_packed_sequence(x_packed,
                                   batch_first=True,
                                   total_length=x.size(1))

        x = self.projection_dropout(x)
        x = self.projection(x)

        x = embedding_masking(x=x, pad_mask=pad_mask)

        x = x + residual

        return x


class CausalCNN(nn.Module):

    def __init__(self, model_dim: int, kernel_size: int, output_dim: Optional[int] = None):
        super().__init__()

        output_dim = output_dim if output_dim is not None else model_dim

        self.layer = nn.Conv1d(in_channels=model_dim, out_channels=output_dim, kernel_size=kernel_size)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = F.pad(input=x.transpose(1, 2), pad=[self.layer.kernel_size[0] - 1, 0])

        x = self.layer(x).transpose(1, 2)

        x = torch.relu(x)

        x = embedding_masking(x, pad_mask=pad_mask)

        return x


class IncreasedCausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 inner_dim: int,
                 kernel_size_increase: int,
                 kernel_size_decrease: Optional[int] = None,
                 norm_type: str = 'ln',
                 dropout: float = 0.3):
        super().__init__()

        kernel_size_decrease = kernel_size_decrease if not None else kernel_size_increase

        self.increase_cnn = CausalCNN(model_dim=model_dim, kernel_size=kernel_size_increase, output_dim=inner_dim)
        self.normalization_layer = get_norm(normalized_shape=inner_dim, norm_type=norm_type)
        self.dropout = nn.Dropout2d(p=dropout)
        self.decrease_cnn = CausalCNN(model_dim=inner_dim, kernel_size=kernel_size_decrease, output_dim=model_dim)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = self.increase_cnn(x, pad_mask)

        # (batch_size, seq_len, model_dim) -> (seq_len, batch_size, model_dim)
        x = self.normalization_layer(x).transpose(0, 1)

        # (seq_len, batch_size, model_dim) -> (batch_size, model_dim, seq_len)
        # (batch_size, model_dim, seq_len) -> (batch_size, seq_len, model_dim)
        x = self.dropout(x.permute(1, 2, 0)).transpose(1, 2)

        x = embedding_masking(x, pad_mask)

        x = self.decrease_cnn(x, pad_mask)

        return x


class ParallelCausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 kernel_sizes: List[Union[int, Tuple[int, int]]],
                 inner_dim: Optional[int] = None,
                 norm_type: str = 'ln',
                 dropout: float = 0.3):
        super().__init__()

        self.layers = nn.ModuleList()

        for ks in kernel_sizes:

            if isinstance(ks, tuple):

                inner_dim = inner_dim if inner_dim is not None else model_dim

                layer = IncreasedCausalCNN(model_dim=model_dim,
                                           inner_dim=inner_dim,
                                           kernel_size_increase=ks[0],
                                           kernel_size_decrease=ks[1],
                                           dropout=dropout)
            elif isinstance(ks, int):
                layer = CausalCNN(model_dim=model_dim, kernel_size=ks)

            self.layers.append(layer)

        self.normalization_layer = get_norm(normalized_shape=model_dim, norm_type=norm_type)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:

        x_cnn = [layer(x, pad_mask=pad_mask) for layer in self.layers]

        for part in x_cnn:
            x += part

        # (batch_size, seq_len, model_dim) -> (seq_len, batch_size, model_dim)
        x = self.normalization_layer(x).transpose(0, 1)

        # (seq_len, batch_size, model_dim) -> (batch_size, model_dim, seq_len)
        # (batch_size, model_dim, seq_len) -> (batch_size, seq_len, model_dim)
        x = self.dropout(x.permute(1, 2, 0)).transpose(1, 2)

        x = embedding_masking(x, pad_mask)

        return x


class AttentionPooling(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        inner_dim = model_dim if inner_dim is None else inner_dim

        self.key_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)
        self.value_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)

        self.pooling_projection = nn.Linear(in_features=inner_dim, out_features=num_heads, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.scaling: float = inner_dim ** 0.5
        self.output_dim: int = inner_dim * num_heads

    def forward(self, x, pad_mask):

        key = self.key_projection(x)
        value = self.value_projection(x)

        key /= self.scaling

        attention_scores = self.pooling_projection(key).transpose(1, 2)

        attention_scores = attention_scores.masked_fill(~(pad_mask.bool()).unsqueeze(1), -float('inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        x = torch.bmm(attention_scores, value)

        x = x.view(x.size(0), -1)

        return x


class FactorizedEmbedding(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 token_embedding_dim: int,
                 pad_index: int = 0,
                 zeroing_pad: bool = False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.zeroing_pad = zeroing_pad

        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=token_embedding_dim,
                                            padding_idx=pad_index)

        self.projection = nn.Linear(in_features=token_embedding_dim, out_features=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding_layer(x)
        emb = self.projection(emb)

        if self.zeroing_pad:
            pad_mask = x != self.embedding_layer.padding_idx
            emb = embedding_masking(emb, pad_mask)

        return emb


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 n_positions: int = 0,
                 token_embedding_dim: Optional[int] = None,
                 n_segments: int = 3,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 zeroing_pad: bool = False,
                 pad_index: int = 0):
        super().__init__()

        self.zeroing_pad = zeroing_pad

        self.embedding_dim = embedding_dim
        self.pad_index = pad_index

        self.token_embedding_dim = token_embedding_dim if token_embedding_dim is not None \
            else self.embedding_dim

        self.scaling = embedding_dim ** 0.5

        if self.token_embedding_dim != self.embedding_dim:
            self.token_embedding = FactorizedEmbedding(num_embeddings=vocab_size,
                                                       embedding_dim=self.embedding_dim,
                                                       token_embedding_dim=self.token_embedding_dim,
                                                       pad_index=self.pad_index,
                                                       zeroing_pad=False)
        else:
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=self.pad_index)

        if n_segments > 1:
            self.segment_embedding = nn.Embedding(num_embeddings=n_segments + 1,
                                                  embedding_dim=self.embedding_dim,
                                                  padding_idx=self.pad_index)
        else:
            self.segment_embedding = None

        if n_positions > 1:
            self.positional_embedding = nn.Embedding(num_embeddings=n_positions + 1,
                                                     embedding_dim=self.embedding_dim,
                                                     padding_idx=self.pad_index)
        else:
            self.positional_embedding = None

        self.normalization_layer = get_norm(normalized_shape=self.embedding_dim, norm_type=norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                sequence_indices: Tensor,
                position_indices: Optional[Tensor] = None,
                segment_indices: Optional[Tensor] = None) -> Tensor:
        """
        :param sequence_indices: [sequence_length, batch_size]
        :param position_indices: [sequence_length, batch_size]
        :param segment_indices: [sequence_length, batch_size]
        :return: [sequence_length, batch_size, model_dim]
        """

        emb = self.token_embedding(sequence_indices) * self.scaling

        if self.positional_embedding is not None:

            if position_indices is None:
                position_indices = torch.arange(1, sequence_indices.size(1) + 1)
                position_indices = position_indices.unsqueeze(0).repeat(sequence_indices.size(0), 1)
                position_indices = position_indices.to(sequence_indices.device)

            position_emb = self.positional_embedding(position_indices)
            emb += position_emb

        if self.segment_embedding is not None:

            if segment_indices is None:
                segment_indices = torch.ones_like(sequence_indices).to(sequence_indices.device)

            segment_emb = self.segment_embedding(segment_indices)
            emb += segment_emb

        emb = self.dropout(self.normalization_layer(emb))

        if self.zeroing_pad:
            pad_mask = (sequence_indices != self.pad_index).long()
            emb = embedding_masking(emb, pad_mask)

        return emb


class RelativeAttentionPositions(nn.Module):

    def __init__(self,
                 head_dim: int,
                 max_relative_position: int,
                 num_heads: Optional[int] = None,
                 use_bias: bool = True):
        super().__init__()

        self.max_relative_position = max_relative_position
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.relative_positions_keys = nn.Embedding(num_embeddings=self.max_relative_position * 2 + 1,
                                                    embedding_dim=head_dim)

        if self.use_bias and self.num_heads is not None:
            self.bias = nn.Parameter(torch.rand(1, self.num_heads, 1, 1))
            nn.init.xavier_uniform_(self.bias)

        self.relative_positions_values = nn.Embedding(num_embeddings=self.max_relative_position * 2 + 1,
                                                      embedding_dim=head_dim)

    def _generate_relative_positions_embeddings(self,
                                                length: int,
                                                is_key: bool = True) -> Tensor:
        """
        :param length: sequence_length
        :param is_key: use key positions
        :return relative_position_embeddings: [sequence_length, sequence_length, head_dim]
        """

        if is_key:
            embedding_layer = self.relative_positions_keys
        else:
            embedding_layer = self.relative_positions_values

        range_vector = torch.arange(length)
        distance_matrix = range_vector[None, :] - range_vector[:, None]

        distance_matrix_clipped = torch.clamp(distance_matrix,
                                              -self.max_relative_position,
                                              self.max_relative_position)

        final_matrix = (distance_matrix_clipped + self.max_relative_position)
        final_matrix = final_matrix.long().to(embedding_layer.weight.device)

        relative_positions_embeddings = embedding_layer(final_matrix)

        return relative_positions_embeddings

    def forward(self,
                tensor: Tensor,
                is_key: bool = True) -> Tensor:
        """
        :param tensor: [batch_size, num_heads, sequence_length, head_dim or sequence_length]
        :param is_key: use key positions
        :return:
        """

        batch_size, num_heads, sequence_length, fourth_dim = tensor.size()

        relative_position_embeddings = self._generate_relative_positions_embeddings(length=sequence_length,
                                                                                    is_key=is_key)

        if sequence_length != fourth_dim:
            relative_position_embeddings = relative_position_embeddings.transpose(-1, -2)

        tensor = tensor.permute(2, 0, 1, 3)
        tensor = tensor.reshape(sequence_length, num_heads * batch_size, -1)

        relative_attention_scores = torch.matmul(tensor, relative_position_embeddings)
        relative_attention_scores = relative_attention_scores.view(sequence_length, batch_size, num_heads, -1)
        relative_attention_scores = relative_attention_scores.permute(1, 2, 0, 3)

        if is_key and self.use_bias:
            relative_attention_scores += self.bias

        return relative_attention_scores

    def extra_repr(self) -> str:
        return f'(keys_bias={self.use_bias})'


class BaseAttention(nn.Module, ABC):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias

        if head_dim is None:
            self.head_dim = model_dim // num_heads
            self.layer_dim = model_dim
        else:
            self.head_dim = head_dim
            self.layer_dim = self.head_dim * self.num_heads

        self.scaling = self.head_dim ** 0.5

    def _split_heads(self,
                     embeddings: Tensor,
                     batch_size: int,
                     sequence_length: int) -> Tensor:
        """
        From [batch_size * self.num_heads, sequence_length, sequence_length]
        To [batch_size, self.num_heads, sequence_length, sequence_length]
        """
        return embeddings.view(batch_size, self.num_heads, sequence_length, sequence_length)

    def _join_heads(self,
                    embeddings: Tensor,
                    batch_size: int,
                    sequence_length: int) -> Tensor:
        """
        From [batch_size, self.num_heads, sequence_len, sequence_len]
        To [batch_size * self.num_heads, sequence_len, sequence_len]
        """
        return embeddings.view(batch_size * self.num_heads, sequence_length, sequence_length)


class MultiHeadSelfAttention(BaseAttention):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 8,
                 use_bias_positions: bool = True):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_bias=use_bias
        )

        self.in_projection = nn.Linear(self.model_dim, 3 * self.layer_dim, bias=self.use_bias)

        self.out_projection = nn.Linear(self.layer_dim, self.model_dim, bias=self.use_bias)

        self.dropout_layer = nn.Dropout(self.dropout)

        if use_relative_positions:
            self.relative_positions = RelativeAttentionPositions(head_dim=self.head_dim,
                                                                 max_relative_position=max_relative_position,
                                                                 num_heads=self.num_heads,
                                                                 use_bias=use_bias_positions)
        else:
            self.relative_positions = None

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        x = x.transpose(0, 1)

        sequence_length, batch_size, model_dim = x.size()

        query, key, value = self.in_projection(x).chunk(3, dim=-1)

        # [batch_size * self.num_heads, sequence_len, self.head_dim]
        query = query.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)
        key = key.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)
        value = value.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        query /= self.scaling

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = torch.bmm(query, key.transpose(1, 2))

        if self.num_heads > 1:
            # [batch_size, self.num_heads, sequence_length, sequence_length]
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        if self.relative_positions is not None:
            query = query.transpose(0, 1).view(sequence_length, batch_size, self.num_heads, self.head_dim)
            query = query.permute(1, 2, 0, 3)
            relative_attention_scores_keys = self.relative_positions(query, is_key=True)
            attention_scores += relative_attention_scores_keys

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            if self.num_heads > 1:
                # [batch_size, sequence_length, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if pad_mask is not None:
            # [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
            pad_mask = ~(pad_mask.bool()).unsqueeze(1)

            if self.num_heads > 1:
                # [batch_size, 1, sequence_length] -> [batch_size, 1, 1, sequence_length]
                pad_mask = pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                pad_mask,
                -float('inf'),
            )

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, sequence_length]
            attention_scores = self._join_heads(attention_scores, batch_size, sequence_length)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout_layer(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_length, sequence_length]
        # value = [batch_size * self.num_heads, sequence_length, self.head_dim]
        # [batch_size * self.num_heads, sequence_length, self.head_dim]
        attention_output = torch.bmm(attention_scores, value)

        if self.relative_positions is not None:
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)
            relative_attention_scores_values = self.relative_positions(attention_scores,
                                                                       is_key=False)
            attention_output = attention_output.view(batch_size, self.num_heads,
                                                     sequence_length, self.head_dim)
            attention_output += relative_attention_scores_values
            attention_output = attention_output.view(batch_size * self.num_heads,
                                                     sequence_length, self.head_dim)

        # [sequence_length, batch_size, model_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length,
                                                                              batch_size,
                                                                              self.layer_dim)

        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            if self.num_heads > 1 and len(attention_scores) == 3:
                # [batch_size, self.num_heads, sequence_length, sequence_length]
                attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        attention_output = attention_output.transpose(0, 1)

        return attention_scores, attention_output


class RandomSynthesizedMultiHeadSelfAttention(BaseAttention):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 max_sequence_length: int = 32,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 8,
                 use_bias_positions: bool = True):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_bias=use_bias
        )

        self.attention_scores = nn.Parameter(torch.rand(1,
                                                        self.num_heads,
                                                        max_sequence_length,
                                                        max_sequence_length))
        nn.init.xavier_uniform_(self.attention_scores)

        self.in_projection = nn.Linear(self.model_dim, self.layer_dim, bias=self.use_bias)

        self.out_projection = nn.Linear(self.layer_dim, self.model_dim, bias=self.use_bias)

        self.dropout_layer = nn.Dropout(dropout)

        if use_relative_positions:
            self.relative_positions = RelativeAttentionPositions(head_dim=self.head_dim,
                                                                 max_relative_position=max_relative_position,
                                                                 num_heads=self.num_heads,
                                                                 use_bias=use_bias_positions)
        else:
            self.relative_positions = None

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        x = x.transpose(0, 1)

        sequence_length, batch_size, _ = x.size()

        value = self.in_projection(x)

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, self.head_dim]
            value = value.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)

        value = value.transpose(0, 1)

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = self.attention_scores.repeat(batch_size, 1, 1, 1)
        attention_scores = attention_scores[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.view(batch_size * self.num_heads,
                                                 sequence_length, sequence_length)

        if self.num_heads > 1:
            # [batch_size, self.num_heads, sequence_length, sequence_length]
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            if self.num_heads > 1:
                # [batch_size, sequence_length, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if pad_mask is not None:
            # [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
            pad_mask = ~(pad_mask.bool()).unsqueeze(1)

            if self.num_heads > 1:
                # [batch_size, 1, sequence_length] -> [batch_size, 1, 1, sequence_length]
                pad_mask = pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                pad_mask,
                -float('inf'),
            )

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_length, sequence_length]
            attention_scores = self._join_heads(attention_scores, batch_size, sequence_length)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_length, sequence_length]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout_layer(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_length, sequence_length]
        # value = [batch_size * self.num_heads, sequence_length, self.head_dim]
        # [batch_size * self.num_heads, sequence_length, self.head_dim]
        attention_output = torch.bmm(attention_scores, value)

        if self.relative_positions is not None:
            attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)
            relative_attention_scores_values = self.relative_positions(attention_scores,
                                                                       is_key=False)
            attention_output = attention_output.view(batch_size, self.num_heads,
                                                     sequence_length, self.head_dim)
            attention_output += relative_attention_scores_values
            attention_output = attention_output.view(batch_size * self.num_heads,
                                                     sequence_length, self.head_dim)

        # [sequence_length, batch_size, model_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length,
                                                                              batch_size,
                                                                              self.layer_dim)

        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            if self.num_heads > 1 and len(attention_scores) == 3:
                # [batch_size, self.num_heads, sequence_length, sequence_length]
                attention_scores = self._split_heads(attention_scores, batch_size, sequence_length)

        attention_output = attention_output.transpose(0, 1)

        return attention_output, attention_scores


class GatedMultiHeadAttention(nn.Module):

    def __init__(self,
                 model_dim: int,
                 first_attention: nn.Module,
                 second_attention: nn.Module,
                 gate_activation: str = 'sigmoid'):
        super().__init__()

        self.gate_projection = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.activation = get_activation_function(activation=gate_activation)
        self.first_attention = first_attention
        self.second_attention = second_attention

        if self.first_attention.relative_positions is not None:
            self.relative_positions = self.first_attention.relative_positions

            if self.second_attention.relative_positions is not None:
                self.second_attention.relative_positions = self.relative_positions

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                need_weights: bool = False) -> Tuple[Tensor,
                                                     Tuple[Optional[Tensor],
                                                           Optional[Tensor]]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        gate = self.activation(self.gate_projection(x))

        first_attention_output, first_attention_scores = self.first_attention(x,
                                                                              pad_mask,
                                                                              attention_mask,
                                                                              need_weights)

        second_attention_output, second_attention_scores = self.second_attention(x,
                                                                                 pad_mask,
                                                                                 attention_mask,
                                                                                 need_weights)

        attention_output = gate * first_attention_output + (1. - gate) * second_attention_output

        return attention_output, (first_attention_scores, second_attention_scores)


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 increased_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'geglu',
                 norm_type: Optional[str] = 'rms'):
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('increase', nn.Sequential())
        self.layers.add_module('decrease', nn.Sequential())

        if 'glu' in activation:
            self.layers.increase.add_module('glu',
                                            get_activation_function(activation=activation,
                                                                    in_features=model_dim,
                                                                    out_features=increased_dim))
        else:
            self.layers.increase.add_module('linear', nn.Linear(in_features=model_dim, out_features=increased_dim))
            self.layers.increase.add_module('activation', get_activation_function(activation=activation,
                                                                                  in_features=model_dim,
                                                                                  out_features=increased_dim))

        if norm_type is not None:
            self.layers.increase.add_module('normalization', get_norm(normalized_shape=increased_dim,
                                                                      norm_type=norm_type))

        self.layers.decrease.add_module('dropout', nn.Dropout(dropout))
        self.layers.decrease.add_module('linear', nn.Linear(in_features=increased_dim, out_features=model_dim))
        self.layers.decrease.add_module('output_dropout', nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [*, *, model_dim]
        :return: [*, *, model_dim]
        """

        x = self.layers(x)

        return x


class LabelMemoryStorage(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_labels: int,
                 momentum: float = 0.2,
                 min_samples_per_label: int = 3,
                 enough_samples_per_label: int = 16):
        super().__init__()

        self.model_dim = model_dim
        self.num_labels = num_labels
        self.min_samples_per_label = min_samples_per_label
        self.enough_samples_per_label = enough_samples_per_label

        self.memory = nn.Parameter(data=torch.zeros(self.num_labels, self.model_dim), requires_grad=False)
        self.memory_mask = nn.Parameter(data=torch.zeros(self.num_labels).bool(), requires_grad=False)

        self.momentum = momentum

    def _compute_momentum(self, num_samples: int) -> float:
        ratio = min(1., num_samples / self.enough_samples_per_label)
        momentum = ratio * self.momentum
        return momentum

    def update_memory(self, embeddings: Tensor, labels: Tensor):

        for n_class in range(self.memory.size(0)):

            memory_vectors = embeddings[labels == n_class]

            if memory_vectors.size(0) < self.min_samples_per_label:
                continue

            batch_memory = memory_vectors.mean(dim=0)

            if not self.memory_mask[n_class]:
                momentum = self._compute_momentum(num_samples=memory_vectors.size(0))
                self.memory[n_class] = (1. - momentum) * self.memory[n_class] + momentum * batch_memory
            else:
                self.memory[n_class] = batch_memory

            self.memory_mask[n_class] = True

    def forward(self):
        return self.memory[self.memory_mask]

    def extra_repr(self) -> str:

        description = [
            f'model_dim={self.model_dim}',
            f'num_labels={self.num_labels}',
            f'momentum={self.momentum}',
            f'min_samples_per_label={self.min_samples_per_label}',
            f'enough_samples_per_label={self.enough_samples_per_label}'
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
                 scaled_momentum: bool = True,
                 min_samples_per_label: int = 3,
                 max_candidates: int = 10,
                 max_no_updates: int = 32):
        super().__init__()

        self.model_dim = model_dim
        self.num_labels = num_labels
        self.memory_size_per_label = self._set_memory_size_per_label(memory_size_per_label)
        self.momentum = momentum
        self.scaled_momentum = scaled_momentum
        self.label_samples_ratio = label_samples_ratio
        self.min_samples_per_label = min_samples_per_label
        self.max_candidates = max_candidates
        self.max_no_updates = max_no_updates

        self.updating = True

        self.bounds = self._set_bounds()

        self.memory = torch.zeros(sum(self.memory_size_per_label), self.model_dim)
        self.memory_norms = torch.zeros(sum(self.memory_size_per_label))
        self.memory_mask = torch.zeros(sum(self.memory_size_per_label)).bool()
        self.memory_collected_flag = torch.zeros(self.num_labels).bool()
        self.memory_n_no_updates = torch.zeros(sum(self.memory_size_per_label))
        self.memory_n_updates = torch.zeros(sum(self.memory_size_per_label))
        self.memory_indices = torch.cat([torch.arange(self.memory_size_per_label[n])
                                         for n in range(self.num_labels)])

    @property
    def _to_cuda_attributes(self):
        attributes = [
            'memory',
            'memory_norms',
            'memory_mask',
            'memory_collected_flag',
            'memory_n_no_updates',
            'memory_n_updates',
            'memory_indices'
        ]
        return attributes

    def extra_cuda(self, device: Optional[Union[int, torch.device]] = None):

        for attribute_name in self._to_cuda_attributes:
            setattr(self, attribute_name, getattr(self, attribute_name).to(device))

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

    def _set_memory(self, memory: Tensor, insert_indices: Tensor):

        self.memory[insert_indices] = memory
        self.memory_norms[insert_indices] = memory.norm(dim=1)
        self.memory_mask[insert_indices] = torch.ones(insert_indices.size(0)).bool().to(self.memory_mask.device)
        self.memory_n_updates[insert_indices] += 1
        self.memory_n_no_updates[insert_indices] = -1

    def _update_exist_memory(self, candidates: Tensor, n_label: int):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_subset = self.memory[lower_bound:upper_bound].to(candidates.device)
        memory_norms_subset = self.memory_norms[lower_bound:upper_bound].unsqueeze(1).to(candidates.device)
        memory_mask_subset = self.memory_mask[lower_bound:upper_bound].to(candidates.device)

        candidates_norms = candidates.norm(dim=1).unsqueeze(1)

        similarity = torch.matmul(candidates / candidates_norms,
                                  (memory_subset[memory_mask_subset] / memory_norms_subset).t())

        scores, top_indices = similarity.topk(candidates.size(0))

        best_scores = torch.zeros(candidates.size(0))
        insert_indices = torch.zeros(candidates.size(0)) - 1

        for i_column in range(top_indices.size(1)):
            for i_row in range(top_indices.size(0)):
                index = top_indices[i_row, i_column].item()
                if insert_indices[i_row] == -1 and index not in insert_indices:
                    insert_indices[i_row] = index
                    best_scores[i_row] = scores[i_row, i_column]
            if set(insert_indices) == len(insert_indices):
                break

        best_scores = best_scores.unsqueeze(1).to(candidates.device)
        insert_indices = insert_indices.long().to(candidates.device)

        if self.scaled_momentum:
            momentum = (1. - abs(best_scores)) * self.momentum
        else:
            momentum = self.momentum

        updated_memory = momentum * candidates + (1. - momentum) * memory_subset[insert_indices]

        insert_indices = insert_indices + lower_bound

        self._set_memory(memory=updated_memory.cpu(), insert_indices=insert_indices.cpu())

    def _update_memory(self, candidates, n_label):

        lower_bound, upper_bound = self.bounds[n_label]
        memory_indices_subset = self.memory_indices[lower_bound:upper_bound].to(candidates.device)
        memory_n_no_updates = self.memory_n_no_updates[lower_bound:upper_bound].to(candidates.device)

        no_update_indices = memory_indices_subset[memory_n_no_updates >= self.max_no_updates] + lower_bound

        if no_update_indices.size(0) > 0:

            no_update_indices = no_update_indices[:candidates.size(0)]

            if candidates.size(0) > no_update_indices.size(0):

                replace_candidates = candidates[:no_update_indices.size(0)]
                update_candidates = candidates[no_update_indices.size(0):]

                # порядок важен, потому что эмбеддинги кандидатов могут быть похожими
                # потому что есть пересечения
                self._update_exist_memory(candidates=update_candidates, n_label=n_label)
                self._set_memory(memory=replace_candidates.cpu(), insert_indices=no_update_indices.cpu())

            else:
                self._set_memory(memory=candidates.cpu(), insert_indices=no_update_indices.cpu())

        else:
            self._update_exist_memory(candidates=candidates, n_label=n_label)

    def update_memory(self, embeddings, labels):

        label_candidates = self._get_label_candidates(embeddings=embeddings, labels=labels)

        for n_label, candidates in label_candidates.items():

            lower_bound, upper_bound = self.bounds[n_label]

            if len(candidates) > self.max_candidates:
                candidates = random.sample(candidates, self.max_candidates)

            candidates = torch.stack([candidate.mean(dim=0) for candidate in candidates]).to(embeddings.device)

            if self.memory_collected_flag[n_label]:
                # отправить на обновление центроид
                self._update_memory(candidates=candidates, n_label=n_label)
            else:
                memory_indices_subset = self.memory_indices[lower_bound:upper_bound].to(embeddings.device)
                memory_mask_subset = self.memory_mask[lower_bound:upper_bound].to(embeddings.device)
                insert_indices = memory_indices_subset[memory_mask_subset == False][:candidates.size(0)]
                if insert_indices.size(0) == 0:
                    self.memory_collected_flag[n_label] = True
                    # отправить на обновление центроид
                    self._update_memory(candidates=candidates, n_label=n_label)
                else:
                    # добавление новых центроид
                    candidates = candidates[:insert_indices.size(0)]
                    insert_indices += lower_bound
                    self._set_memory(memory=candidates.cpu(), insert_indices=insert_indices.cpu())

        self.memory_n_no_updates += 1

    def update(self, mode: bool = True):
        self.updating = mode

    def forward(self) -> Tensor:
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


class MultiHeadMemoryAttention(BaseAttention):

    def __init__(self,
                 model_dim: int,
                 storage: nn.Module,
                 num_heads: int,
                 encoder_sizes: Sequence[int],
                 head_dim: Optional[int] = None,
                 use_bias: bool = False,
                 attention_dropout: float = 0.1,
                 encoder_dropout: float = 0.15,
                 activation: str = 'gelu',
                 norm_type: str = 'bn',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True):
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            use_bias=use_bias
        )

        self.storage = storage

        self.norm_type = norm_type
        self.activation = activation

        self.updating = True
        self.need_embeddings = True

        self.pooling = GlobalMaskedPooling(pooling_type=pooling_type,
                                           length_scaling=length_scaling,
                                           scaling_square_root=scaling_square_root)

        self.encoder = MultiLayerPerceptron(sizes=encoder_sizes,
                                            norm_type=self.norm_type,
                                            dropout=encoder_dropout,
                                            activation=self.activation)

        self.query_projection = nn.Linear(in_features=self.model_dim, out_features=self.layer_dim)
        self.key_projection = self._get_head()
        self.value_projection = self._get_head()

        self.dropout_layer = nn.Dropout(self.dropout)

        self.output_projection = nn.Linear(in_features=self.layer_dim, out_features=self.model_dim)

    def _get_head(self) -> nn.Module:

        head = nn.Sequential(
            get_activation_function(activation=self.activation),
            nn.Linear(in_features=self.model_dim, out_features=self.layer_dim, bias=self.use_bias)
        )

        return head

    def update(self, mode: bool = True):
        self.storage.updating = mode

    def forward(self,
                x: Tensor,
                pad_mask: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        if self.updating and labels is not None:
            embeddings = self.encoder(self.pooling(x, pad_mask))
            with torch.no_grad():
                self.storage.update_memory(embeddings, labels)
        elif not self.updating and self.need_embeddings:
            embeddings = self.encoder(self.pooling(x, pad_mask))
        else:
            embeddings = None

        memory = self.storage().to(x.device)

        batch_size, sequence_length, _ = x.size()
        n_memory = memory.size(0)

        query = self.query_projection(x)
        key = self.key_projection(memory).unsqueeze(0).repeat(batch_size, 1, 1)
        value = self.value_projection(memory).unsqueeze(0).repeat(batch_size, 1, 1)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, n_memory, self.num_heads, self.head_dim)
        value = value.view(batch_size, n_memory, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = query.contiguous().view(batch_size * self.num_heads, sequence_length, self.head_dim)
        key = key.contiguous().view(batch_size * self.num_heads, n_memory, self.head_dim)
        value = value.contiguous().view(batch_size * self.num_heads, n_memory, self.head_dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2))

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        attention_distribution = torch.softmax(attention_scores, dim=-1, dtype=tensor_type)
        attention_distribution = self.dropout_layer(attention_distribution)

        attention_output = torch.bmm(attention_distribution, value)

        attention_output = attention_output.view(batch_size, self.num_heads,
                                                 sequence_length, self.head_dim)

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, sequence_length, -1)

        attention_output = self.output_projection(attention_output)

        return attention_output, embeddings


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 activation: str = 'geglu',
                 use_attention_bias: bool = False,
                 use_relative_positions: bool = True,
                 max_relative_position: int = 16,
                 use_bias_positions: bool = True,
                 use_fusion_gate: bool = False):
        super().__init__()

        self.norm_attention = get_norm(normalized_shape=model_dim, norm_type=norm_type)

        self.self_attention = MultiHeadSelfAttention(model_dim=model_dim,
                                                     num_heads=num_heads,
                                                     head_dim=head_dim,
                                                     dropout=dropout,
                                                     use_bias=use_attention_bias,
                                                     use_relative_positions=use_relative_positions,
                                                     max_relative_position=max_relative_position,
                                                     use_bias_positions=use_bias_positions)

        self.dropout_attention = nn.Dropout(dropout)

        self.fusion_gate = FusionGate(model_dim=model_dim) if use_fusion_gate else None

        self.norm_feed_forward = get_norm(normalized_shape=model_dim, norm_type=norm_type)

        self.position_wise_feed_forward = PositionWiseFeedForwardLayer(model_dim=model_dim,
                                                                       increased_dim=feed_forward_dim,
                                                                       dropout=dropout,
                                                                       activation=activation)

        self.dropout_feed_forward = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param attention_mask: [batch_size, sequence_length, sequence_length]
        :return: [batch_size, sequence_length, model_dim]
        """

        hidden = self.norm_attention(x)

        attention_scores, hidden = self.self_attention(x=hidden,
                                                       pad_mask=pad_mask,
                                                       attention_mask=attention_mask)

        hidden = self.dropout_attention(hidden)

        if self.fusion_gate is not None:
            x = self.fusion_gate(x, hidden)
        else:
            x = x + hidden

        hidden = self.norm_feed_forward(x)

        hidden = self.position_wise_feed_forward(hidden)

        x = x + self.dropout_feed_forward(hidden)

        return attention_scores, x


class MemoryAugmentedTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 storage: nn.Module,
                 num_heads: int,
                 feed_forward_dim: int,
                 encoder_sizes: Sequence[int],
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = 'rms',
                 activation: str = 'geglu',
                 use_bias: bool = False,
                 attention_dropout: float = 0.1,
                 encoder_dropout: float = 0.15,
                 encoder_activation: str = 'gelu',
                 encoder_norm_type: str = 'bn',
                 pooling_type: str = 'mean',
                 length_scaling: bool = True,
                 scaling_square_root: bool = True,
                 use_fusion_gate: bool = False):
        super().__init__()

        self.norm_attention = get_norm(normalized_shape=model_dim, norm_type=norm_type)

        self.memory_attention = MultiHeadMemoryAttention(
            model_dim=model_dim,
            storage=storage,
            num_heads=num_heads,
            encoder_sizes=encoder_sizes,
            head_dim=head_dim,
            use_bias=use_bias,
            attention_dropout=attention_dropout,
            encoder_dropout=encoder_dropout,
            activation=encoder_activation,
            norm_type=encoder_norm_type,
            pooling_type=pooling_type,
            length_scaling=length_scaling,
            scaling_square_root=scaling_square_root
        )

        self.dropout_attention = nn.Dropout(dropout)

        self.fusion_gate = FusionGate(model_dim=model_dim) if use_fusion_gate else None

        self.norm_feed_forward = get_norm(normalized_shape=model_dim, norm_type=norm_type)

        self.position_wise_feed_forward = PositionWiseFeedForwardLayer(model_dim=model_dim,
                                                                       increased_dim=feed_forward_dim,
                                                                       dropout=dropout,
                                                                       activation=activation)

        self.dropout_feed_forward = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                pad_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param pad_mask: [batch_size, sequence_length]
        :param labels: [batch_size]
        :return: [batch_size, sequence_length, model_dim], [batch_size, model_dim]
        """

        hidden = self.norm_attention(x)

        hidden, embeddings = self.memory_attention(x=hidden,
                                                   pad_mask=pad_mask,
                                                   labels=labels)

        hidden = self.dropout_attention(hidden)

        if self.fusion_gate is not None:
            x = self.fusion_gate(x, hidden)
        else:
            x = x + hidden

        hidden = self.norm_feed_forward(x)

        hidden = self.position_wise_feed_forward(hidden)

        x = x + self.dropout_feed_forward(hidden)

        return x, embeddings
