import torch
from torch import nn
from src import models


model_name = 'distilbert-base-uncased'
batch_size = 128

model_dim = 768

num_labels = 10
num_heads = 12
feed_forward_dim = 768 * 2
num_layers = 6
backbone_encoder_sizes = (model_dim,) * 4

attention_pooling_num_heads = 4

head_sizes = (model_dim,) * 3 + (num_labels,)

backbone = models.MemoryAugmentedPreTrainedBert(model_name=model_name,
                                                num_labels=num_labels,
                                                num_heads=num_heads,
                                                feed_forward_dim=feed_forward_dim,
                                                num_layers=num_layers,
                                                encoder_sizes=backbone_encoder_sizes)

aggregation = models.ResidualAttentionAggregation(model_dim=model_dim,
                                                  num_heads=attention_pooling_num_heads)

head = models.Head(sizes=head_sizes)

model = nn.Sequential()
model.add_module('backbone', backbone)
model.add_module('aggregation', aggregation)
model.add_module('head', head)
