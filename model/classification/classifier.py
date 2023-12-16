#!usr/bin/env python
# coding:utf-8


import torch

from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from model.embedding import Embedding
from model.embedding import EmbeddingProcessType
from model.embedding import EmbeddingType
from model.model_util import ActivationType


class Classifier(torch.nn.Module):
    def __init__(self, dataset, config):
            super(Classifier, self).__init__()
            self.config = config
            self.token_embedding = \
                Embedding(dataset.token_map, config.embedding.dimension,
                          cDataset.DOC_TOKEN, config, dataset.VOCAB_PADDING,
                          pretrained_embedding_file=
                          config.feature.token_pretrained_file,
                          mode=EmbeddingProcessType.FLAT,
                          dropout=self.config.embedding.dropout,
                          init_type=self.config.embedding.initializer,
                          low=-self.config.embedding.uniform_bound,
                          high=self.config.embedding.uniform_bound,
                          std=self.config.embedding.random_stddev,
                          fan_mode=self.config.embedding.fan_mode,
                          activation_type=ActivationType.NONE,
                          model_mode=dataset.model_mode)
            self.char_embedding = \
                Embedding(dataset.char_map, config.embedding.dimension,
                          cDataset.DOC_CHAR, config, dataset.VOCAB_PADDING,
                          mode=EmbeddingProcessType.FLAT,
                          dropout=self.config.embedding.dropout,
                          init_type=self.config.embedding.initializer,
                          low=-self.config.embedding.uniform_bound,
                          high=self.config.embedding.uniform_bound,
                          std=self.config.embedding.random_stddev,
                          fan_mode=self.config.embedding.fan_mode,
                          activation_type=ActivationType.NONE,
                          model_mode=dataset.model_mode)
            self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_embedding(self, batch, pad_shape=None, pad_value=0):

            token_id = batch[cDataset.DOC_TOKEN].to(self.config.device)
            if pad_shape is not None:
                token_id = torch.nn.functional.pad(
                    token_id, pad_shape, mode='constant', value=pad_value)
            embedding = self.token_embedding(token_id)
            length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
            mask = batch[cDataset.DOC_TOKEN_MASK].to(self.config.device)

            return embedding, length, mask

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append(
            {'params': self.token_embedding.parameters(), 'is_embedding': True})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        raise NotImplementedError
