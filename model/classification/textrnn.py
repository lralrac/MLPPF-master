#!usr/bin/env python
# coding:utf-8
import torch

from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.embedding import PositionEmbedding
from model.layers import SumAttention
from model.rnn import RNN
from util import Type


class DocEmbeddingType(Type):
    """Standard names for doc embedding type.
    """
    AVG = 'AVG'
    ATTENTION = 'Attention'
    LAST_HIDDEN = 'LastHidden'
    OUT='Out'


    @classmethod
    def str(cls):
        return ",".join(
            [cls.AVG, cls.ATTENTION, cls.LAST_HIDDEN, cls.OUT])


class TextRNN(Classifier):
    """Implement TextRNN, contains UniLSTM，BiLSTM，UniGRU，BiGRU
    """

    def __init__(self, dataset, config):
        super(TextRNN, self).__init__(dataset, config)

        self.pad = dataset.token_map[dataset.VOCAB_PADDING]

        seq_max_len = config.feature.max_token_len
        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)
        self.doc_embedding_type = config.TextRNN.doc_embedding_type
        self.rnn = RNN(
            config.embedding.dimension, config.TextRNN.hidden_dimension,
            num_layers=config.TextRNN.num_layers, batch_first=True,
            bidirectional=config.TextRNN.bidirectional,
            rnn_type=config.TextRNN.rnn_type)
        hidden_dimension = config.TextRNN.hidden_dimension
        if config.TextRNN.bidirectional:
            hidden_dimension *= 2
        self.sum_attention = SumAttention(hidden_dimension,
                                          config.TextRNN.attention_dimension,
                                          config.device)
        self.linear = torch.nn.Linear(hidden_dimension, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)


    def get_parameter_optimizer_dict(self):
        params = super(TextRNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.linear.parameters()})
        if self.doc_embedding_type == DocEmbeddingType.ATTENTION:
            params.append({'params': self.sum_attention.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0.0

    def forward(self, batch):
        # if self.config.feature.feature_names[0] == "token":
        src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
        embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
        length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)

        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, lengths in enumerate(batch_lens):
            src_pos[row][:lengths] = torch.arange(1, lengths + 1)

        embedding = embedding  + self.position_enc(src_pos)
        # embedding = embedding
        output, last_hidden = self.rnn(embedding, length)

        if self.doc_embedding_type == DocEmbeddingType.AVG:
            doc_embedding = torch.sum(output, 1) / length.unsqueeze(1)
        elif self.doc_embedding_type == DocEmbeddingType.ATTENTION:
            doc_embedding = self.sum_attention(output)
        elif self.doc_embedding_type == DocEmbeddingType.LAST_HIDDEN:
            doc_embedding = last_hidden
        elif self.doc_embedding_type == DocEmbeddingType.OUT:
            doc_embedding = last_hidden

        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    self.doc_embedding_type, DocEmbeddingType.str()))



        return self.dropout(self.linear(doc_embedding))

