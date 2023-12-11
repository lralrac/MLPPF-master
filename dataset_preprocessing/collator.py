#!/usr/bin/env python
#coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

"""Collator for NeuralClassifier"""

import torch

from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from util import Type


class Collator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        raise NotImplementedError


class ClassificationType(Type):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"

    @classmethod
    def str(cls):
        return ",".join([cls.SINGLE_LABEL, cls.MULTI_LABEL])

# 定义一个名为ClassificationCollator的类,继承自Collator(父类)
class ClassificationCollator(Collator):
    def __init__(self, conf, label_size):
        # 初始化ClassificationCollator类
        # 接受两个参数:conf 和 label_size
        super(ClassificationCollator, self).__init__(conf.device)
        # 根据模型的不同设置最小序列长度 min_seq
        min_seq = 1
        if conf.model_name == "TextCNN":
            min_seq = conf.TextCNN.top_k_max_pooling
        elif conf.model_name == "DPCNN":
            min_seq = conf.DPCNN.kernel_size * 2 ** conf.DPCNN.blocks
        elif conf.model_name == "RegionEmbedding":
            min_seq = conf.feature.max_token_len
        self.min_token_max_len = min_seq
        self.min_char_max_len = min_seq
        # 用于表示分类任务的标签大小
        self.label_size = label_size

    def _get_multi_hot_label(self, doc_labels):
        """For multi-label classification
        Generate multi-hot for input labels
        e.g. input: [[0,1], [2]]
        表示在第一个位置和第二个位置都存在【0，1】
             output: [[1,1,0], [0,0,1]]
        """
        # print(doc_labels)
        batch_size = len(doc_labels)
        # print("collator文件中batch_size大小:",batch_size)
        max_label_num = max([len(x) for x in doc_labels])
        # print("输出max_label_num长度",max_label_num)
        # 从原始文档标签列表中取出第一个标签，并将其添加到doc_labels_extend中。通过不断重复第一个标签，指导新列表的长度等于max_label_extend
        # 初始化doc_labels_extend列表
        doc_labels_extend = \
            [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
        # print("初始化doc_label_extend",doc_labels_extend)
        for i in range(0, batch_size):
            doc_labels_extend[i][0 : len(doc_labels[i])] = doc_labels[i]
        # print("标签数量截断为原始文档标签列表中的实际标签数量",doc_labels_extend)
        y = torch.Tensor(doc_labels_extend).long()
        # 创建一个全零的张量，作为初始的独热编码
        y_onehot = torch.zeros(batch_size, self.label_size).scatter_(1, y, 1)
        # print("输出y_onehot编码向量：",y_onehot)
        # print(y_onehot)

        return y_onehot

    def _append_label(self, doc_labels, sample):
        doc_labels.append(sample[cDataset.DOC_LABEL])


    def __call__(self, batch):
        def _append_vocab(ori_vocabs, vocabs, max_len):
            # 生成一个包含填充值的列表，用于填充到原始词汇列表后面
            padding = [cDataset.VOCAB_PADDING] * (max_len - len(ori_vocabs))
            vocabs.append(ori_vocabs + padding)
        # 用于存储文档标签
        doc_labels = []
        # 用于存储文档的词汇、字符和字符在词汇中的表示
        doc_token = []
        doc_char = []
        doc_char_in_token = []
        # 用于存储文档的词汇长度，字符长度和字符在词汇中的表示长度
        doc_token_len = []
        doc_char_len = []
        doc_char_in_token_len = []
        # 用于记录批次中文档的最大词汇长度、字符长度以及字符在词汇中的表示的最大长度
        doc_token_max_len = self.min_token_max_len
        doc_char_max_len = self.min_char_max_len
        doc_char_in_token_max_len = 0


        # print("在collator.py中输出batch:",batch)
        for _, value in enumerate(batch):
            # print("在collator.py中输出batch:",_,value)
            # print("输出_:",_)
            # print("输出doc_token-----------------------",value[cDataset.DOC_TOKEN])
            # 输出doc_token----------------------- [180, 70, 40, 47, 27, 35, 5, 4, 3, 6, 5, 4, 3, 6, 5, 4, 3, 14, 15, 21, 4, 90, 110, 94, 65, 136, 128, 157, 210]
            doc_token_max_len = max(doc_token_max_len,
                                    len(value[cDataset.DOC_TOKEN]))
            doc_char_max_len = max(doc_char_max_len,
                                   len(value[cDataset.DOC_CHAR]))
            for char_in_token in value[cDataset.DOC_CHAR_IN_TOKEN]:
                doc_char_in_token_max_len = max(doc_char_in_token_max_len,
                                                len(char_in_token))
        # doc_token_max_len长度为30，doc_char_max_len为120，doc_char_in_token_max_len为4
        # print("输出最大doc_token_max_len长度：",doc_token_max_len)
        # print("输出最大doc_char_max_len长度：",doc_char_max_len)
        # print("输出最大doc_char_in_token_max_len:",doc_char_in_token_max_len)


        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            #value[cDataset.DOC_TOKEN] 表示当前文档中的词汇列表
            # _append_vocab 函数的作用是将原始词汇列表扩展到指定的最大长度 max_len，并将扩展后的词汇列表追加到目标词汇列表 vocabs
            _append_vocab(value[cDataset.DOC_TOKEN], doc_token,
                          doc_token_max_len)
            # 用于记录当前文档的词汇列表的长度
            doc_token_len.append(len(value[cDataset.DOC_TOKEN]))
            _append_vocab(value[cDataset.DOC_CHAR], doc_char, doc_char_max_len)
            doc_char_len.append(len(value[cDataset.DOC_CHAR]))

            doc_char_in_token_len_tmp = []
            for char_in_token in value[cDataset.DOC_CHAR_IN_TOKEN]:
                _append_vocab(char_in_token, doc_char_in_token,
                              doc_char_in_token_max_len)
                doc_char_in_token_len_tmp.append(len(char_in_token))

            padding = [cDataset.VOCAB_PADDING] * doc_char_in_token_max_len
            for _ in range(
                    len(value[cDataset.DOC_CHAR_IN_TOKEN]), doc_token_max_len):
                doc_char_in_token.append(padding)
                doc_char_in_token_len_tmp.append(0)
            doc_char_in_token_len.append(doc_char_in_token_len_tmp)

        # print("输出doc_label:",doc_labels)
        tensor_doc_labels = self._get_multi_hot_label(doc_labels)
        # print("输出tensor_doc_labels：",tensor_doc_labels)
        doc_label_list = doc_labels

        # print("输出doc_token:",doc_token)
        # print("输出doc_token的长度:",len(doc_token))
        # print("输出tensor_doc_token:",torch.tensor(doc_token))
        # print("输出tensor_doc_token的形状:",torch.tensor(doc_token).shape)
        # print("输出torch.tensor(doc_char：",torch.tensor(doc_char))
        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,

            cDataset.DOC_TOKEN: torch.tensor(doc_token),
            cDataset.DOC_CHAR: torch.tensor(doc_char),
            cDataset.DOC_CHAR_IN_TOKEN: torch.tensor(doc_char_in_token),

            cDataset.DOC_TOKEN_MASK: torch.tensor(doc_token).gt(0).float(),
            cDataset.DOC_CHAR_MASK: torch.tensor(doc_char).gt(0).float(),
            cDataset.DOC_CHAR_IN_TOKEN_MASK:
                torch.tensor(doc_char_in_token).gt(0).float(),

            cDataset.DOC_TOKEN_LEN: torch.tensor(
                doc_token_len, dtype=torch.float32),
            cDataset.DOC_CHAR_LEN: torch.tensor(
                doc_char_len, dtype=torch.float32),
            cDataset.DOC_CHAR_IN_TOKEN_LEN: torch.tensor(
                doc_char_in_token_len, dtype=torch.float32),

            cDataset.DOC_TOKEN_MAX_LEN:
                torch.tensor([doc_token_max_len], dtype=torch.float32),
            cDataset.DOC_CHAR_MAX_LEN:
                torch.tensor([doc_char_max_len], dtype=torch.float32),
            cDataset.DOC_CHAR_IN_TOKEN_MAX_LEN:
                torch.tensor([doc_char_in_token_max_len], dtype=torch.float32)
        }
        # print("______________________输出batch_map_______________________________",batch_map)
        return batch_map


class FastTextCollator(ClassificationCollator):
    """FastText Collator
    Extra support features: token, token-ngrams, keywords, topics.
    """
    def __call__(self, batch):
        def _append_vocab(sample, vocabs, offsets, lens, name):
            filtered_vocab = [x for x in sample[name] if
                              x is not cDataset.VOCAB_UNKNOWN]
            vocabs.extend(filtered_vocab)
            offsets.append(offsets[-1] + len(filtered_vocab))
            lens.append(len(filtered_vocab))

        doc_labels = []

        doc_tokens = []
        doc_token_ngrams = []
        doc_keywords = []
        doc_topics = []

        doc_tokens_offset = [0]
        doc_token_ngrams_offset = [0]
        doc_keywords_offset = [0]
        doc_topics_offset = [0]

        doc_tokens_len = []
        doc_token_ngrams_len = []
        doc_keywords_len = []
        doc_topics_len = []
        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            _append_vocab(value, doc_tokens, doc_tokens_offset,
                          doc_tokens_len,
                          cDataset.DOC_TOKEN)
            _append_vocab(value, doc_token_ngrams, doc_token_ngrams_offset,
                          doc_token_ngrams_len,
                          cDataset.DOC_TOKEN_NGRAM)
            _append_vocab(value, doc_keywords, doc_keywords_offset,
                          doc_keywords_len, cDataset.DOC_KEYWORD)
            _append_vocab(value, doc_topics, doc_topics_offset,
                          doc_topics_len, cDataset.DOC_TOPIC)
        doc_tokens_offset.pop()
        doc_token_ngrams_offset.pop()
        doc_keywords_offset.pop()
        doc_topics_offset.pop()

        if self.classification_type == ClassificationType.SINGLE_LABEL:
            tensor_doc_labels = torch.tensor(doc_labels)
            doc_label_list = [[x] for x in doc_labels]
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            tensor_doc_labels = self._get_multi_hot_label(doc_labels)
            doc_label_list = doc_labels

        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,

            cDataset.DOC_TOKEN: torch.tensor(doc_tokens),
            cDataset.DOC_TOKEN_NGRAM: torch.tensor(doc_token_ngrams),
            cDataset.DOC_KEYWORD: torch.tensor(doc_keywords),
            cDataset.DOC_TOPIC: torch.tensor(doc_topics),

            cDataset.DOC_TOKEN_OFFSET: torch.tensor(doc_tokens_offset),
            cDataset.DOC_TOKEN_NGRAM_OFFSET:
                torch.tensor(doc_token_ngrams_offset),
            cDataset.DOC_KEYWORD_OFFSET: torch.tensor(doc_keywords_offset),
            cDataset.DOC_TOPIC_OFFSET: torch.tensor(doc_topics_offset),

            cDataset.DOC_TOKEN_LEN:
                torch.tensor(doc_tokens_len, dtype=torch.float32),
            cDataset.DOC_TOKEN_NGRAM_LEN:
                torch.tensor(doc_token_ngrams_len, dtype=torch.float32),
            cDataset.DOC_KEYWORD_LEN:
                torch.tensor(doc_keywords_len, dtype=torch.float32),
            cDataset.DOC_TOPIC_LEN:
                torch.tensor(doc_topics_len, dtype=torch.float32)}
        return batch_map
