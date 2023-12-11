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

import json
import os

import torch

from util import Logger
from util import ModeType
from util import Type


class InsertVocabMode(Type):
    """Standard names for embedding mode
    Given the vocab tensor shape[batch_size, sequence_len].
    The following keys are defined:
    * `FLAT`: Normal mode, return tensor shape will be
    *         [batch_size, sequence_len, embedding_size]
    * `MEAN`: Mean mode, return tensor shape will be
    *         [batch_size, embedding_size]
    * `SUM`: Sum mode, return tensor shape will be
    *        [batch_size, embedding_size]
    """
    ALL = 'all'
    LABEL = 'label'
    OTHER = 'other'

    def str(self):
        return ",".join(
            [self.ALL, self.LABEL, self.OTHER])


class DatasetBase(torch.utils.data.dataset.Dataset):
    """Base dataset_preprocessing class
    """
    CHARSET = "utf-8"

    # 了解VOCAB_PADDING、VOCAB_UNKNOWN、VOCAB_PADDING_LEARNABLE
    VOCAB_PADDING = 0  # Embedding is all zero and not learnable
    VOCAB_UNKNOWN = 1
    VOCAB_PADDING_LEARNABLE = 2  # Embedding is random initialized and learnable


    BIG_VALUE = 1000 * 1000 * 1000

    def __init__(self, config, json_files, generate_dict=False,
                 mode=ModeType.EVAL):
        """
        Another way to do this is keep the file handler. But when DataLoader's
            num_worker bigger than 1, error will occur.
        Args:
            config:
        """
        self.config = config
        self.logger = Logger(config)
        self._init_dict()
        self.sample_index = []
        self.sample_size = 0
        self.model_mode = mode

        # 如果需要中断并重新启动脚本，这可以用于从每个文件的最后位置恢复读取
        self.files = json_files
        # print("dataset.py里面的json_files内容：",json_files)
        for i, json_file in enumerate(json_files):
            with open(json_file) as fin:
                self.sample_index.append([i, 0])
                while True:
                    # 从当前打开的JSON文件中读取一行，并将其存储在变量json_str中
                    json_str = fin.readline()
                    # print("输出文件每行的内容：",json_str)
                    # 输出文件每行的内容： {"doc_label": ["mRNA"], "doc_token": ["TAGG", "AGGA", "GGAT", "GATT", "ATTA", "TTAA", "TAAA", "AAAG", "AAGG", "AGGT", "GGTA", "GTAT", "TATA", "ATAC", "TACA", "ACAT", "CATC", "ATCA", "TCAC", "CACT", "ACTG", "CTGC", "TGCT", "GCTG", "CTGC"]}
                    # 如果json_str为空，则达到文件结尾
                    if not json_str:
                        # 如果到达文件结尾，将从列表sample_index移除最后一个元素
                        self.sample_index.pop()
                        break
                    self.sample_size += 1
                    self.sample_index.append([i, fin.tell()])# 获取当前文件位置
        # print("输出文件所有内容存储的sample_index:",self.sample_index)

        def _insert_vocab(files, _mode=InsertVocabMode.ALL):
            # print("输出files:",files)
            for _i, _json_file in enumerate(files):
                with open(_json_file) as _fin:
                    for _json_str in _fin:
                        # print(json.loads(_json_str)){'doc_label': ['lncRNA'], 'doc_token': ['TTAC', 'TACC']}
                        try:
                            self._insert_vocab(json.loads(_json_str), mode)
                        except:
                            print(_json_str)

        # Dict can be generated using:
        # json files or/and pretrained embedding
        #
        if generate_dict:
            # Use train json files to generate dict
            # If generate_dict_using_json_files is true, then all vocab in train
            # will be used, else only part vocab will be used. e.g. label
            vocab_json_files = config.data.train_json_files
            # print("输出vocab_json_files文件：",vocab_json_files)
            # 输出vocab_json_files文件： ['MiRNA_dataset/4\\0\\train.json']
            mode = InsertVocabMode.LABEL
            # print("输出mode模式:",mode)
            # 输出mode模式: label
            if self.config.data.generate_dict_using_json_files:
                mode = InsertVocabMode.ALL
                # print("输出InsertVocabMode.ALL：",mode)_insert_vocab
                # 输出InsertVocabMode.ALL： all
                self.logger.info("Use dataset_preprocessing to generate dict.")
            _insert_vocab(vocab_json_files, mode)

            if self.config.data.generate_dict_using_all_json_files:
                vocab_json_files += self.config.data.validate_json_files + \
                                    self.config.data.test_json_files
                # print("输出vocab_json_files：",vocab_json_files)
                # 输出vocab_json_files： ['MiRNA_dataset/4\\0\\train.json', 'MiRNA_dataset/4\\0\\valid.json', 'MiRNA_dataset/4\\0\\test.json']
                # print("输出InsertVocabMode.OTHER：",InsertVocabMode.OTHER)
                # 输出InsertVocabMode.OTHER： other
                _insert_vocab(vocab_json_files, InsertVocabMode.OTHER)

            if self.config.data.generate_dict_using_pretrained_embedding:
                self.logger.info("Use pretrained embedding to generate dict.")
                self._load_pretrained_dict()
            self._print_dict_info()

            self._shrink_dict()
            self.logger.info("Shrink dict over.")
            self._print_dict_info(True)
            self._save_dict()
            self._clear_dict()
        self._load_dict()

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        if idx >= self.sample_size:
            raise IndexError
        index = self.sample_index[idx]
        with open(self.files[index[0]]) as fin:
            fin.seek(index[1])
            json_str = fin.readline()
        return self._get_vocab_id_list(json.loads(json_str))

    def _init_dict(self):
        """Init all dict
        """
        raise NotImplementedError

    def _save_dict(self, dict_name=None):
        """Save vocab to file and generate id_to_vocab_dict_map
        Args:
            dict_name: Dict name, if None save all dict. Default None.
        """
        # print("输出_save_dict中dict_name的内容：",dict_name)
        # 输出_save_dict中dict_name的内容： None
        # 输出_save_dict中dict_name的内容： doc_label
        # 输出_save_dict中dict_name的内容： doc_token
        # 输出_save_dict中dict_name的内容： doc_char
        if dict_name is None:
            # 如果不存在dict_miRNA,就新建dict_miRNA文件夹进行存放
            if not os.path.exists(self.config.data.dict_dir):
                os.makedirs(self.config.data.dict_dir)
            for name in self.dict_names:
                self._save_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            # print("输出_save_dict中dict_id的内容：",dict_idx)
            # 打开文件夹
            dict_file = open(self.dict_files[dict_idx], "w")
            # print("输出_save_dict中dict_file的内容：",dict_file)
            #获取与词汇表相关联的一个字典，用于将词汇的编号映射到词汇本身
            id_to_vocab_dict_map = self.id_to_vocab_dict_list[dict_idx]
            index = 0
            for vocab, count in self.count_list[dict_idx]:
                # print(vocab,count)
                # mRNA  3777
                # lncRNA 1499
                id_to_vocab_dict_map[index] = vocab
                index += 1
                dict_file.write("%s\t%d\n" % (vocab, count))
            dict_file.close()

    def _load_dict(self, dict_name=None):
        """Load dict from file.
        Args:
            dict_name: Dict name, if None load all dict. Default None.
        Returns:
            dict.
        """
        # print("在_load_dict函数中dict_name的内容:",dict_name)
        # 在_load_dict函数中dict的内容: None
        # 在_load_dict函数中dict_name的内容: doc_label
        # 在_load_dict函数中dict_name的内容: doc_token
        # 在_load_dict函数中dict_name的内容: doc_char
        if dict_name is None:
            for name in self.dict_names:
                self._load_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            if not os.path.exists(self.dict_files[dict_idx]):
                self.logger.warn("Not exists %s for %s" % (
                    self.dict_files[dict_idx], dict_name))
            else:
                dict_map = self.dicts[dict_idx]
                id_to_vocab_dict_map = self.id_to_vocab_dict_list[dict_idx]
                if dict_name != self.DOC_LABEL:
                    dict_map[self.VOCAB_PADDING] = 0
                    dict_map[self.VOCAB_UNKNOWN] = 1
                    dict_map[self.VOCAB_PADDING_LEARNABLE] = 2
                    id_to_vocab_dict_map[0] = self.VOCAB_PADDING
                    id_to_vocab_dict_map[1] = self.VOCAB_UNKNOWN
                    id_to_vocab_dict_map[2] = self.VOCAB_PADDING_LEARNABLE

                for line in open(self.dict_files[dict_idx], "r"):
                    vocab = line.strip("\n").split("\t")
                    dict_idx = len(dict_map)
                    dict_map[vocab[0]] = dict_idx
                    id_to_vocab_dict_map[dict_idx] = vocab[0]

    def _load_pretrained_dict(self, dict_name=None,
                              pretrained_file=None, min_count=0):
        """Use pretrained embedding to generate dict
        """
        # print("输出pretrained_file:",pretrained_file)
        if dict_name is None:
            # 第一次的时候进入if
            # print("输出pretrained_dict_names：",self.pretrained_dict_names)
            # print("输出pretrained_dict_files：",self.pretrained_dict_files)
            # print("输出pretrained_min_count：",self.pretrained_min_count)
            # 输出pretrained_dict_names： ['doc_token']
            # 输出pretrained_dict_files： ['pre-train-embeddings/node_dict.csv']
            # 输出pretrained_min_count： [0]
            for i, _ in enumerate(self.pretrained_dict_names):
                self._load_pretrained_dict(
                    self.pretrained_dict_names[i],
                    self.pretrained_dict_files[i],
                    self.pretrained_min_count[i])

        else:
            # 第二次的时候进入else
            # 第二次时，pretrained_file为'pre-train-embeddings/node_dict.csv'
            if pretrained_file is None or pretrained_file == "":
                return
            # print("输出dict_names:",self.dict_names)
            # 输出dict_names: ['doc_label', 'doc_token', 'doc_char']
            index = self.dict_names.index(dict_name)
            # print("输出index:",index) 1
            # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            dict_map = self.dicts[index]
            # print("输出原dict_map:",dict_map)
            # print("输出原dict_map的长度：",len(dict_map))
            # 输出dict_map: {'TAGG': 625, 'AGGA': 1481, 'GGAT': 1237, 'GATT': 1100, 'ATTA': 1124, 'TTAA': 893,
            # print(os.getcwd())
            with open(pretrained_file) as fin:
                for line in fin:
                    # 读取pre-train-embedding文件夹下的node_dict.csv中的内容(data)
                    data = line.strip().split(' ')
                    # print("请输出data的内容：",data)
                    # 请输出data的内容： ['AGCC', '-0.1377866566181183', '0.234967902302742', '0.4696999490261078',
                    if len(data) == 2:
                        continue
                    # 判断 预训练词向量中的token是否存在于dict_map
                    if data[0] not in dict_map:
                        dict_map[data[0]] = 0
                    dict_map[data[0]] += min_count + 1
            # print("输出更新dict_map的内容：",dict_map)
            # print("输出更新dict_map的长度：", len(dict_map))

    def _insert_vocab(self, json_obj, mode=InsertVocabMode.ALL):
        """Insert vocab to dict
        """
        raise NotImplementedError

    def _shrink_dict(self, dict_name=None):
        # dict_names: ['doc_label', 'doc_token', 'doc_char']
        # print("在_shrink_dict精简词汇表函数中dict_name的对象：",dict_name)
        # 首次当进入这个精简函数中dict_name为None,进入for循环对所有词汇表都进行精简
        # 在_shrink_dict精简词汇表函数中dict_name的对象： doc_label
        # 在_shrink_dict精简词汇表函数中dict_name的对象： doc_token
        # 在_shrink_dict精简词汇表函数中dict_name的对象： doc_char
        if dict_name is None:
            for name in self.dict_names:
                self._shrink_dict(name)
        else:

            dict_idx = self.dict_names.index(dict_name)
            # self.dicts为总的词汇表
            self.count_list[dict_idx] = sorted(self.dicts[dict_idx].items(),
                                               key=lambda x: (x[1], x[0]),
                                               reverse=True)
            self.count_list[dict_idx] = \
                [(k, v) for k, v in self.count_list[dict_idx] if
                 v >= self.min_count[dict_idx]][0:self.max_dict_size[dict_idx]]
            # print("输出排序后的count_lsit：",self.count_list)
    #         输出三个词汇表都排序后的count_lsit： [[('mRNA', 3777), ('lncRNA', 1499)], [('CAGA', 1893), ('ACAG', 1826), ('GACA', 1627), ('
    def _clear_dict(self):
        """Clear all dict
        """
        # self.dicts为总的词汇表
        # 清空存储词汇表和相关映射的数据结构
        for dict_map in self.dicts:
            dict_map.clear()
        for id_to_vocab_dict in self.id_to_vocab_dict_list:
            id_to_vocab_dict.clear()

    def _print_dict_info(self, count_list=False):
        """Print dict info
        """
        # dict_names: ['doc_label', 'doc_token', 'doc_char']
        # print("count_list内容：",count_list)
        for i, dict_name in enumerate(self.dict_names):

            if count_list:
                self.logger.info(
                    "Size of %s dict is %d" % (
                        dict_name, len(self.count_list[i])))
            else:
                # len(self.dict[0])表示词汇表中label的长度
                # len(self.dict[1])表示词汇表中token的长度
                # len(self.dict[2])表示词汇表中char的长度
                self.logger.info(
                    "Size of %s dict is %d" % (dict_name, len(self.dicts[i])))

    def _insert_sequence_tokens(self, sequence_tokens, token_map,
                                 char_map):
        # print("输出sequence_token的内容：",sequence_tokens)
        for token in sequence_tokens:
            # print("在dataset.py里面输出token:",token)
            for char in token:
                # print("遍历token中的字符:",char)
                self._add_vocab_to_dict(char_map, char)
            # print("输出char_map:",char_map)
            self._add_vocab_to_dict(token_map, token)


    def _insert_sequence_vocab(self, sequence_vocabs, dict_map):
        for vocab in sequence_vocabs:
            self._add_vocab_to_dict(dict_map, vocab)

    @staticmethod
    def _add_vocab_to_dict(dict_map, vocab):
        if vocab not in dict_map:
            dict_map[vocab] = 0
        dict_map[vocab] += 1

    def _get_vocab_id_list(self, json_obj):
        """Use dict to convert all vocabs to ids
        """
        return json_obj

    def _label_to_id(self, sequence_labels, dict_map):
        """Convert label to id. The reason that label is not in label map may be
          label is filtered or label in validate/test does not occur in train set
        """
        label_id_list = []
        for label in sequence_labels:
            if label not in dict_map:
                self.logger.warn("Label not in label map: %s" % label)
            else:
                label_id_list.append(self.label_map[label])
        assert label_id_list, "Label is empty: %s" % " ".join(sequence_labels)

        return label_id_list

    def _token_to_id(self, sequence_tokens, token_map, char_map,  max_char_sequence_length=-1,
                     max_char_length_per_token=-1):
        """Convert token to id. Vocab not in dict map will be map to _UNK
        """
        token_id_list = []
        char_id_list = []
        char_in_token_id_list = []
        for token in sequence_tokens:
            char_id = [char_map.get(x, self.VOCAB_UNKNOWN) for x in token]
            char_id_list.extend(char_id[0:max_char_sequence_length])
            char_in_token = [char_map.get(x, self.VOCAB_UNKNOWN)
                             for x in token[0:max_char_length_per_token]]
            char_in_token_id_list.append(char_in_token)

            token_id_list.append(
                token_map.get(token, token_map[self.VOCAB_UNKNOWN]))

        if not sequence_tokens:
            token_id_list.append(self.VOCAB_PADDING)
            char_id_list.append(self.VOCAB_PADDING)
            char_in_token_id_list.append([self.VOCAB_PADDING])

        return token_id_list, char_id_list, char_in_token_id_list

    def _vocab_to_id(self, sequence_vocabs, dict_map):
        """Convert vocab to id. Vocab not in dict map will be map to _UNK
        """
        vocab_id_list = \
            [dict_map.get(x, self.VOCAB_UNKNOWN) for x in sequence_vocabs]
        if not vocab_id_list:
            vocab_id_list.append(self.VOCAB_PADDING)
        return vocab_id_list
