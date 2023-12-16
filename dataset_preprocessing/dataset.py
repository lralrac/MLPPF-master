#!/usr/bin/env python
#coding:utf-8


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

    # VOCAB_PADDING、VOCAB_UNKNOWN、VOCAB_PADDING_LEARNABLE
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

        # This can be used to resume reading from the last position of each file if the script needs to be interrupted and restarted
        self.files = json_files
        for i, json_file in enumerate(json_files):
            with open(json_file) as fin:
                self.sample_index.append([i, 0])
                while True:
                    # Read a line from the currently open JSON file and store it in the variable json_str
                    json_str = fin.readline()
                    # Output the contents of each line of the file： {"doc_label": ["mRNA"], "doc_token": ["TAGG", "AGGA", "GGAT", "GATT", "ATTA", "TTAA", "TAAA", "AAAG", "AAGG", "AGGT", "GGTA", "GTAT", "TATA", "ATAC", "TACA", "ACAT", "CATC", "ATCA", "TCAC", "CACT", "ACTG", "CTGC", "TGCT", "GCTG", "CTGC"]}
                    # If json_str is empty, end of file reached
                    if not json_str:
                        self.sample_index.pop()
                        break
                    self.sample_size += 1
                    self.sample_index.append([i, fin.tell()])# Get current file location


        def _insert_vocab(files, _mode=InsertVocabMode.ALL):
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
            mode = InsertVocabMode.LABEL
            if self.config.data.generate_dict_using_json_files:
                mode = InsertVocabMode.ALL
                self.logger.info("Use dataset_preprocessing to generate dict.")
            _insert_vocab(vocab_json_files, mode)

            if self.config.data.generate_dict_using_all_json_files:
                vocab_json_files += self.config.data.validate_json_files + \
                                    self.config.data.test_json_files
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
        if dict_name is None:
            if not os.path.exists(self.config.data.dict_dir):
                os.makedirs(self.config.data.dict_dir)
            for name in self.dict_names:
                self._save_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            dict_file = open(self.dict_files[dict_idx], "w")
            id_to_vocab_dict_map = self.id_to_vocab_dict_list[dict_idx]
            index = 0
            for vocab, count in self.count_list[dict_idx]:
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
        # print("pretrained_file:",pretrained_file)
        if dict_name is None:
            for i, _ in enumerate(self.pretrained_dict_names):
                self._load_pretrained_dict(
                    self.pretrained_dict_names[i],
                    self.pretrained_dict_files[i],
                    self.pretrained_min_count[i])

        else:
            if pretrained_file is None or pretrained_file == "":
                return
            # dict_names: ['doc_label', 'doc_token', 'doc_char']
            index = self.dict_names.index(dict_name)
            # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            dict_map = self.dicts[index]
            with open(pretrained_file) as fin:
                for line in fin:
                    data = line.strip().split(' ')
                    # data： ['AGCC', '-0.1377866566181183', '0.234967902302742', '0.4696999490261078',
                    if len(data) == 2:
                        continue
                    if data[0] not in dict_map:
                        dict_map[data[0]] = 0
                    dict_map[data[0]] += min_count + 1


    def _insert_vocab(self, json_obj, mode=InsertVocabMode.ALL):
        """Insert vocab to dict
        """
        raise NotImplementedError

    def _shrink_dict(self, dict_name=None):
        # dict_names: ['doc_label', 'doc_token', 'doc_char']
        if dict_name is None:
            for name in self.dict_names:
                self._shrink_dict(name)
        else:

            dict_idx = self.dict_names.index(dict_name)
            self.count_list[dict_idx] = sorted(self.dicts[dict_idx].items(),
                                               key=lambda x: (x[1], x[0]),
                                               reverse=True)
            self.count_list[dict_idx] = \
                [(k, v) for k, v in self.count_list[dict_idx] if
                 v >= self.min_count[dict_idx]][0:self.max_dict_size[dict_idx]]

    def _clear_dict(self):
        """Clear all dict
        """
        for dict_map in self.dicts:
            dict_map.clear()
        for id_to_vocab_dict in self.id_to_vocab_dict_list:
            id_to_vocab_dict.clear()

    def _print_dict_info(self, count_list=False):
        """Print dict info
        """
        # dict_names: ['doc_label', 'doc_token', 'doc_char']
        # print("count_list：",count_list)
        for i, dict_name in enumerate(self.dict_names):

            if count_list:
                self.logger.info(
                    "Size of %s dict is %d" % (
                        dict_name, len(self.count_list[i])))
            else:
                # len(self.dict[0]) represents the length of label in the vocabulary
                # len(self.dict[1]) represents the length of token in the vocabulary
                self.logger.info(
                    "Size of %s dict is %d" % (dict_name, len(self.dicts[i])))

    def _insert_sequence_tokens(self, sequence_tokens, token_map,
                                 char_map):
        for token in sequence_tokens:
            for char in token:
                self._add_vocab_to_dict(char_map, char)
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
