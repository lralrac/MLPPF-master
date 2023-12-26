# -*- coding: utf-8 -*-
from eval import *
import os
import shutil
import sys
import time
import csv
import torch
from torch.utils.data import DataLoader
from colorama import Fore
import util
from config import Config

def test(conf, dir,ngram):
    sum_precision = []
    sum_recall = []
    sum_accuracy = []
    sum_f1_scor = []
    sum_micro_fscore = []
    sum_macro_fscore = []
    sum_hamming_loss = []

    conf.data.test_json_files = [os.path.join(dir)]
    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)
    model_name = conf.model_name  # TextRNN/Textcnn/Transformer
    evaluation_measures, fpr, tpr, roc_auc, prprecision, prrecall, praverage_precision, all_actual, all_pred, Cprecision, Crecall, Cf1, CAUC, CAUPR = kfold_eval(
            config)

    sum_precision.append(evaluation_measures["Precision"])
    sum_recall.append(evaluation_measures["Recall"])
    sum_accuracy.append(evaluation_measures["Accuracy"])
    sum_f1_scor.append(evaluation_measures["F1 score"])
    sum_micro_fscore.append(evaluation_measures["f-1 Micro"])
    sum_macro_fscore.append(evaluation_measures["f-1 Macro"])
    sum_hamming_loss.append(evaluation_measures["Hamming Loss"])
    print(Fore.BLUE + "k-fold  precision", sum(sum_precision))
    print(Fore.RED + str(sum_precision))
    print(Fore.BLUE + "k-fold recall", sum(sum_recall))
    print(Fore.RED + str(sum_recall))
    print(Fore.BLUE + "k-fold fscore", sum(sum_f1_scor))
    print(Fore.RED + str(sum_f1_scor))
    print(Fore.BLUE + "k-fold Micro Fscore", sum(sum_micro_fscore))
    print(Fore.RED + str(sum_micro_fscore))
    print(Fore.BLUE + "k-fold Macro Fscore", sum(sum_macro_fscore))
    print(Fore.RED + str(sum_macro_fscore))
    print(Fore.BLUE + "k-fold Accuracy", sum(sum_accuracy))
    print(Fore.RED + str(sum_accuracy))
    print(Fore.BLUE + "k-fold Hamming Loss", sum(sum_hamming_loss))
    print(Fore.RED + str(sum_hamming_loss))


if __name__ == '__main__':
    t1= time.time()
    ngrams=4
    """
    Get super parameters from conf/train.json
    """
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    config.feature.max_char_len_per_token = ngrams
    test(config, "test.json", ngrams)
    t2 = time.time()
    print('Time:',t2-t1)
