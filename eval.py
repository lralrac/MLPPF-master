# !/usr/bin/env python
# coding:utf-8
import sys
import torch
import sklearn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from evaluate.examplebasedclassification import *
from evaluate.labelbasedclassification import *
from evaluate.examplebasedranking import *
from evaluate.labelbasedranking import *
import util
import numpy as np
from config import Config
from dataset_preprocessing.classification_dataset import ClassificationDataset
from dataset_preprocessing.collator import ClassificationCollator
from dataset_preprocessing.collator import ClassificationType
from dataset_preprocessing.collator import FastTextCollator
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator


from model.classification.textrnn import TextRNN
from model.classification.transformer import Transformer
from model.classification.textcnn import Textcnn

from model.model_util import get_optimizer
from util import ModeType
from precision_test import take_values
from k_ROC import built_ROC,class_Metric


def get_classification_model(model_name, dataset, conf):
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def kfold_eval(conf):
    logger = util.Logger(conf)
    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"

    # Build test data set
    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    # Create a data loader collate_fn
    collate_fn = globals()[collate_name](conf, len(test_dataset.label_map))
    # In the testing phase, there is generally no need to disrupt the data, so set it to False
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    empty_dataset = globals()[dataset_name](conf, [])
    model = get_classification_model(model_name, empty_dataset, conf)
    optimizer = get_optimizer(conf, model)
    load_checkpoint(conf.eval.model_dir, conf, model, optimizer)
    model.eval()
    predict_probs = []
    standard_labels = []
    evaluator = cEvaluator(conf.eval.dir)
    for batch in test_data_loader:
        logits = model(batch)
        result = torch.sigmoid(logits).cpu().tolist()
        predict_probs.extend(result)
        standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

        # ============================ EVALUATION API ============================================================================================
    y_test, predictions = [], []

    print ("standard_labels-------------------------:")
    print(standard_labels)
    print("predict_probs-------------------:")
    print(predict_probs)
    print("<-------------------------------------------------------------------------------------------------->")
    for i, j in zip(standard_labels, predict_probs):
            y_test.append(i)
            predictions.append(j)
    print("y_test:")

    print(y_test)
    print("predictions:")
    #print("qqqqqqqqq")
    print(predictions)

    #print("<===========================================================================================================>")
    pred, actual = take_values(predictions, y_test , conf.eval.threshold, conf.eval.top_k )
    print("pred:")
    print(pred)
    #print("qqqqqqqqq")
    print("actual:")
    print(actual)

    # print("</////////////////////////////////////////////////////////////////////////////////////////////////////////////>")

    sum_actual = actual
    actual=np.array(actual)
    pred=np.array(pred)
    pred_b = np.array(predictions)
    # print("---------------------actual--------------------------:",actual)
    # print("---------------------pred----------------------------:",pred_b)

    sum_pred_b = predictions
    #print(sum_actual)
    #print(sum_pred_b)
    fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision= built_ROC(actual,pred_b)

    precision_class,recall_class,f1_class,AUC_class,AUPR_class=class_Metric(actual,pred,pred_b)

    evaluation_measures={"Accuracy": accuracy(actual, pred) ,
                             "Precision": precision(actual, pred) ,
                             "Recall": recall(actual, pred) ,
                             "F1 score": f1_scor(actual, pred) ,
                             "Hamming Loss":hammingLoss(actual, pred),
                             "f-1 Macro":macroF1(actual, pred) ,
                             "f-1 Micro":microF1(actual, pred),
                             "averagePrecision":averagePrecision(actual, pred),
                             "specificity ": specificity (actual, pred)
                             }
    return evaluation_measures,fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision,sum_actual, sum_pred_b,precision_class,recall_class,f1_class,AUC_class,AUPR_class

if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    eval(config)
    # kfold_eval(config)
