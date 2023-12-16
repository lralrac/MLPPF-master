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
from dataset_preprocessing.classification_dataset import ClassificationDataset
from dataset_preprocessing.collator import ClassificationCollator
from dataset_preprocessing.collator import FastTextCollator
from dataset_preprocessing.collator import ClassificationType
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from k_ROC import save_ROC
from k_ROC import save_sum

from model.classification.textrnn import TextRNN
from model.classification.transformer import Transformer
from model.classification.textcnn import Textcnn
from model.loss import ClassificationLoss
from model.model_util import get_optimizer
from util import ModeType
# from torchviz import make_dot


"""
DataLoader for obtaining training, validation and test data
dataset_name specifies the data set type, collate_name specifies the data processing function type, conf is a configuration object
"""
def get_data_loader(dataset_name, collate_name, conf):  #例 ClassificationDataset,ClassificationCollator,conf
    """Get data loader: Train, Validate, Test
    """
    # Generate a vocabulary from the training dataset and convert text documents into numerical representations
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)#generate_dict是一个布尔值，表示是否需要生成词典,这个类会读取训练数据集文件，并将其转换为模型可以处理的格式

    """
    This class will convert each sample into a dictionary and pack these dictionaries into a batch.
     During this process, it converts the text data of the sample into a digital representation and converts the labels into One-Hot encoded form. Finally, it returns a list containing the batch data.
     Example ClassificationCollator(conf,label_size) label_size is the number of labels (2)
    """
    collate_fn = globals()[collate_name](conf, len(train_dataset.label_map))

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = globals()[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader

def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model

"""
Class for ClassificationTrainer, used to train and evaluate text classification models. This class contains the following methods and properties:
__init__(self, label_map, logger, evaluator, conf, loss_fn): Constructor of the class,
Used to initialize various properties and parameters such as label maps, loggers, evaluators, configurations, loss functions, etc.
train(self, data_loader, model, optimizer, stage, epoch): Method used to train the model. It accepts a data loader, model, optimizer, training phase, and current epoch number as input, and returns the results of the training.
eval(self, data_loader, model, optimizer, stage, epoch): Method used to evaluate model performance. It accepts a data loader, model, optimizer, evaluation stage, and current round number as input, and returns the results of the evaluation.
run(self, data_loader, model, optimizer, stage, epoch, mode=ModeType.EVAL): Method used to run training or evaluation. It accepts a data loader, model, optimizer, stage, current round number, and mode (defaults to evaluation mode) as input, and returns the results.
label_map: Label mapping dictionary, used to convert labels into digital representations.
logger: object used for logging.
evaluator: Object used to evaluate model performance.
conf: Object containing various training parameters and configurations.
loss_fn: Loss function object, used to calculate the loss of the model.
"""
class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        # if self.conf.task_info.hierarchical:
        #     self.hierar_relations = get_hierar_relations(
        #         self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)# Update learning rate
        model.train()# Method used to set the model into training mode
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            # Logits are used to store the original prediction results of the neural network model
            # batch represents batch processing of input data
            logits = model(batch)
            loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            # Check whether the current running model is TRAIN
            if mode == ModeType.TRAIN:
                with torch.autograd.set_detect_anomaly(True):
                    # Clear the gradient information of previous iterations stored in the optimizer to calculate new gradients for the current iteration
                    optimizer.zero_grad()
                    # Used to calculate the loss of the current batch of data, and then use the back propagation algorithm to calculate the gradient. The gradient is used to adjust the weights and parameters of the model to reduce the loss value.
                    loss.backward()
                    # Reduce the loss value by updating the weights and parameters of the model according to the gradient
                    optimizer.step()
                    continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

            """
            First calculate the average loss function value.
             Then, call the self.evaluator.evaluate() function to calculate the performance indicators of the model on the evaluation set.
             The input parameters of this function include predicted probability value predict_probs, standard label standard_labels, label map label_map, threshold threshold, top-k value top_k
             And whether to perform multi-label classification is_multi.
             The function returns a tuple containing multiple performance indicators, such as precision, recall, fscore, etc.
            """
        # Check whether the current running mode is evaluation mode
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            # precision_list is used to store the precise value of each category
            # recall_list is used to store the recall value of each category
            # fscore_list is used to store the F1 score value of each category
            # right_list is used to store the number of correctly predicted samples
            # predict_list is used to store the prediction results of the model
            # standard_list is used to store real label information
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list,AUC_epoch,AUPR_epoch) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                     is_multi=is_multi)
            # print("--------------",stage,AUC_epoch,AUPR_epoch)
            # The terminal outputs the performance indicators evaluated in each epoch of each training phase, verification phase, and test phase.
            self.logger.warn(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                    standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            return precision_list[0][cEvaluator.MICRO_AVERAGE], precision_list[0][cEvaluator.MACRO_AVERAGE],recall_list[0][cEvaluator.MICRO_AVERAGE],recall_list[0][cEvaluator.MACRO_AVERAGE], fscore_list[0][cEvaluator.MICRO_AVERAGE],fscore_list[0][cEvaluator.MACRO_AVERAGE],AUC_epoch,AUPR_epoch

def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def train(conf, dir, fold,ngram):
    sum_precision = []
    sum_recall = []
    sum_accuracy = []
    sum_f1_scor = []
    sum_micro_fscore = []
    sum_macro_fscore = []
    sum_hamming_loss = []
    sum_averagePrecision = []
    sum_specificity = []
    sum_actual = []
    sum_pred = []
    precision_class=[]
    recall_class=[]
    f1_score_class=[]
    auc_class=[]
    aupr_class=[]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prprecision = dict()
    prrecall = dict()
    praverage_precision = dict()
    for i in range(fold):

        print("______________________Fold",i,"______________________")
        """
        train.json, test.json, valid.json
        """
        conf.data.train_json_files=[os.path.join(dir, str(i),"train.json")]
        conf.data.test_json_files=[os.path.join(dir, str(i),"test.json")]
        conf.data.validate_json_files=[os.path.join(dir, str(i),"valid.json")]

        logger = util.Logger(conf)# write log

        if not os.path.exists(conf.checkpoint_dir):
            os.makedirs(conf.checkpoint_dir)

        model_name = conf.model_name# TextRNN/Textcnn/Transformer
        dataset_name = "ClassificationDataset"

        """
        collate_name is used to specify the data processing function (collate_fn) in the data loader (DataLoader).
         In this script, there are two data processing functions, one is FastTextCollator and the other is ClassificationCollator.
         FastTextCollator is used to process data from FastText models, while ClassificationCollator is used to process data from other classification models.
         The value of collate_name will be passed to DataLoader in the get_data_loader() function to determine which data processing function to use.
        """
        collate_name = "FastTextCollator" if model_name == "FastText" \
            else "ClassificationCollator"
        train_data_loader, validate_data_loader, test_data_loader = \
            get_data_loader(dataset_name, collate_name, conf)


        print(train_data_loader)
        count=0
        for flag in train_data_loader:
            if count == 0:
                print(flag)
            count+=1
        """
        empty_dataset
         Use it as a placeholder so you can replace it later if needed.
         In actual use, sometimes we need to dynamically adjust the size of the data set during the training process, such as gradually increasing the size of the data set during the training process.
         At this time, we can first use empty_dataset to create an empty dataset object, and then dynamically expand the dataset by adding sample data when needed.
        """

        empty_dataset = globals()[dataset_name](conf, [])
        model = get_classification_model(model_name, empty_dataset, conf)
        loss_fn = globals()["ClassificationLoss"](
            label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
        if(model == "Transformer" or model =="Textcnn"):
            optimizer  = model.parameters()
        else:
            optimizer = get_optimizer(conf, model)
        evaluator = cEvaluator(conf.eval.dir)
        trainer = globals()["ClassificationTrainer"](
            empty_dataset.label_map, logger, evaluator, conf, loss_fn)

        best_epoch = -1
        best_performance = 0
        model_file_prefix = conf.checkpoint_dir + "/" + model_name
        for epoch in range(conf.train.start_epoch,
                           conf.train.start_epoch + conf.train.num_epochs):
            start_time = time.time()
            trainer.train(train_data_loader, model, optimizer, "Train", epoch)
            trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
            performance = trainer.eval(
                validate_data_loader, model, optimizer, "Validate", epoch)
            trainer.eval(test_data_loader, model, optimizer, "test", epoch)
            if performance[4] > best_performance:
                best_epoch = epoch
                best_performance = performance[4]
            save_checkpoint({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict(),
            }, model_file_prefix)#save the data of the model
            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

        best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
        best_file_name = model_file_prefix + "_best"
        shutil.copyfile(best_epoch_file_name, best_file_name)

        load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
                        optimizer)
        trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch)

        evaluation_measures, fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision,all_actual, all_pred,Cprecision,Crecall,Cf1,CAUC,CAUPR = kfold_eval(config)
        for z in range(len(all_actual)):
            sum_actual.append(all_actual[z])
            sum_pred.append(all_pred[z])
        save_ROC(ngram, i, fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision)
        # Store the Precision, Recall, Accuracy, F1 score, f-1 Micro, f-1 Macro, Hamming Loss, averagePrecision, and specificity of each fold first.
        sum_precision.append(evaluation_measures["Precision"])
        sum_recall.append(evaluation_measures["Recall"])
        sum_accuracy.append(evaluation_measures["Accuracy"])
        sum_f1_scor.append(evaluation_measures["F1 score"])
        sum_micro_fscore.append(evaluation_measures["f-1 Micro"])
        sum_macro_fscore.append(evaluation_measures["f-1 Macro"])
        sum_hamming_loss.append(evaluation_measures["Hamming Loss"])
        sum_averagePrecision.append(evaluation_measures["averagePrecision"])
        sum_specificity.append(evaluation_measures["specificity "])
        precision_class.append(Cprecision)
        recall_class.append(Crecall)
        f1_score_class.append(Cf1)
        auc_class.append(CAUC)
        aupr_class.append(CAUPR)
        # shutil.rmtree(conf.eval.model_dir.split("/")[0])

    print("_________________________________kfolds Metrics____________________________________")
    save_sum('Improved TextRNN', sum_actual, sum_pred)

    print(Fore.BLUE + "k-fold  precision", sum(sum_precision) / fold)
    print(Fore.RED + str(sum_precision))
    print(Fore.BLUE + "k-fold recall", sum(sum_recall) / fold)
    print(Fore.RED + str(sum_recall))
    print(Fore.BLUE + "k-fold fscore", sum(sum_f1_scor) / fold)
    print(Fore.RED + str(sum_f1_scor))
    print(Fore.BLUE + "k-fold Micro Fscore", sum(sum_micro_fscore) / fold)
    print(Fore.RED + str(sum_micro_fscore))
    print(Fore.BLUE + "k-fold Macro Fscore", sum(sum_macro_fscore) / fold)
    print(Fore.RED + str(sum_macro_fscore))
    print(Fore.BLUE + "k-fold Accuracy", sum(sum_accuracy) / fold)
    print(Fore.RED + str(sum_accuracy))
    print(Fore.BLUE + "k-fold Hamming Loss", sum(sum_hamming_loss) / fold)
    print(Fore.RED + str(sum_hamming_loss))
    print(Fore.BLUE + "k-fold averagePrecision", sum(sum_averagePrecision) / fold)
    print(Fore.RED + str(sum_averagePrecision))
    print(Fore.BLUE + "k-flod specificity ",sum(sum_specificity) / fold)
    print(Fore.RED + str(sum_specificity))

    with open('cnn_results', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([str(ngram) + "Grams"])
        writer.writerow(
            ["rand", str(round(sum(sum_precision) / fold, 2)), str(round(sum(sum_recall) / fold,2)), str(round(sum(sum_f1_scor) / fold,2)),
             str(round(sum(sum_micro_fscore) / fold,2)), str(round(sum(sum_macro_fscore) / fold,2)), str(round(sum(sum_accuracy) / fold,2)),
             str(round(sum(sum_hamming_loss) / fold,2)), str(round(sum(sum_averagePrecision) / fold,2)),str(round(sum(sum_specificity) / fold,2))])

if __name__ == '__main__':
    t1= time.time()
    kfold = 10
    ngrams=4
    """
    Get super parameters from conf/train.json
    """
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    config.feature.max_char_len_per_token = ngrams
    train(config, "piRNA_dataset/" + str(ngrams), kfold, ngrams)
    t2 = time.time()
    print('Time:',t2-t1)
