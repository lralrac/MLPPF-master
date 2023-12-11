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
用于获取训练、验证和测试数据的数据加载器(DataLoader)
dataset_name指定数据集类型，collate_name指定数据处理函数类型，conf是一个配置对象
"""
def get_data_loader(dataset_name, collate_name, conf):  #例 ClassificationDataset,ClassificationCollator,conf
    """Get data loader: Train, Validate, Test
    """
    # 将训练数据集生成一个词汇表，将文本文档转换为数字表示形式
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)#generate_dict是一个布尔值，表示是否需要生成词典,这个类会读取训练数据集文件，并将其转换为模型可以处理的格式

    """
    这个类会将每个样本转换为一个字典，并将这些字典打包成一个批次(batch)。
    在这个过程中，它会将样本的文本数据转换为数字化的表示，并将标签转换为One-Hot编码的形式。最后，它会返回一个包含批次数据的列表。
    例ClassificationCollator(conf,label_size)label_size是标签的数量（2）
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
    print("使用cuda")
    return model

"""
ClassificationTrainer的类，用于训练和评估文本分类模型。该类包含以下方法和属性：
__init__(self, label_map, logger, evaluator, conf, loss_fn)：类的构造函数，
用于初始化各种属性和参数，例如标签映射、日志记录器、评估器、配置和损失函数等。
train(self, data_loader, model, optimizer, stage, epoch)：用于训练模型的方法。
它接受一个数据加载器、模型、优化器、训练阶段和当前轮数作为输入，并返回训练的结果。
eval(self, data_loader, model, optimizer, stage, epoch)：用于评估模型性能的方法。
它接受一个数据加载器、模型、优化器、评估阶段和当前轮数作为输入，并返回评估的结果。
run(self, data_loader, model, optimizer, stage, epoch, mode=ModeType.EVAL)：用于运行训练或评估的方法。
它接受一个数据加载器、模型、优化器、阶段、当前轮数和模式（默认为评估模式）作为输入，并返回结果。
label_map：标签映射字典，用于将标签转换为数字化的表示。
logger：用于记录日志的对象。
evaluator：用于评估模型性能的对象。
conf：包含各种训练参数和配置的对象。
loss_fn：损失函数对象，用于计算模型的损失。
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
        model.update_lr(optimizer, epoch)#更新学习率
        model.train()#用于将模型设置为训练模式的方法
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        # 表示是多标签分类
        is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.

        for batch in data_loader:
            # logits用于存储神经网络模型的原始预测结果
            # model表示神经网络模型的实例
            # batch表示输入数据的批处理
            logits = model(batch)
            # print("在train.py输出logits:",logits)
            # print(make_dot(logits, params=dict(model.named_parameters())))
            # hierarchical classification
            # loss用于存储模型的损失值的对象
            # 在深度学习中,损失函数(loss function)用于衡量模型预测与真实目标之间的差距,训练模型的目的通常是最小化损失,使得模型尽可能地接近
            # 使用损失函数"self.loss_fn"来计算模型的预测结果"logits",这个损失值通常用于反向传播(backpropagation)以更新模型的权重和参数
            loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            #检查当前的运行模型是否为TRAIN(训练模式)
            if mode == ModeType.TRAIN:
                with torch.autograd.set_detect_anomaly(True):
                    # 清空优化器中存储的之前迭代的梯度信息,以便为当前迭代计算新的梯度
                    optimizer.zero_grad()
                    # 用于计算当前批次数据的损失,然后使用反向传播算法计算梯度,梯度用于调整模型的权重和参数,以减小损失值
                    loss.backward()
                    # 通过根据梯度更新模型的权重和参数，以减小损失值
                    optimizer.step()
                    continue
            total_loss += loss.item()
            # 这一段代码用于处理模型的原始预测结果的logits,将其转换为概率或分数的形式
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
                # print("输出result:",result)
            # 将模型的预测结果和真实标签收集到两个不同的列表predict_probs和standarf_labels
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

            """
            首先计算平均损失函数值。
            然后，调用self.evaluator.evaluate()函数计算模型在评估集上的性能指标。
            该函数的输入参数包括预测概率值predict_probs、标准标签standard_labels、标签映射label_map、阈值threshold、top-k值top_k
            以及是否进行多标签分类is_multi。
            函数返回的是一个元组，包含多个性能指标，如precision、recall、fscore等。
            """
        # 检查当前的运行模式是否为评估模式
        if mode == ModeType.EVAL:
            # 计算了总损失值的平均值
            total_loss = total_loss / num_batch
            # self.evaluator.evaluate是一个评估器（evaluator）对象的evaluate方法的调用
            # 评估器通常用于计算模型的性能指标如精度、召回率、F1分数
            # 评估器使用参数包括 模型的预测概率、真实标签、标签映射以及其他配置参数来计算性能指标

            #precision_list用于存储每各类别的精确值
            # recall_list用于存储每个类别的召回率值
            # fscore_list用于存储每个类别的F1分数值
            # right_list用于存储正确预测的样本数
            # predict_list用于存储模型的预测结果
            # standard_list用于存储真实的标签信息
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list,AUC_epoch,AUPR_epoch) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                     is_multi=is_multi)
            # print("--------------",stage,AUC_epoch,AUPR_epoch)

            # with open(csv_file_path, 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([stage, AUC_epoch, AUPR_epoch,total_loss])
            # 在终端输出每训练阶段，验证阶段，测试阶段每一个epoch评估下的性能指标
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
        数据集中的train.json，test.json，valid.json文件
        """
        conf.data.train_json_files=[os.path.join(dir, str(i),"train.json")]
        # ['MiRNA_dataset/4\\0\\train.json']
        conf.data.test_json_files=[os.path.join(dir, str(i),"test.json")]
        conf.data.validate_json_files=[os.path.join(dir, str(i),"valid.json")]

        logger = util.Logger(conf)#写入日志
        #保存每次fold的训练参数在checkpoint_dir_miRNA中
        if not os.path.exists(conf.checkpoint_dir):
            os.makedirs(conf.checkpoint_dir)

        model_name = conf.model_name#训练的模型PositionalCNN/TextRNN/Textcnn/Transformer
        dataset_name = "ClassificationDataset"

        """
        collate_name用于指定数据加载器(DataLoader)中的数据处理函数(collate_fn)。
        在这个脚本中，数据处理函数有两种，一种是FastTextCollator，另一种是ClassificationCollator。
        FastTextCollator用于处理FastText模型的数据，而ClassificationCollator用于处理其他分类模型的数据。
        collate_name的值会在get_data_loader()函数中被传递给DataLoader，以确定使用哪种数据处理函数。
        """
        collate_name = "FastTextCollator" if model_name == "FastText" \
            else "ClassificationCollator"
        train_data_loader, validate_data_loader, test_data_loader = \
            get_data_loader(dataset_name, collate_name, conf)

        print("-----------------------输出加载后的训练数据----------------------------------------")
        print(train_data_loader)
        count=0
        for flag in train_data_loader:
            if count == 0:
                print(flag)
            count+=1
        """
        empty_dataset
        用它来占位，以便在后续需要时进行替换。
        在实际使用中，有时候我们需要在训练过程中动态地调整数据集的大小，比如在训练过程中逐渐增加数据集的规模。
        这时，我们可以先使用empty_dataset创建一个空的数据集对象，然后在需要时再通过添加样本数据的方式来动态地扩充数据集。
        """

        empty_dataset = globals()[dataset_name](conf, [])
        # print("________________________输出empty_dataset_________________________________",empty_dataset)
        # print("____________________输出model_name_______________________",model_name)
        model = get_classification_model(model_name, empty_dataset, conf)
        print("__________________________输出model________________________________",model)
        loss_fn = globals()["ClassificationLoss"](
            label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
        if(model == "Transformer" or model =="Textcnn"):
            optimizer  = model.parameters()
        else:
            optimizer = get_optimizer(conf, model)
        evaluator = cEvaluator(conf.eval.dir)#(用于计算指标)
        trainer = globals()["ClassificationTrainer"](
            empty_dataset.label_map, logger, evaluator, conf, loss_fn)

        best_epoch = -1
        best_performance = 0
        model_file_prefix = conf.checkpoint_dir + "/" + model_name
        for epoch in range(conf.train.start_epoch,
                           conf.train.start_epoch + conf.train.num_epochs):
            start_time = time.time()
            trainer.train(train_data_loader, model, optimizer, "Train", epoch)
            # 对模型在训练过程中对训练集性能进行评估
            # 输出  Train performance at epoch 14 is precision: 0.883192, recall: 0.971586, fscore: 0.925283, macro-fscore: 0.919722, right: 2291, predict: 2594, standard: 2358.
            trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
            performance = trainer.eval(
                validate_data_loader, model, optimizer, "Validate", epoch)
            trainer.eval(test_data_loader, model, optimizer, "test", epoch)
            if performance[4] > best_performance:  # record the best model
                best_epoch = epoch
                best_performance = performance[4]
            save_checkpoint({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict(),
            }, model_file_prefix)#保存模型数据
            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

        # best model on validateion set
        best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
        best_file_name = model_file_prefix + "_best"
        shutil.copyfile(best_epoch_file_name, best_file_name)

        load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
                        optimizer)
        #这边使用最佳模型对测试集进行评估，得到结果
        # Best test performance at epoch 5 is precision: 0.953405, recall: 0.923611, fscore: 0.938272, macro-fscore: 0.934683, right: 266, predict: 279, standard: 288.
        trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch)

        # 通过 训练好的模型对test数据集进行测试，调用eval.py
        evaluation_measures, fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision,all_actual, all_pred,Cprecision,Crecall,Cf1,CAUC,CAUPR = kfold_eval(config)
        for z in range(len(all_actual)):
            sum_actual.append(all_actual[z])
            sum_pred.append(all_pred[z])
        save_ROC(ngram, i, fpr, tpr,roc_auc,prprecision,prrecall,praverage_precision)
        # 将每一折的Precision、Recall、Accuracy、F1 score、f-1 Micro、f-1 Macro、Hamming Loss、averagePrecision、specificity先存储起来
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
    save_sum('TextRNN_ATT+test', sum_actual, sum_pred)

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
    # for ngrams in range(3,5):
    """
    从conf/train.json中获取超参
    """

    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    config.feature.max_char_len_per_token = ngrams
    """
    with open('./ROC_results', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Grams","kfold","fpr","tpr","roc_auc"])
        file.close()
    """
    #训练 （超参，数据集，kfold,ngrams）

    train(config, "dataset_preprocessing/piRNA_dataset/" + str(ngrams), kfold, ngrams)
    t2 = time.time()

    print('Time:',t2-t1)
