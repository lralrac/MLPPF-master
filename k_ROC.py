import sys
import torch
import os
import csv
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc,precision_recall_curve

import util
import numpy as np
import pandas as pd

"""
fpr, tpr, thresholds = sklearn.metrics.roc_curve(actual, pred)
plt.plot(fpr, tpr, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
"""
def built_ROC(actual,pred):
    #actual = np.array([[1,0],[1,0],[1,0],[1,1],[0,1],[1,1],[0,1],[0,1],[1,1],[1,0]])
    #pred = np.array([[0.87, 0.96], [0.94, 0.24], [0.91, 0.20], [0.91, 0.96], [0.50, 0.91], [0.93, 0.91], [0.81, 0.86], [0.34, 0.78], [0.92, 0.20], [0.89, 0.40]])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()

    fpr["micro"], tpr["micro"], _ = sk.roc_curve(actual.ravel(), pred.ravel())
    roc_auc["micro"] = sk.auc(fpr["micro"], tpr["micro"])
    precision["micro"], recall["micro"], _ = sk.precision_recall_curve(actual.ravel(), pred.ravel())
    average_precision["micro"] = sk.average_precision_score(actual, pred, average="micro")
    print(roc_auc["micro"])
    print(average_precision["micro"])

    return fpr, tpr, roc_auc,precision,recall,average_precision

def save_ROC(Grams,kfold,fpr,tpr,roc_auc,precision,recall,average_precision):

    with open('./ROC_results', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([Grams,kfold,fpr["micro"].tolist(), tpr["micro"].tolist(),roc_auc["micro"].tolist(),precision["micro"].tolist(),recall["micro"].tolist(),average_precision["micro"].tolist()])
        file.close()
    print("ok")
def save_sum(mean,actual,pred):
    fpr,tpr ,roc_auc,precision,recall,average_precision= create_sum_ROC(actual, pred)
    with open('./SUM_ROC', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([mean,fpr,tpr,roc_auc,precision,recall,average_precision])
        file.close()
    print("ok")


def create_sum_ROC(actual,pred):
    actual = np.array(actual)
    pred = np.array(pred)
    fpr, tpr, _ = sk.roc_curve(actual.ravel(), pred.ravel())
    roc_auc = sk.auc(fpr, tpr)
    precision, recall, _ = sk.precision_recall_curve(actual.ravel(), pred.ravel())
    average_precision = sk.average_precision_score(actual, pred, average="micro")
    return fpr.tolist(),tpr.tolist(),roc_auc.tolist(),precision.tolist(),recall.tolist(),average_precision.tolist()

def show_sum_ROC():
    path = 'SUM_ROC'
    data = pd.read_csv(path)
    dict_mean = data["mean"].tolist()
    fpr= data["fpr"].tolist()
    tpr= data["tpr"].tolist()
    auc = data["roc_auc"].tolist()
    fpr = change(fpr)
    tpr = change(tpr)
    for i in range(len(dict_mean)):
        plt.plot(fpr[i], tpr[i], label='%s (AUC = %0.4f)' % (dict_mean[i],auc[i]))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    save_path = './picture/method_ROC.png'
    plt.savefig(save_path, dpi=300)

    plt.show()

def show_sum_PR():
    path = 'SUM_ROC'
    data = pd.read_csv(path)
    dict_mean = data["mean"].tolist()
    precision= data["precision"].tolist()
    recall= data["recall"].tolist()
    average_precision = data["average_precision"].tolist()
    precision = change(precision)
    recall = change(recall)

    for i in range(len(dict_mean)):
        plt.plot(recall[i], precision[i], label='%s (PRC = %0.4f)' % (dict_mean[i],average_precision[i]))

    plt.plot([0, 1], [1, 0], color='gray', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="lower left")
    save_path = './picture/method_PR.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

def loda_ROC():
    path = 'ROC_results'
    data = pd.read_csv(path)
    dict_Grams = data["Grams"].to_dict()
    listGrams = list(set(data["Grams"]))
    kfold = data["kfold"].tolist()
    fpr= data["fpr"].tolist()
    tpr= data["tpr"].tolist()
    roc_auc = data["roc_auc"].tolist()
    fpr = change(fpr)
    tpr = change(tpr)
    dict_ROC_kfold = {}
    dict_ROC_fpr = {}
    dict_ROC_tpr= {}
    dict_ROC_roc_auc = {}
    for i in range(len(listGrams)):
            dict_ROC_kfold[listGrams[i]] = []
            dict_ROC_fpr[listGrams[i]] = []
            dict_ROC_tpr[listGrams[i]] = []
            dict_ROC_roc_auc[listGrams[i]] = []
    for i in range(len(dict_Grams)):
        dict_ROC_kfold[dict_Grams[i]].append(kfold[i])
        dict_ROC_fpr[dict_Grams[i]].append( fpr[i])
        dict_ROC_tpr[dict_Grams[i]].append(tpr[i])
        dict_ROC_roc_auc[dict_Grams[i]].append(roc_auc[i])


    return dict_Grams,dict_ROC_kfold,dict_ROC_fpr,dict_ROC_tpr,dict_ROC_roc_auc


def change(x):
    for i in range(len(x)):
        x[i] = x[i].strip()  # 去除后面的换行元素
        x[i] = x[i].strip("[]")  # 去除列表的[]符号
        x[i] = x[i].split(",")
        x[i] = list(map(float, x[i]))

    return x

def show_all_ROC(dict_Grams ,dict_kfold,dict_fpr ,dict_tpr,dict_auc,show_grams,show_kfold):
    cont =-1
    if show_grams != -1:
        for i in range(len(dict_Grams)):
            if dict_Grams[i] == show_grams:
                cont = i
                break
        i = cont
        kfold = list(dict_kfold[dict_Grams[i]])
        fpr = list(dict_fpr[dict_Grams[i]])
        tpr = list(dict_tpr[dict_Grams[i]])
        auc = list(dict_auc[dict_Grams[i]])
        for j in range(len(kfold)):
            plt.plot(fpr[j], tpr[j], label='kfold = %d ROC curve (area = %0.4f)' % (kfold[j],auc[j]))
        plt.title('%d grams' % show_grams)
    else:
        setcont = set()
        for i in range(len(dict_Grams)):
            setcont.add(dict_Grams[i])
        print(setcont)
        for i in range(len(setcont)):
            t = dict_Grams[i]
            fpr = list(dict_fpr[dict_Grams[i]])
            tpr = list(dict_tpr[dict_Grams[i]])
            auc = list(dict_auc[dict_Grams[i]])
            print(t)
            auc = np.average(auc,axis=0)
            fpr = np.average(fpr,axis=0)
            tpr = np.average(tpr,axis=0)
            plt.plot(fpr,tpr , label='%d Grams ROC curve (area = %0.4f)' % (t,auc))

    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

def class_Metric(actual,pred,y_pred_b):


    y_true=actual
    y_pred=pred

    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list=[]
    prc_list=[]

    for class_index in range(y_true.shape[1]):

        precision = precision_score(y_true[:, class_index], y_pred[:, class_index])
        recall = recall_score(y_true[:, class_index], y_pred[:, class_index])
        f1 = f1_score(y_true[:, class_index], y_pred[:, class_index])

        fpr, tpr, thresholds = roc_curve(y_true[:, class_index], y_pred_b[:, class_index])
        roc_auc = auc(fpr, tpr)

        precision1, recall1, _ = precision_recall_curve(y_true[:, class_index], y_pred_b[:, class_index])
        prc_auc = auc(recall1, precision1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1)
        auc_list.append(roc_auc)
        prc_list.append(prc_auc)




    return precision_list,recall_list,f1_score_list,auc_list,prc_list

if __name__ == '__main__':

    #see = np.average(see, axis=0)
    p = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
    t = np.array([[9.63095963e-01,1.75727922e-02], [9.79060829e-01,3.15358713e-02], [7.85742998e-01,6.52132779e-02],[9.64212358e-01,4.93585095e-02],[9.42467809e-01,3.99554558e-02],[9.91547048e-01,3.48307751e-02],[8.61993790e-01,6.66181326e-01] ,[9.62756276e-01,1.89335495e-02],[9.84570563e-01,8.27044435e-03] ,[9.83874083e-01,1.49833700e-02],])
    show_sum_ROC()
    show_sum_PR()
