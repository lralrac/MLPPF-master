import sys
import torch
import os
import csv
import sklearn.metrics as sk
import matplotlib.pyplot as plt

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
    """
    # 每个标签的预测
    thresholds = np.linspace(0, 1, 100)
    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(actual.shape[1]):
        fpr, tpr, _ = sk.roc_curve(actual[:, i], pred[:, i],pos_label=1)
        roc_auc =sk.auc(fpr,tpr)
        threshold_idxs = np.round(thresholds * (len(fpr) - 1)).astype(int)
        fpr = fpr[threshold_idxs]
        tpr = tpr[threshold_idxs]
        roc_auc = roc_auc[threshold_idxs]
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc[i])
    mean_fpr = np.mean(fprs,axis=0)
    mean_tpr = np.mean(tprs,axis=0)
    mean_roc_auc = np.mean(roc_aucs,axis=0)
    plt.plot(mean_fpr, mean_tpr,label='ROC curve (area = %0.4)' % mean_roc_auc)
    fpr.clear()
    tpr.clear()
    roc_auc.clear()
    fpr["micro"] = mean_fpr
    tpr["micro"] = mean_tpr
    roc_auc["micro"] = mean_roc_auc
    """
    # 微平均
    fpr["micro"], tpr["micro"], _ = sk.roc_curve(actual.ravel(), pred.ravel())
    roc_auc["micro"] = sk.auc(fpr["micro"], tpr["micro"])
    print(roc_auc["micro"])
    #plt.plot(fpr["micro"], tpr["micro"], label='ROC curve (area = %0.2f)' % roc_auc["micro"])

    #plt.xlabel('FPR')
    #plt.ylabel('TPR')
    #plt.legend()
    #plt.show()
    return fpr, tpr, roc_auc

def save_ROC(Grams,kfold,fpr,tpr,roc_auc):
    #fpr = fpr.tolist()
    #tpr = tpr.tolist()
    with open('./ROC_results', 'a', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(["Grams","fpr","tpr"])
        #print(Grams,kfold,fpr["micro"].tolist(), tpr["micro"].tolist())
        writer.writerow([Grams,kfold,fpr["micro"].tolist(), tpr["micro"].tolist(),roc_auc["micro"].tolist()])
        file.close()
    print("ok")
def save_sum(mean,actual,pred):
    fpr,tpr ,roc_auc= create_sum_ROC(actual, pred)
    with open('./SUM_ROC', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([mean,fpr,tpr,roc_auc])
        file.close()
    print("ok")


def create_sum_ROC(actual,pred):
    actual = np.array(actual)
    pred = np.array(pred)
    fpr, tpr, _ = sk.roc_curve(actual.ravel(), pred.ravel())
    roc_auc = sk.auc(fpr, tpr)
    return fpr.tolist(),tpr.tolist(),roc_auc.tolist()

def show_sum_ROC():
    path = 'SUM_ROC'
    data = pd.read_csv(path)
    dict_mean = data["mean"].tolist()
    fpr= data["fpr"].tolist()
    tpr= data["tpr"].tolist()
    auc = data["roc_auc"].tolist()
    fpr = change(fpr)
    tpr = change(tpr)
    print(dict_mean)
    print(fpr)
    print(tpr)
    """
    dict_ROC_fpr = {}
    dict_ROC_tpr= {}
    dict_ROC_roc_auc = {}
    for z in range(len(dict_mean)):
            dict_ROC_fpr[dict_mean[z]] = []
            dict_ROC_tpr[dict_mean[z]] = []
            dict_ROC_roc_auc[dict_mean[z]] = []
    for q in range(len(dict_mean)):
        dict_ROC_fpr[dict_mean[q]].append( fpr[q])
        dict_ROC_tpr[dict_mean[q]].append(tpr[q])
        dict_ROC_roc_auc[dict_mean[q]].append(roc_auc[q])
    """
    for i in range(len(dict_mean)):
        plt.plot(fpr[i], tpr[i], label='%s ROC curve (area = %0.4f)' % (dict_mean[i],auc[i]))
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
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
        x[i] = x[i].strip()
        x[i] = x[i].strip("[]")
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

if __name__ == '__main__':

    show_sum_ROC()
