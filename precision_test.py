
import numpy as np

#=========================START SECTION predicted scores to one hot encoding ============================================


def pred_scores_converter(a, threshold, top_k):
    a = list(a)
    xi_pred = []
    temp_list = [0]*top_k
    count = 0
    for k in a:
        k = list(k)
        for i in k:
            if i > threshold:
                # print(k.index(i))
                temp_list.insert(k.index(i), 1)
            if len(temp_list) > top_k:
                temp_list.pop()
        xi_pred.append(temp_list)
        temp_list = [0]*top_k
    return xi_pred

#=========================END SECTION predicted scores to one hot encoding ============================================

#=========================START SECTION actual labels to one hot encoding ============================================


def true_label_convert(b, top_k):
    b = list(b)
    yi_true = []
    temp_list = [0]*top_k
    count = 0
    for k in b:
        k = list(k)
        k = list(np.sort(k))
        for i in k:
            if k != '':
                # print(i)
                temp_list.insert(i,1)
            if len(temp_list) > top_k:
                temp_list.pop()
        yi_true.append(temp_list)
        temp_list = [0]*top_k

    print(yi_true)
    return yi_true

#=========================END SECTION actual labels to one hot encoding ============================================
def take_values(predicted,actual, threshold, top_k):
    a, b = predicted,actual
    return pred_scores_converter(a, threshold, top_k),true_label_convert(b, top_k)
