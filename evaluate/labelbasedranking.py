from sklearn.metrics import roc_curve, auc,roc_auc_score
def AucMacro(y_test, predictions):
    """
    AUC Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    aucMacro : float
        AUC Macro
    """
    aucMacro = roc_auc_score(y_test, predictions, average='macro')
    return aucMacro

def AucMicro(y_test, predictions):
    """
    AUC Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    aucMicro : float
        AUC Micro
    """
    aucMicro = roc_auc_score(y_test, predictions, average='micro')

    return aucMicro