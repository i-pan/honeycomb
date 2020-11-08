import numpy as np

from sklearn import metrics


def auc(t, p, **kwargs):
    # y_pred.shape = (N, 2)
    return {'auc': metrics.roc_auc_score(t, p[:,1])}