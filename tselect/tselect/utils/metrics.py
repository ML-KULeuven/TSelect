import numpy as np
from sklearn.metrics import roc_auc_score

from TSelect.tsfuse.tsfuse.utils import encode_onehot


def auroc_score(y_true, y_proba):
    return roc_auc_score(encode_onehot(y_true), y_proba)
