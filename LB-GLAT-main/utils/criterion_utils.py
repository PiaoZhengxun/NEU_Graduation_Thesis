# -*- coding: utf-8 -*-


import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def accuracy_mask(output, data, mask):
    """The accuracy of GNNs (LB_GLAT)"""
    pred = output.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    return int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.


def targ_pro_mask(output, data, mask):
    """Labels and predicted values converted into binary classification"""
    pred_pro = F.softmax(output.cpu().numpy(), dim=1)[:, 0][mask]
    targ_label = data.y.cpu().numpy()[mask]
    assert (targ_label >= 2).sum() == 0
    targ_label = (~targ_label.astype(np.bool)).astype(np.int32)
    return targ_label, pred_pro


def targ_pro(output, y):
    """Labels and predicted values converted into binary classification"""
    pred_pro = F.softmax(output, dim=1)[:, 0].cpu().detach().numpy()
    targ_label = y.cpu().numpy()
    assert (targ_label >= 2).sum() == 0
    targ_label = (~targ_label.astype(np.bool_)).astype(np.int32)
    return targ_label, pred_pro


