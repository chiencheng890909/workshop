import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BaseMetric


class AUC(BaseMetric):
    def __init__(self):
        super().__init__("AUC")
        self.worst = -1

    def __call__(self, prob, label):
        all_prob = prob.flatten()
        assert all_prob.shape == label.shape
        return roc_auc_score(label, all_prob)