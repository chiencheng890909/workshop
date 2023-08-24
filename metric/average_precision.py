import numpy as np
from sklearn.metrics import average_precision_score
from .base import BaseMetric


class AveragePrecision(BaseMetric):
    def __init__(self):
        super().__init__('Average_Precision')
        self.worst = -1

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return average_precision_score(label, all_pred)