import numpy as np
from sklearn.metrics import f1_score
from .base import BaseMetric


class F1Score(BaseMetric):
    def __init__(self):
        super().__init__('F1_Score')
        self.worst = -1

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return f1_score(label, all_pred)