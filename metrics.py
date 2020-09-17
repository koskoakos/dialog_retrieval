import torch
from sklearn.metrics import f1_score, recall_score
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class RecallAt(Metric):

    def __init__(self, k, latent_space, output_transform=lambda x: x):
        self.latent_space = latent_space
        self.k = k
        self.correct = None
        self.total = None
        super(RecallAt, self).__init__(output_transform=output_transform)

    def reset(self):
        self.correct = 0
        self.total = 0
        super(RecallAt, self).reset()

    def update(self, output):
        y_pred, y = output
        nearest_k, _ = self.latent_space.query(y_pred, self.k)
        isin_k = (nearest_k[..., None] == y).any(-1).nonzero()
        self.correct += isin_k
        self.total += y.size(0)

    def compute(self):
        return self.correct / self.total


def recall(model, space, data, rank=1):
    pass