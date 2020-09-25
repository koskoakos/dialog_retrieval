import torch
from sklearn.metrics import f1_score, recall_score
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class RecallAt(Metric):

    def __init__(self, latent_space, output_transform=lambda x: x):
        self.latent_space = latent_space
        self.correct = None
        self.total = None
        super(RecallAt, self).__init__(output_transform=output_transform)

    def reset(self):
        self.correct = {'r1': 0, 'r3': 0, 'r10': 0}
        self.total = 0.000001
        super(RecallAt, self).reset()

    def update(self, output):
        y_pred, y = output
        y_pred = y_pred.cpu().numpy()
        _, nearest_k = self.latent_space.query(y_pred, 10)
        approximate_truth = self.latent_space.get_arrays()[0][nearest_k]
        isin10 = (approximate_truth[:, :, None] == y_pred).all(-1).any(-1).any(-1)
        isin3 = (approximate_truth[:, :3, None] == y_pred).all(-1).any(-1).any(-1)
        isin1 = (approximate_truth[:, :1, None] == y_pred).all(-1).any(-1).any(-1)
        self.correct['r10'] += isin10.sum()
        self.correct['r3'] += isin3.sum()
        self.correct['r1'] += isin1.sum()
        self.total += y.shape[0]

    def compute(self):
        return {'r1': self.correct['r1'] / self.total, 
                'r3': self.correct['r3'] / self.total,
                'r10': self.correct['r10'] / self.total}


def recall(model, space, data, rank=1):
    pass

