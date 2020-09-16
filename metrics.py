import torch
from sklearn.metrics import f1_score, recall_score
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class RecallAt(Metric):

    def __init__(self, k, latent_space, output_transform=lambda x: x):
        self.latent_space = latent_space
        self._num_correct = None
        self._num_examples = None
        super(RecallAt, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RecallAt, self).reset()

    def update(self, output):
        y_pred, y = output
        
        nearest_k, _ = self.latent_space.query(y_pred, k)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]


def recall(model, space, data, rank=1):
    pass