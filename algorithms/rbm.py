import numpy as np

from algorithms.model import Model


class RBM(Model):
    def __init__(self, learn_rate=0.001, num_hidden=100, num_visible=100):
        self.learn_rate = learn_rate
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        # TODO: soft_max, weight_cost, momentum, final_momentum
        self.train_points = np.array([])

    def set_train_points(self, train_points):
        self.train_points = train_points
