from math import sqrt
import numpy as np

from algorithms.model import Model
from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL


class SVD(Model):
    def __init__(self, feature_initial=None, num_features=3):
        self.feature_initial = feature_initial
        self.num_features = num_features
        if self.feature_initial is None:
            self.feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                        self.num_features)

    def set_users_and_movies(self):
        pass

    def train(self, ratings, epochs=2):
        self.set_users_and_movies()
        for _ in range(epochs):
            self.update_features()
