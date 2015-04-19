from math import sqrt
import numpy as np

from algorithms.model import Model
from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL


class SVD(Model):
    def __init__(self, num_features=3, feature_initial=None):
        self.num_features = num_features
        self.feature_initial = feature_initial
        if self.feature_initial is None:
            self.feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                        self.num_features)
        self.users = None
        self.movies = None
        self.ratings = None

    def calculate_users_shape(self):
        return self.ratings.shape[0], self.num_features

    def calculate_movies_shape(self):
        return self.num_features, self.ratings.shape[1]

    def initialize_users_and_movies(self):
        self.users = np.full(self.calculate_users_shape(),
                             self.feature_initial)
        self.movies = np.full(self.calculate_movies_shape(),
                              self.feature_initial)

    def train(self, ratings, epochs=2):
        self.ratings = ratings
        self.initialize_users_and_movies()
        for _ in range(epochs):
            self.update_features()

    def update_features(self):
        for feature in range(self.num_features):
            self.update_feature(feature)
