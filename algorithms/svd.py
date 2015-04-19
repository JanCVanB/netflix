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
        self.num_users = 0
        self.num_movies = 0

    def calculate_prediction_error(self):
        pass

    def initialize_users_and_movies(self):
        self.num_users, self.num_movies = self.ratings.shape
        self.users = np.full((self.num_users, self.num_features),
                             self.feature_initial)
        self.movies = np.full((self.num_features, self.num_movies),
                              self.feature_initial)

    def train(self, ratings, epochs=2):
        self.ratings = ratings
        self.initialize_users_and_movies()
        for _ in range(epochs):
            self.update_features()

    def update_feature(self, feature):
        self.calculate_prediction_error()
        for user in range(self.num_users):
            self.update_user(user)
        for movie in range(self.num_movies):
            self.update_movie(movie)

    def update_features(self):
        for feature in range(self.num_features):
            self.update_feature(feature)
