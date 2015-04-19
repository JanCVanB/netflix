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

    def calculate_num_movies(self):
        return np.amax(self.ratings[:, 1]) + 1

    def calculate_num_users(self):
        return np.amax(self.ratings[:, 0]) + 1

    def calculate_prediction_error(self, user, movie, rating):
        return rating - np.dot(self.users[user, :], self.movies[:, movie])

    def initialize_users_and_movies(self):
        self.num_users = self.calculate_num_users()
        self.num_movies = self.calculate_num_movies()
        self.users = np.full((self.num_users, self.num_features),
                             self.feature_initial)
        self.movies = np.full((self.num_features, self.num_movies),
                              self.feature_initial)

    def iterate_training_points(self):
        for point in self.ratings:
            if point[3] > 0:
                yield point

    def set_ratings(self, ratings):
        self.ratings = ratings

    def train(self, ratings, epochs=2):
        self.set_ratings(ratings)
        self.initialize_users_and_movies()
        for _ in range(epochs):
            self.update_features()

    def update_feature(self, feature):
        for user, movie, _, rating in self.iterate_training_points():
            error = self.calculate_prediction_error(user, movie, rating)
            self.update_user(user, movie, feature, error)
            self.update_movie(user, movie, feature, error)

    def update_features(self):
        for feature in range(self.num_features):
            self.update_feature(feature)

    def update_movie(self, user, movie, feature, error):
        pass

    def update_user(self, user, movie, feature, error):
        pass
