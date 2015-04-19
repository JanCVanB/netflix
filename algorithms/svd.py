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
        self.train_points = None
        self.num_users = 0
        self.num_movies = 0

    def calculate_num_movies(self):
        return np.amax(self.train_points[:, 1]) + 1

    def calculate_num_users(self):
        return np.amax(self.train_points[:, 0]) + 1

    def calculate_prediction(self, user, movie):
        return np.dot(self.users[user, :], self.movies[:, movie])

    def calculate_prediction_error(self, user, movie, rating):
        return rating - self.calculate_prediction(user, movie)

    def initialize_users_and_movies(self):
        self.num_users = self.calculate_num_users()
        self.num_movies = self.calculate_num_movies()
        self.users = np.full((self.num_users, self.num_features),
                             self.feature_initial)
        self.movies = np.full((self.num_features, self.num_movies),
                              self.feature_initial)

    def iterate_train_points(self):
        for train_point in self.train_points:
            if train_point[3] > 0:
                yield train_point

    def predict(self, test_points):
        num_test_points = test_points.shape[0]
        predictions = np.zeros(num_test_points)
        for i, test_point in enumerate(test_points):
            user, movie, _, _ = test_point
            predictions[i] = self.calculate_prediction(user, movie)
        return predictions

    def set_train_points(self, train_points):
        self.train_points = train_points

    def train(self, train_points, epochs=2):
        self.set_train_points(train_points)
        self.initialize_users_and_movies()
        for _ in range(epochs):
            self.update_features()

    def update_feature(self, feature):
        for user, movie, _, rating in self.iterate_train_points():
            error = self.calculate_prediction_error(user, movie, rating)
            self.update_user_and_movie(user, movie, feature, error)

    def update_features(self):
        for feature in range(self.num_features):
            self.update_feature(feature)

    def update_user_and_movie(self, user, movie, feature, error):
        temporary_user_value = self.users[user, feature]
        self.users[user, feature] += error * self.movies[feature, movie]
        self.movies[feature, movie] += error * temporary_user_value
