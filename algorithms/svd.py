from math import sqrt
import numpy as np
import sys
from time import time

from algorithms.model import Model
from utils.c_interface import c_svd_update_feature, c_svd_update_feature_with_pointers, c_svd_update_feature_with_residuals
from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
from utils.constants import MOVIE_INDEX, USER_INDEX
from utils.data_io import get_user_movie_time_rating


class SVD(Model):
    def __init__(self, learn_rate=0.001, num_features=3, feature_initial=None):
        self.learn_rate = learn_rate
        self.num_features = num_features
        self.feature_initial = feature_initial
        if self.feature_initial is None:
            self.feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                        self.num_features)
        self.users = np.array([])
        self.movies = np.array([])
        self.residuals = np.array([])
        self.train_points = np.array([])
        self.max_user = 0
        self.max_movie = 0
        self.debug = False
        self.run_c = False

    def calculate_max_movie(self):
        return np.amax(self.train_points[:, MOVIE_INDEX]) + 1

    def calculate_max_user(self):
        return np.amax(self.train_points[:, USER_INDEX]) + 1

    def calculate_prediction(self, user, movie):
        return np.dot(self.users[user, :], self.movies[movie, :])

    def calculate_prediction_error(self, user, movie, rating):
        return rating - self.calculate_prediction(user, movie)

    def initialize_users_and_movies(self):
        self.max_user = self.calculate_max_user()
        self.max_movie = self.calculate_max_movie()
        self.users = np.full((self.max_user, self.num_features),
                             self.feature_initial, dtype=np.float32)
        self.movies = np.full((self.max_movie, self.num_features),
                              self.feature_initial, dtype=np.float32)

    def predict(self, test_points):
        num_test_points = test_points.shape[0]
        predictions = np.zeros(num_test_points, dtype=np.float32)
        for i, test_point in enumerate(test_points):
            user, movie, _, _ = get_user_movie_time_rating(test_point)
            predictions[i] = self.calculate_prediction(user, movie)
        return predictions

    def set_train_points(self, train_points):
        self.train_points = train_points
        num_train_points = train_points.shape[0] + 1
        self.residuals = np.zeros(num_train_points, dtype=np.float32)

    def train(self, train_points, epochs=1):
        self.set_train_points(train_points)
        self.initialize_users_and_movies()
        for epoch in range(epochs):
            if self.debug:
                print('Epoch #{}'.format(epoch + 1))
            self.update_all_features()

    def train_more(self, train_points=None, epochs=1):
        if train_points is not None:
            self.set_train_points(train_points)
        for epoch in range(epochs):
            if self.debug:
                print('Epoch #{}'.format(epoch + 1))
            self.update_all_features()

    def update_all_features(self):
        for feature in range(self.num_features):
            if self.debug:
                print('  Feature #{}'.format(feature + 1))
            if self.run_c:
                self.update_feature_in_c_with_residuals(feature)
            else:
                self.update_feature(feature)

    def update_feature(self, feature):
        if self.debug:
            print('    time left: ', end='')
            sys.stdout.flush()
            num_points = self.train_points.shape[0]
            cutoff = 5e5
            next_point = 1e5
            step = 0
            start = time()
        for index, train_point in enumerate(self.train_points):
            if self.debug:
                if next_point <= index < num_points - cutoff:
                    stop = time()
                    step += 1
                    elapsed_secs = (stop - start)
                    elapsed_minutes = elapsed_secs / 60
                    elapsed_time = (elapsed_minutes if elapsed_minutes > 1
                                    else elapsed_secs)
                    label = 'min' if elapsed_time == elapsed_minutes else 'sec'
                    print('{:.2g}{} '
                          .format(elapsed_time * (num_points - index) / index,
                                  label), end='')
                    sys.stdout.flush()
                    next_point = num_points - num_points / 2 ** step
                    if next_point >= num_points - cutoff:
                        print()
            user, movie, _, rating = get_user_movie_time_rating(train_point)
            error = self.calculate_prediction_error(user, movie, rating)
            self.update_user_and_movie(user, movie, feature, error)

    def update_feature_in_c(self, feature):
        c_svd_update_feature(self.train_points, self.users, self.movies,
                             feature, self.num_features, self.learn_rate)

    def update_feature_in_c_with_pointers(self, feature):
        c_svd_update_feature_with_pointers(self.train_points, self.users, self.movies,
                             feature, self.num_features, self.learn_rate)

    def update_feature_in_c_with_residuals(self, feature):
        c_svd_update_feature_with_residuals(self.train_points, self.users, self.movies, self.residuals,
                             feature, self.num_features, self.learn_rate)

    def update_user_and_movie(self, user, movie, feature, error):
        user_change = self.learn_rate * error * self.movies[movie, feature]
        movie_change = self.learn_rate * error * self.users[user, feature]
        self.users[user, feature] += user_change
        self.movies[movie, feature] += movie_change
