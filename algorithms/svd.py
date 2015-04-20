from math import sqrt
import numpy as np
import sys
from time import time

from algorithms.model import Model
from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
from utils.constants import MOVIE_INDEX, RATING_INDEX, USER_INDEX
from utils.data_io import get_user_movie_time_rating


class SVD(Model):
    def __init__(self, learn_rate=0.001, num_features=3, feature_initial=None):
        self.learn_rate = learn_rate
        self.num_features = num_features
        self.feature_initial = feature_initial
        if self.feature_initial is None:
            self.feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                        self.num_features)
        self.users = None
        self.movies = None
        self.train_points = None
        self.max_user = 0
        self.max_movie = 0
        self.debug = False

    def calculate_max_movie(self):
        return np.amax(self.train_points[:, MOVIE_INDEX]) + 1

    def calculate_max_user(self):
        return np.amax(self.train_points[:, USER_INDEX]) + 1

    def calculate_prediction(self, user, movie):
        return np.dot(self.users[user, :], self.movies[:, movie])

    def calculate_prediction_error(self, user, movie, rating):
        return rating - self.calculate_prediction(user, movie)

    def initialize_users_and_movies(self):
        self.max_user = self.calculate_max_user()
        self.max_movie = self.calculate_max_movie()
        self.users = np.full((self.max_user, self.num_features),
                             self.feature_initial)
        self.movies = np.full((self.num_features, self.max_movie),
                              self.feature_initial)

    def predict(self, test_points):
        num_test_points = test_points.shape[0]
        predictions = np.zeros(num_test_points)
        for i, test_point in enumerate(test_points):
            user, movie, _, _ = get_user_movie_time_rating(test_point)
            predictions[i] = self.calculate_prediction(user, movie)
        return predictions

    def set_train_points(self, train_points):
        self.train_points = train_points

    def train(self, train_points, epochs=2):
        self.set_train_points(train_points)
        self.initialize_users_and_movies()
        for epoch in range(epochs):
            if self.debug:
                print('Epoch #{}'.format(epoch + 1))
            self.update_all_features()

    def update_all_features(self):
        for feature in range(self.num_features):
            if self.debug:
                print('  Feature #{}'.format(feature + 1))
            self.update_feature(feature)

    def update_feature(self, feature):
        if self.debug:
            print('    time left: ', end='')
            sys.stdout.flush()
            num_points = self.train_points.shape[0]
            num_steps = 10
            progress = 0
            progress_step = int(num_points / num_steps)
            steps = 0
            start = time()
        for train_point_index, train_point in enumerate(self.train_points):
            if self.debug:
                if train_point_index >= progress + progress_step:
                    stop = time()
                    elapsed_secs = (stop - start)
                    elapsed_mins = elapsed_secs / 60
                    number = elapsed_mins if elapsed_mins > 1 else elapsed_secs
                    label = 'min' if number == elapsed_mins else 'sec'
                    print('{:.2g}{} '
                          .format(number * (num_steps - steps), label), end='')
                    sys.stdout.flush()
                    progress += progress_step
                    steps += 1
                    if steps == num_steps:
                        print()
                    start = time()
            user, movie, _, rating = get_user_movie_time_rating(train_point)
            error = self.calculate_prediction_error(user, movie, rating)
            self.update_user_and_movie(user, movie, feature, error)

    def update_user_and_movie(self, user, movie, feature, error):
        user_change = self.learn_rate * error * self.movies[feature, movie]
        movie_change = self.learn_rate * error * self.users[user, feature]
        self.users[user, feature] += user_change
        self.movies[feature, movie] += movie_change
