import numpy as np
from algorithms.svd import SVD
from utils.data_io import get_user_movie_time_rating
import sys


class SVDEuclidean(SVD):
    def initialize_users_and_movies(self):
        self.max_user = self.calculate_max_user()
        self.max_movie = self.calculate_max_movie()
        self.users = np.array(
            np.random.normal(loc=0.0, scale=self.feature_initial,
                             size=(self.max_user, self.num_features)),
            dtype=np.float32)
        self.movies = np.array(
            np.random.normal(loc=0.0, scale=self.feature_initial,
                             size=(self.max_movie, self.num_features)),
            dtype=np.float32)

    def train(self, train_points, stats, epochs=1):
        self.set_train_points(train_points=train_points)
        self.set_stats(stats=stats)
        self.initialize_users_and_movies()
        for epoch in range(epochs):
            if self.debug:
                print('Epoch {}'.format(epoch+1))
                print('movies: {}'.format(self.movies))
                print('users: {}'.format(self.users))
            if np.isnan(np.sum(self.movies)) or np.isnan(np.sum(self.users)):
                print("So, I found a NaN..")
                import pdb
                pdb.set_trace()
            self.train_epoch()

    def train_epoch(self):
        count = 0
        for train_point in self.train_points:
            count += 1
            if count % 100000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            user, movie, _, rating = get_user_movie_time_rating(train_point)
            self.update_euclidean_all_features(user=user, movie=movie,
                                               rating=rating)

    def update_euclidean_all_features(self, user, movie, rating):
        prediction_error = self.calculate_prediction_error(user=user, movie=movie,
                                                           rating=rating)
        for feature in range(self.num_features):
            self.update_user_and_movie(user=user, movie=movie, feature=feature,
                                       error=prediction_error)

