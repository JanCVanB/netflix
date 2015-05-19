import numpy as np
from algorithms.svd import SVD


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
