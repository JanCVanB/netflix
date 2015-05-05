import numpy as np
import pickle
from utils.constants import USER_INDEX, MOVIE_INDEX
from utils.data_io import get_user_movie_time_rating


class DataStats():
    def load_data_set(self, data_set):
        self.data_set = data_set
        self.num_users = np.amax(data_set[:, USER_INDEX]) + 1
        self.num_movies = np.amax(data_set[:, MOVIE_INDEX]) + 1

    def __init__(self):
        self.data_set = []
        self.num_users = None
        self.num_movies = None
        self.average_of_all_movies = None
        self.movie_averages = []
        self.movie_rating_count = []
        self.user_offsets = []
        self.user_rating_count = []

    def compute_movie_stats(self):
        self.movie_averages = np.zeros(shape=(self.num_movies,), dtype=np.float32)
        self.movie_rating_count = np.zeros(shape=(self.num_movies,), dtype=np.int32)
        movie_ratings_sum = np.zeros(shape=(self.num_movies,))

        for data_point in self.data_set:
            _, movie_id, _, rating = get_user_movie_time_rating(data_point=data_point)
            self.movie_rating_count[movie_id] += 1
            movie_ratings_sum[movie_id] += rating

        for movie_id in range(0, self.num_movies):
            if self.movie_rating_count[movie_id] != 0:
                self.movie_averages[movie_id] = movie_ratings_sum[movie_id] / \
                    self.movie_rating_count[movie_id]
            else:
                self.movie_averages[movie_id] = np.nan
        self.average_of_all_movies = np.mean(np.ma.masked_array(self.movie_averages,
                                             np.isnan(self.movie_averages)))
        self.movie_averages[np.isnan(self.movie_averages)] = self.average_of_all_movies

    def compute_user_stats(self):
        if self.movie_averages == []:
            raise Exception('User statistics depend on movie stats. Please run DataStats.' +
                            'compute_movie_stats() prior to calling this function.')
        user_offsets_sum = np.zeros(shape=(self.num_users,))
        self.user_offsets = np.zeros(shape=(self.num_users,), dtype=np.float32)
        self.user_rating_count = np.zeros(shape=(self.num_users,), dtype=np.int32)
        for data_point in self.data_set:
            user_id, movie_id, _, rating = get_user_movie_time_rating(
                data_point=data_point)
            user_offsets_sum[user_id] += rating - self.movie_averages[movie_id]
            self.user_rating_count[user_id] += 1
        for user_id in range(0, self.num_users):
            if self.user_rating_count[user_id] != 0:
                self.user_offsets[user_id] = user_offsets_sum[user_id] / \
                    self.user_rating_count[user_id]
            else:
                self.user_offsets[user_id] = np.nan
        average_of_all_offsets = np.mean(np.ma.masked_array(self.user_offsets,
                                                            np.isnan(self.user_offsets)))
        self.user_offsets[np.isnan(self.user_offsets)] = average_of_all_offsets

    def get_baseline(self, user, movie):
        return self.movie_averages[movie] + self.user_offsets[user]

    def write_stats_to_file(self, file_path):
        pickle.dump(self, file=open(file_path, 'wb'))


def load_stats_from_file(file_path):
    pickle_file = open(file_path, 'rb')
    stats_object = pickle.load(pickle_file)
    return stats_object

