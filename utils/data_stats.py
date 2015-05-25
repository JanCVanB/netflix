import numpy as np
import pickle
from utils.constants import USER_INDEX, MOVIE_INDEX, RATING_INDEX, BLENDING_RATIO, K_NEIGHBORS


class DataStats():
    def __init__(self):
        self.data_set = []
        self.mu_data_set = []
        self.num_users = None
        self.num_movies = None
        self.global_average = None
        self.rated_code_bytes_needed = (K_NEIGHBORS / 32) + 1 # does this truncate to integer???
        self.movie_averages = []
        self.movie_rating_count = []
        self.movie_rating_sum = []
        self.movie_rating_squared_sum = []
        self.similarity_coefficient = []
        self.similarity_matrix_sorted = []
        self.similarity_matrix_rated = []
        self.user_offsets = []
        self.user_rating_count = []
        self.user_offsets_sum = []


    def init_movie_and_user_arrays(self):
        self.movie_averages = np.zeros(shape=(self.num_movies,), dtype=np.float32)
        self.movie_rating_count = np.zeros(shape=(self.num_movies,), dtype=np.int32)
        self.movie_rating_sum = np.zeros(shape=(self.num_movies,), dtype=np.float32)
        self.movie_rating_squared_sum = np.zeros(shape=(self.num_movies,), dtype=np.float32)
        self.similarity_coefficient = np.zeros(shape=(self.num_movies, self.num_movies), dtype=np.float32)
        self.similarity_matrix_rated = np.zeros(shape=(len(self.data_set[:, USER_INDEX]),self.rated_code_bytes_needed),
                                                dtype=np.int32)
        self.similarity_matrix_sorted = np.zeros(shape=(self.num_movies, K_NEIGHBORS), dtype=np.int32)
        self.user_offsets_sum = np.zeros(shape=(self.num_users,))
        self.user_offsets = np.zeros(shape=(self.num_users,), dtype=np.float32)
        self.user_rating_count = np.zeros(shape=(self.num_users,), dtype=np.int32)


    def load_data_set(self, data_set, mu_data_set):
        self.data_set = data_set
        self.mu_data_set = mu_data_set
        self.num_users = np.amax(data_set[:, USER_INDEX]) + 1
        self.num_movies = np.amax(data_set[:, MOVIE_INDEX]) + 1


    def compute_stats(self):
        if self.data_set == []:
            raise Exception('No Data set loaded. Please use DataStats.load_data_set' +
                            '(data_set) to load a data set before calling compute_stats')
        else:
            self.init_movie_and_user_arrays()
            self.compute_movie_stats()
            self.compute_user_stats()
            try:
                self.compute_similarity_coefficient()
                print('Finised computing the similarity matrix')
                import pickle
                with open('similarity_coefficient_dump.pkl', 'wb+') as dump_file:
                    pickle.dump(self, dump_file)
                self.compute_nearest_neighbors()
                print('Finished computing the nearest neighbors')
            except Exception as the_exception:
                import pdb
                pdb.set_trace()


    def compute_movie_stats(self):
        squared_sum, simple_sum, simple_count = compute_simple_indexed_sum_and_count(
            data_values=self.data_set[:, RATING_INDEX],
            data_indices=self.data_set[:, MOVIE_INDEX]
        )
        global_average = compute_global_average_rating(data_set=self.data_set)
        self.movie_averages = compute_blended_indexed_averages(
            simple_sum=simple_sum,
            simple_count=simple_count,
            global_average=global_average
        )
        self.movie_rating_count = simple_count
        self.movie_rating_sum = simple_sum
        self.movie_rating_squared_sum = squared_sum
        self.global_average = global_average


    def compute_similarity_coefficient(self, similarity_factor=100):
        movie_averages = np.append(0, np.divide(self.movie_rating_sum[1:], self.movie_rating_count[1:]))
        std_deviation = self.compute_standard_deviation()
        rating_y_index = 0
        for movie_y in range(1, self.num_movies):
            rating_x_index = 0
            for movie_x in range(1, self.num_movies):
                correlation_factor = 0
                expected_value_sum = 0
                num_similar_ratings = 0
                for user_y_count in range(self.movie_rating_count[movie_y]):
                    rating_y_user_index = rating_y_index + user_y_count
                    if rating_x_index >= len(self.mu_data_set[:, MOVIE_INDEX]):
                        break
                    if rating_y_user_index >= len(self.mu_data_set[:, MOVIE_INDEX]):
                        break
                    user_y = self.mu_data_set[rating_y_user_index, USER_INDEX]
                    user_x = self.mu_data_set[rating_x_index, USER_INDEX]
                    while (user_y > user_x):
                        rating_x_index += 1
                        if rating_x_index >= len(self.mu_data_set[:, MOVIE_INDEX]):
                            break
                        if (self.mu_data_set[rating_x_index, MOVIE_INDEX] == movie_x):
                            user_x = self.mu_data_set[rating_x_index, USER_INDEX]
                        else:
                            rating_x_index -= 1
                            break
                    if user_x == user_y:
                        expected_value_sum += (self.mu_data_set[rating_x_index, RATING_INDEX] -
                            movie_averages[movie_x]) * (
                            self.mu_data_set[rating_y_user_index, RATING_INDEX] -
                            movie_averages[movie_y])
                        num_similar_ratings += 1
                        #print('expected sum #{}'.format(expected_value_sum))
                        #print('movie avg #{}'.format(movie_averages[movie_y]))
                        #print('movie x rating #{}'.format(self.mu_data_set[rating_x_index, RATING_INDEX]))
                        #print('movie y rating #{}'.format(self.mu_data_set[rating_y_user_index, RATING_INDEX]))
                        #print('user #{}'.format(user_x))
                rating_x_index += 1
                if num_similar_ratings > 0:
                    correlation_factor = expected_value_sum / num_similar_ratings
                    print('Correlation factor_first #{}'.format(correlation_factor))
                if std_deviation[movie_x] > 0 and std_deviation[movie_y] > 0:
                    correlation_factor /= (std_deviation[movie_y] * std_deviation[movie_x])
                    print('Correlation factor_second #{}'.format(correlation_factor))
                self.similarity_coefficient[movie_y, movie_x] = np.absolute(num_similar_ratings * correlation_factor /
                                                                            (num_similar_ratings + similarity_factor))
            if movie_y % 1000 == 1:
                print('Similarity coefficient for movie: {}, {}'.format(movie_y, self.similarity_coefficient[movie_y, movie_x]))
            rating_y_index += self.movie_rating_count[movie_y]
        print('Similarity coefficient matrix #{}'.format(self.similarity_coefficient))

    def compute_nearest_neighbors(self):
        current_user = 0
        current_user_index = 0
        for training_index, training_point in enumerate(self.data_set[:, MOVIE_INDEX]):
            if current_user != self.data_set[training_index, USER_INDEX]:
                current_user = self.data_set[training_index, USER_INDEX]
                current_user_index = training_index
            movie_of_rating = self.data_set[training_index, MOVIE_INDEX]
            movie_similarity = self.similarity_coefficient[:, movie_of_rating]
            sorted_similarity_indices = np.argsort(-movie_similarity)
            movies_the_user_rated = self.data_set[current_user_index:
                                                  current_user_index +
                                                  self.user_rating_count[current_user],
                                                  MOVIE_INDEX]
            user_ratings_of_similar_movies = np.zeros(shape=(K_NEIGHBORS,), dtype=np.int32)
            user_ratings_of_similar_movies[np.in1d(sorted_similarity_indices, movies_the_user_rated)] = 1
            encoded_rating_binary = self.encode_nearest_neighbors(user_ratings_of_similar_movies)
            self.similarity_matrix_rated[training_index, :] = encoded_rating_binary
            if training_index % 1000000 == 0:
                print('Computing nearest neighbor of index {}'.format(training_index))
                print('Sorted similarity indices {}'.format(sorted_similarity_indices))
                print('Movies the user rated {}'.format(movies_the_user_rated))
                print('User ratings of similar movies {}'.format(user_ratings_of_similar_movies))
                print('Encoded binary ratings array {}'.format(encoded_rating_binary))

    def encode_nearest_neighbors(self, user_rated__binary_array):
        current_integer_slot = 0
        encoded_rated_binary_array = np.zeros(shape=(self.rated_code_bytes_needed,), dtype=np.int32)
        for binary_rating in range(len(user_rated__binary_array)):
            current_code = encoded_rated_binary_array[current_integer_slot]
            current_code = current_code << 1
            if user_rated__binary_array[binary_rating] != 0:
                current_code += 1
            encoded_rated_binary_array[current_integer_slot] = current_code
            if binary_rating % 32 == 0 and binary_rating != 0:
                current_integer_slot += 1
        return encoded_rated_binary_array


    def compute_standard_deviation(self):
        std_deviation = np.zeros(shape=(self.num_movies,), dtype=np.float32)
        print('Move averages squared #{}'.format(self.movie_rating_sum))
        print('Movie squared averages #{}'.format(self.movie_rating_squared_sum))
        print('Movie rating count #{}'.format(self.movie_rating_count))
        print('Movie rating count non finite #{}'.format(self.movie_rating_count[~np.isfinite(self.movie_rating_count)]))
        print('Movie squared averages non finite #{}'.format(self.movie_rating_squared_sum[~np.isfinite(self.movie_rating_count)]))
        print('Movie averages squared non finite #{}'.format(self.movie_rating_sum[~np.isfinite(self.movie_rating_count)]))
        movie_averages_squared = np.append(0, np.divide(self.movie_rating_sum[1:], self.movie_rating_count[1:]) ** 2)
        movie_squared_averages = np.append(0, np.divide(self.movie_rating_squared_sum[1:], self.movie_rating_count[1:]))
        np.sqrt(movie_squared_averages - movie_averages_squared, std_deviation)
        return std_deviation

    def compute_user_stats(self):
        simple_offsets = compute_offsets(
            data_indices=self.data_set[:, MOVIE_INDEX],
            data_values=self.data_set[:, RATING_INDEX],
            averages=self.movie_averages)
        _, simple_sum, simple_count = compute_simple_indexed_sum_and_count(
            data_indices=self.data_set[:, USER_INDEX],
            data_values=simple_offsets,
        )
        user_offset_global_average = np.sum(simple_sum) / np.sum(simple_count)
        if user_offset_global_average == np.nan:
            raise Exception('Error NaN in global average of offsets')
        self.user_offsets = compute_blended_indexed_averages(
            simple_sum=simple_sum,
            simple_count=simple_count,
            global_average=user_offset_global_average
        )
        self.user_offsets_sum = simple_sum
        self.user_rating_count = simple_count

    def get_baseline(self, user, movie):
        if self.movie_averages == []:
            raise Exception('Cannot get baseline: Missing Movie averages!')
        if self.user_offsets == []:
            raise Exception('Cannot get baseline: Missing user offsets!')
        mov_avg = self.movie_averages[movie]
        usr_off = self.user_offsets[user]
        return mov_avg + usr_off

    def write_stats_to_file(self, file_path):
        self.data_set = []
        self.mu_data_set = []
        self.similarity_coefficient = []
        pickle.dump(self, file=open(file_path, 'wb'))


def compute_simple_indexed_sum_and_count(data_indices, data_values):
    if data_indices.shape != data_values.shape:
        raise ValueError('Error! Shapes of index array and data array are not the same!')
    array_length = np.amax(data_indices) + 1
    data = zip(data_indices, data_values)
    indexed_squared_sum = np.zeros(shape=(array_length,), dtype=np.float32)
    indexed_sum = np.zeros(shape=(array_length,), dtype=np.float32)
    indexed_count = np.zeros(shape=(array_length,), dtype=np.int32)
    for index, value in data:
        indexed_squared_sum[index] += value ** 2
        indexed_sum[index] += value
        indexed_count[index] += 1
    for i, x in enumerate(indexed_count):
        if x == 0:
            print('I found a Zero!! {}, {}'.format(i, x))
    return indexed_squared_sum, indexed_sum, indexed_count


def compute_offsets(data_values, data_indices, averages):
    offsets = np.zeros(shape=data_values.shape, dtype=np.float32)
    for index, value in enumerate(data_values):
        offsets[index] += value - averages[data_indices[index]]
    return offsets


def compute_blended_indexed_averages(simple_sum, simple_count, global_average):
    return np.array([((global_average * BLENDING_RATIO + simple_sum[i]) / (BLENDING_RATIO + simple_count[i]))
                     for i in range(len(simple_sum))], dtype=np.float32)


def compute_global_average_rating(data_set):
    from utils.constants import RATING_INDEX

    return np.mean(data_set[:, RATING_INDEX])


def load_stats_from_file(file_path):
    pickle_file = open(file_path, 'rb')
    stats_object = pickle.load(pickle_file)
    return stats_object

