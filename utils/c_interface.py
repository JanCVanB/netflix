class CException(Exception):
    message = ''
    err_no = -1

    def __init__(self, err_no, message=''):
        self.err_no = err_no
        self.message = message

    def __str__(self):
        if self.message == '':
            return ('An Unknown C exception occurred! Error: {}'
                    .format(self.err_no))
        else:
            return '{}. (Error {})'.format(self.message, self.err_no)


def c_svd_update_feature(train_points, users, user_offsets, movies, residuals,
                         movie_averages, feature, num_features, learn_rate, k_factor):
    import ctypes
    from ctypes import c_void_p, c_int32, c_float
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    num_train_points = train_points.shape[0]
    num_users = users.shape[0]
    num_movies = movies.shape[0]
    library_file_name = 'svd.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_lib = ctypes.cdll.LoadLibrary(library_file_path)
#    svd_lib.c_update_feature.argtypes = [c_void_p, c_int32, c_void_p, c_void_p, c_int32, c_void_p,
#                                         c_void_p, c_int32, c_void_p, c_float, c_int32, c_int32]
#    svd_lib.c_update_feature.restype = c_int32
    c_update_feature = svd_lib.c_update_feature
    returned_value = c_update_feature(
        c_void_p(train_points.ctypes.data),    # (void*) train_points
        c_int32(num_train_points),             # (int)   num_train_points
        c_void_p(users.ctypes.data),           # (void*) users
        c_void_p(user_offsets.ctypes.data),    # (void*) user_offsets
        c_int32(num_users),                    # (int)   num_users
        c_void_p(movies.ctypes.data),          # (void*) movies
        c_void_p(movie_averages.ctypes.data),  # (void*) movie_averages
        c_int32(num_movies),                   # (int)   num_movies
        c_void_p(residuals.ctypes.data),       # (void*) residuals
        c_float(learn_rate),                   # (float) learn_rate
        c_int32(feature),                      # (int)   feature
        c_int32(num_features),                 # (int)   num_features
        c_float(k_factor)                      # (float) k_factor
    )
    if returned_value != 0:
        raise CException(returned_value)


def c_svd_euclidean_train_epoch(train_points, users, user_offsets, movies,
                                movie_averages, num_features, learn_rate,
                                k_factor):
    import ctypes
    from ctypes import c_void_p, c_int32, c_float
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    num_train_points = train_points.shape[0]
    num_users = users.shape[0]
    num_movies = movies.shape[0]
    library_file_name = 'svd_euclidean.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_euclidean_lib = ctypes.cdll.LoadLibrary(library_file_path)
    c_train_epoch = svd_euclidean_lib.c_train_epoch
    returned_value = c_train_epoch(
        c_void_p(train_points.ctypes.data),    # (void*) train_points
        c_int32(num_train_points),             # (int)   num_train_points
        c_void_p(users.ctypes.data),           # (void*) users
        c_void_p(user_offsets.ctypes.data),    # (void*) user_offsets
        c_int32(num_users),                    # (int)   num_users
        c_void_p(movies.ctypes.data),          # (void*) movies
        c_void_p(movie_averages.ctypes.data),  # (void*) movie_averages
        c_int32(num_movies),                   # (int)   num_movies
#        c_void_p(residuals.ctypes.data),       # (void*) residuals
        c_float(learn_rate),                   # (float) learn_rate
#        c_int32(feature),                      # (int)   feature
        c_int32(num_features),                 # (int)   num_features
        c_float(k_factor)                      # (float) k_factor
    )
    if returned_value != 0:
        raise CException(returned_value)
