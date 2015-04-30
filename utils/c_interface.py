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


def c_svd_update_feature(train_points, users, movies, feature, num_features, learn_rate):
    import ctypes
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    num_train_points = train_points.shape[0]
    num_users = users.shape[0]
    num_movies = movies.shape[0]
    library_file_name = 'svd.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_lib = ctypes.cdll.LoadLibrary(library_file_path)
    c_update_feature = svd_lib.c_update_feature
    returned_value = c_update_feature(
        ctypes.c_void_p(train_points.ctypes.data),  # (void*) train_points
        ctypes.c_int32(num_train_points),           # (int)   num_train_points
        ctypes.c_void_p(users.ctypes.data),         # (void*) users
        ctypes.c_int32(num_users),                  # (int)   num_users
        ctypes.c_void_p(movies.ctypes.data),        # (void*) movies
        ctypes.c_int32(num_movies),                 # (int)   num_movies
        ctypes.c_float(learn_rate),                 # (float) learn_rate
        ctypes.c_int32(feature),                    # (int)   feature
        ctypes.c_int32(num_features)                # (int)   num_features
    )
    if returned_value != 0:
        raise CException(returned_value)

def c_svd_update_feature_with_pointers(train_points, users, movies, feature, num_features, learn_rate):
    import ctypes
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    num_train_points = train_points.shape[0]
    num_users = users.shape[0]
    num_movies = movies.shape[0]
    library_file_name = 'svd.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_lib = ctypes.cdll.LoadLibrary(library_file_path)
    c_update_feature_with_pointers = svd_lib.c_update_feature_with_pointers
    returned_value = c_update_feature_with_pointers(
        ctypes.c_void_p(train_points.ctypes.data),  # (void*) train_points
        ctypes.c_int32(num_train_points),           # (int)   num_train_points
        ctypes.c_void_p(users.ctypes.data),         # (void*) users
        ctypes.c_int32(num_users),                  # (int)   num_users
        ctypes.c_void_p(movies.ctypes.data),        # (void*) movies
        ctypes.c_int32(num_movies),                 # (int)   num_movies
        ctypes.c_float(learn_rate),                 # (float) learn_rate
        ctypes.c_int32(feature),                    # (int)   feature
        ctypes.c_int32(num_features)                # (int)   num_features
    )
    if returned_value != 0:
        raise CException(returned_value)

def c_svd_update_feature_with_residuals(train_points, users, movies, residuals, feature, num_features, learn_rate):
    import ctypes
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    num_train_points = train_points.shape[0]
    num_users = users.shape[0]
    num_movies = movies.shape[0]
    library_file_name = 'svd.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_lib = ctypes.cdll.LoadLibrary(library_file_path)
    c_update_feature_with_residuals = svd_lib.c_update_feature_with_residuals
    returned_value = c_update_feature_with_residuals(
        ctypes.c_void_p(train_points.ctypes.data),  # (void*) train_points
        ctypes.c_int32(num_train_points),           # (int)   num_train_points
        ctypes.c_void_p(users.ctypes.data),         # (void*) users
        ctypes.c_int32(num_users),                  # (int)   num_users
        ctypes.c_void_p(movies.ctypes.data),        # (void*) movies
        ctypes.c_int32(num_movies),                 # (int)   num_movies
        ctypes.c_void_p(residuals.ctypes.data),     # (void*) residuals
        ctypes.c_float(learn_rate),                 # (float) learn_rate
        ctypes.c_int32(feature),                    # (int)   feature
        ctypes.c_int32(num_features)                # (int)   num_features
    )
    if returned_value != 0:
        raise CException(returned_value)