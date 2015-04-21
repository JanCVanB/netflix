class C_Exception(Exception):
    message = ''
    err_no = -1

    def __init__(self, err_no, message=''):
        self.err_no = err_no
        self.message = message

    def __str__(self):
        if self.message == '':
            return 'An Unknown C exception occurred! Error: {}'.format(self.err_no)
        else:
            return '{}. (Error {})'.format(self.message, self.err_no)


def c_svd_update_feature(train_points, users, movies, feature, num_features, learn_rate):
    import ctypes
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    library_file_name = 'svd.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    svd_lib = ctypes.cdll.LoadLibrary(library_file_path)
    returned_value = svd_lib.svd(
        ctypes.c_void_p(train_points.ctypes.data),  # (void*) train_points
        ctypes.c_int32(train_points.shape[0]),      # (int)   num_training_points
        ctypes.c_void_p(users.ctypes.data),         # (void*) users
        ctypes.c_int32(users.shape[0]),             # (int)   num_users
        ctypes.c_void_p(movies.ctypes.data),        # (void*) movies
        ctypes.c_int32(movies.shape[0]),            # (int)   num_movies
        ctypes.c_float(learn_rate),                 # (float) learn_rate
        ctypes.c_int32(feature),                    # (int)   feature
        ctypes.c_int32(num_features)                # (int)   num_features
    )
    if returned_value != 0:
        raise C_Exception(returned_value)
