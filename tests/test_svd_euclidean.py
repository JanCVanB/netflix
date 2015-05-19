import numpy as np


def make_simple_train_points():
    train_ratings = ((1, 2, 0, 1),
                     (3, 4, 0, 2),
                     (5, 1, 0, 3),
                     (2, 3, 0, 4),
                     (4, 5, 0, 5),
                     (1, 3, 0, 1),
                     (5, 2, 0, 2))
    return np.array(train_ratings, dtype=np.int32)


def test_svd_euclidean_init_instance_is_svd_instance():
    from algorithms.svd import SVD
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    assert isinstance(model, SVD)


def test_svd_euclidean_initialize_users_and_movies_arrays_are_not_constant():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    num_users = model.calculate_max_user()
    num_movies = model.calculate_max_movie()
    feature_initial = model.feature_initial
    num_features = model.num_features
    model.initialize_users_and_movies()
    actual_users = model.users
    undesired_users = np.full((num_users, num_features),
                              feature_initial, dtype=np.float32)
    undesired_movies = np.full((num_movies, num_features),
                               feature_initial, dtype=np.float32)
    actual_movies = model.movies
    try:
        np.testing.assert_array_equal(actual_users, undesired_users)
        np.testing.assert_array_equal(actual_movies, undesired_movies)
        raise Exception('Initialized movie and users array must not be constants')
    except AssertionError:
        pass
