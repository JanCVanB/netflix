import numpy as np
from unittest.mock import Mock, call
from utils.data_io import get_user_movie_time_rating

MockThatAvoidsErrors = Mock
MockThatAvoidsLongRunTime = Mock
MockThatTracksCallsWithoutRunning = Mock


def initialize_model_with_simple_train_points_but_do_not_train(model):
    simple_train_points = make_simple_train_points()
    simple_stats = make_simple_stats()
    model.set_train_points(simple_train_points)
    model.set_stats(simple_stats)
    model.initialize_users_and_movies()


def make_simple_train_points():
    train_ratings = ((1, 2, 0, 1),
                     (3, 4, 0, 2),
                     (5, 1, 0, 3),
                     (2, 3, 0, 4),
                     (4, 5, 0, 5),
                     (1, 3, 0, 1),
                     (5, 2, 0, 2))
    return np.array(train_ratings, dtype=np.int32)


def make_simple_stats():
    from utils.data_stats import DataStats
    stats = DataStats()
    stats.load_data_set(data_set=make_simple_train_points())
    stats.compute_stats()
    return stats


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


def test_train_calls_train_epoch():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    model.train_epoch = MockThatTracksCallsWithoutRunning()
    simple_train_points = make_simple_train_points()
    simple_stats = make_simple_stats()
    model.train(stats=simple_stats, train_points=simple_train_points,
                epochs=1)
    assert model.train_epoch.call_count == 1


def test_train_sets_train_points_and_stats():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    simple_train_points = make_simple_train_points()
    simple_stats = make_simple_stats()
    model.initialize_users_and_movies = MockThatAvoidsErrors()
    model.train_epoch = MockThatAvoidsLongRunTime()
    model.set_train_points = MockThatTracksCallsWithoutRunning()
    model.set_stats = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points, stats=simple_stats)
    assert model.set_train_points.call_count == 1
    assert model.set_stats.call_count == 1


def test_train_epoch_calls_update_all_features_once_for_each_data_point():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.update_euclidean_all_features = MockThatTracksCallsWithoutRunning()
    model.train_epoch()
    expected_num_calls = model.train_points.shape[0]
    assert model.update_euclidean_all_features.call_count == expected_num_calls
    expected_calls = []
    for train_point in model.train_points:
        user, movie, _, rating = get_user_movie_time_rating(train_point)
        expected_calls.append(
            call(user=user, movie=movie, rating=rating)
        )
    model.update_euclidean_all_features.assert_has_calls(expected_calls,
                                                         any_order=True)


def test_update_all_features_calls_calculate_prediction_error_once():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.calculate_prediction_error = MockThatTracksCallsWithoutRunning(
        return_value=0.0
    )
    user, movie, _, rating = get_user_movie_time_rating(model.train_points[0, :])
    model.update_euclidean_all_features(user, movie, rating)
    assert model.calculate_prediction_error.call_count == 1


def test_update_euclidean_all_features_calls_update_user_and_movie_for_each_feature():
    from algorithms.svd_euclidean import SVDEuclidean
    model = SVDEuclidean()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.update_user_and_movie = MockThatTracksCallsWithoutRunning()
    user, movie, _, rating = get_user_movie_time_rating(model.train_points[0, :])
    model.update_euclidean_all_features(user=user, movie=movie, rating=rating)
    expected_num_calls = model.num_features
    assert model.update_user_and_movie.call_count == expected_num_calls
    expected_calls = []
    for feature in range(model.num_features):
        user, movie, _, rating = get_user_movie_time_rating(model.train_points[0, :])
        expected_calls.append(
            call(user=user, movie=movie, feature=feature,
                 error=model.calculate_prediction_error(user, movie, rating)
                 )
        )
    model.update_user_and_movie.assert_has_calls(expected_calls,
                                                         any_order=True)