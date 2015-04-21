import numpy as np
from unittest.mock import call, Mock

from algorithms import svd
from utils.constants import MOVIE_INDEX, USER_INDEX


MockThatAvoidsErrors = Mock
MockThatAvoidsLongRunTime = Mock
MockThatTracksCallsWithoutRunning = Mock


def initialize_model_with_simple_train_points_but_do_not_train(model):
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    model.initialize_users_and_movies()


def make_simple_test_points():
    test_ratings = ((1, 4, 0, 0),
                    (2, 5, 0, 1),
                    (3, 1, 0, 5),
                    (4, 2, 0, 0))
    return np.array(test_ratings)


def make_simple_train_points():
    train_ratings = ((1, 2, 0, 1),
                     (3, 4, 0, 2),
                     (5, 1, 0, 3),
                     (2, 3, 0, 4),
                     (4, 5, 0, 5),
                     (1, 3, 0, 1),
                     (5, 2, 0, 2))
    return np.array(train_ratings)


def test_svd_calculate_max_movie_returns_expected_number():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    expected_max_movie = np.amax(model.train_points[:, MOVIE_INDEX]) + 1
    actual_max_movie = model.calculate_max_movie()
    assert actual_max_movie == expected_max_movie


def test_svd_calculate_max_user_returns_expected_number():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    expected_max_user = np.amax(model.train_points[:, USER_INDEX]) + 1
    actual_max_user = model.calculate_max_user()
    assert actual_max_user == expected_max_user


def test_svd_calculate_prediction_error_returns_expected_error():
    from utils.data_io import get_user_movie_time_rating
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    for train_point in model.train_points:
        user, movie, _, rating = get_user_movie_time_rating(train_point)
        expected_error = rating - model.calculate_prediction(user, movie)
        actual_error = model.calculate_prediction_error(user, movie, rating)
        assert actual_error == expected_error


def test_svd_calculate_prediction_returns_expected_prediction():
    from utils.data_io import get_user_movie_time_rating
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.train(simple_train_points)
    simple_test_points = make_simple_test_points()
    for test_point in simple_test_points:
        user, movie, _, _ = get_user_movie_time_rating(test_point)
        expected_prediction = np.dot(model.users[user, :],
                                     model.movies[:, movie])
        actual_prediction = model.calculate_prediction(user, movie)
        assert actual_prediction == expected_prediction


def test_svd_init_can_create_instance_with_no_arguments():
    svd.SVD()


def test_svd_init_instances_are_model_instances():
    from algorithms.model import Model
    model = svd.SVD()
    assert isinstance(model, Model)


def test_svd_init_sets_custom_feature_initial_value():
    from random import random
    custom_feature_initial_value = random()
    model = svd.SVD(feature_initial=custom_feature_initial_value)
    assert model.feature_initial == custom_feature_initial_value


def test_svd_init_sets_custom_learn_rate():
    from random import random
    custom_learn_rate = random()
    model = svd.SVD(learn_rate=custom_learn_rate)
    assert model.learn_rate == custom_learn_rate


def test_svd_init_sets_custom_number_of_features():
    from random import random
    custom_number_of_features = random()
    model = svd.SVD(num_features=custom_number_of_features)
    assert model.num_features == custom_number_of_features


def test_svd_init_sets_default_feature_initial_value_for_custom_number():
    from math import sqrt
    from random import random
    from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
    custom_number_of_features = random()
    expected_feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                    custom_number_of_features)
    model = svd.SVD(num_features=custom_number_of_features)
    actual_feature_initial = model.feature_initial
    assert float(actual_feature_initial) == expected_feature_initial


def test_svd_init_sets_default_feature_initial_value_for_default_number():
    from math import sqrt
    from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
    default_num_features = 3
    expected_feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                    default_num_features)
    model = svd.SVD()
    actual_feature_initial = model.feature_initial
    assert float(actual_feature_initial) == expected_feature_initial


def test_svd_init_sets_default_learn_rate():
    default_learn_rate = 0.001
    model = svd.SVD()
    assert model.learn_rate == default_learn_rate


def test_svd_init_sets_default_number_of_features():
    default_num_features = 3
    model = svd.SVD()
    assert model.num_features == default_num_features


def test_svd_initialize_users_and_movies_sets_expected_num_users_and_movies():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    expected_num_users = model.calculate_max_user()
    expected_num_movies = model.calculate_max_movie()
    model.initialize_users_and_movies()
    actual_num_users = model.max_user
    actual_num_movies = model.max_movie
    np.testing.assert_array_equal(actual_num_users, expected_num_users)
    np.testing.assert_array_equal(actual_num_movies, expected_num_movies)


def test_svd_initialize_users_and_movies_sets_expected_users_movies_matrices():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    num_users = model.calculate_max_user()
    num_movies = model.calculate_max_movie()
    expected_users = np.full((num_users, model.num_features),
                             model.feature_initial)
    expected_movies = np.full((model.num_features, num_movies),
                              model.feature_initial)
    model.initialize_users_and_movies()
    actual_users = model.users
    actual_movies = model.movies
    np.testing.assert_array_equal(actual_users, expected_users)
    np.testing.assert_array_equal(actual_movies, expected_movies)


def test_svd_predict_returns_expected_ratings():
    from utils.data_io import get_user_movie_time_rating
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    simple_test_points = make_simple_test_points()
    num_test_points = simple_test_points.shape[0]
    expected_ratings = np.zeros(num_test_points)
    for i, test_point in enumerate(simple_test_points):
        user, movie, _, _ = get_user_movie_time_rating(test_point)
        expected_ratings[i] = model.calculate_prediction(user, movie)
    actual_ratings = model.predict(simple_test_points)
    np.testing.assert_array_equal(actual_ratings, expected_ratings)


def test_svd_set_train_points_sets_train_points_to_expected_matrix():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    np.testing.assert_array_equal(model.train_points, simple_train_points)


def test_svd_train_initializes_user_and_movie_feature_matrices():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.update_all_features = MockThatAvoidsLongRunTime()
    model.initialize_users_and_movies = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points)
    assert model.initialize_users_and_movies.call_count == 1


def test_svd_train_sets_ratings():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.initialize_users_and_movies = MockThatAvoidsErrors()
    model.update_all_features = MockThatAvoidsLongRunTime()
    model.set_train_points = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points)
    assert model.set_train_points.call_count == 1


def test_svd_train_updates_all_features_the_expected_number_of_times():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    number_of_epochs = 3
    model.update_all_features = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points, epochs=number_of_epochs)
    assert model.update_all_features.call_count == number_of_epochs


def test_svd_update_all_features_updates_each_feature_once_in_any_order():
    model = svd.SVD()
    model.update_feature = MockThatTracksCallsWithoutRunning()
    model.update_all_features()
    assert model.update_feature.call_count == model.num_features
    expected_calls = [call(feature) for feature in range(model.num_features)]
    model.update_feature.assert_has_calls(expected_calls, any_order=True)


def test_svd_update_feature_calculates_prediction_error_at_least_once():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.update_user_and_movie = MockThatAvoidsLongRunTime()
    for feature in range(model.num_features):
        model.calculate_prediction_error = MockThatTracksCallsWithoutRunning()
        model.update_feature(feature)
        assert model.calculate_prediction_error.call_count >= 1


def test_svd_update_feature_updates_user_movie_for_each_train_point_any_order():
    from utils.data_io import get_user_movie_time_rating
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    for feature in range(model.num_features):
        model.update_user_and_movie = MockThatTracksCallsWithoutRunning()
        model.update_feature(feature)
        expected_num_calls = model.train_points.shape[0]
        assert model.update_user_and_movie.call_count == expected_num_calls
        expected_calls = []
        for train_point in model.train_points:
            user, movie, time, rating = get_user_movie_time_rating(train_point)
            expected_calls.append(
                call(user, movie, feature,
                     model.calculate_prediction_error(user, movie, rating)))
        model.update_user_and_movie.assert_has_calls(expected_calls,
                                                     any_order=True)


def test_svd_update_user_and_movie_modifies_matrices_as_expected():
    from utils.data_io import get_user_movie_time_rating
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    for train_point in model.train_points:
        user, movie, _, rating = get_user_movie_time_rating(train_point)
        for feature in range(model.num_features):
            error = model.calculate_prediction_error(user, movie, rating)
            expected_users = np.copy(model.users)
            expected_movies = np.copy(model.movies)
            expected_user_change = (model.learn_rate * error *
                                    model.movies[feature, movie])
            expected_movie_change = (model.learn_rate * error *
                                     model.users[user, feature])
            expected_users[user, feature] += expected_user_change
            expected_movies[feature, movie] += expected_movie_change
            model.update_user_and_movie(user, movie, feature, error)
            actual_users = model.users
            actual_movies = model.movies
            np.testing.assert_array_equal(actual_users, expected_users)
            np.testing.assert_array_equal(actual_movies, expected_movies)


def test_svd_update_feature_in_c_modifies_users_and_movies_as_expected():
    c_model = svd.SVD()
    py_model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(c_model)
    initialize_model_with_simple_train_points_but_do_not_train(py_model)
    np.testing.assert_array_equal(c_model.train_points, py_model.train_points)
    np.testing.assert_array_equal(c_model.users, py_model.users)
    np.testing.assert_array_equal(c_model.movies, py_model.movies)
    for feature in range(c_model.num_features):
        c_model.update_feature_in_c(feature)
        py_model.update_feature(feature)
        np.testing.assert_array_equal(c_model.users, py_model.users)
        np.testing.assert_array_equal(c_model.movies, py_model.movies)
