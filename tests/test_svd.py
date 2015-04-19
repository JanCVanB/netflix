import numpy as np
from unittest.mock import call, Mock

from algorithms import svd


def help_get_simple_ratings():
    ratings = ((1, 0, 2),
               (0, 3, 0),
               (4, 0, 5),
               (0, 1, 0))
    return np.array(ratings)


def test_svd_can_create_instance_with_no_arguments():
    svd.SVD()


def test_svd_instances_are_model_instances():
    from algorithms.model import Model
    model = svd.SVD()
    assert isinstance(model, Model)


def test_svd_with_no_arguments_sets_default_number_of_features():
    default_num_features = 3
    model = svd.SVD()
    assert model.num_features == default_num_features


def test_svd_sets_custom_number_of_features():
    custom_number_of_features = 999
    model = svd.SVD(num_features=custom_number_of_features)
    assert model.num_features == custom_number_of_features


def test_svd_sets_default_feature_initial_value_for_default_number():
    from math import sqrt
    from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
    default_num_features = 3
    expected_feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                    default_num_features)
    model = svd.SVD()
    actual_feature_initial = model.feature_initial
    assert float(actual_feature_initial) == expected_feature_initial


def test_svd_sets_default_feature_initial_value_for_custom_number():
    from math import sqrt
    from utils.constants import ALGORITHM_DEFAULT_PREDICTION_INITIAL
    custom_number_of_features = 999
    expected_feature_initial = sqrt(ALGORITHM_DEFAULT_PREDICTION_INITIAL /
                                    custom_number_of_features)
    model = svd.SVD(num_features=custom_number_of_features)
    actual_feature_initial = model.feature_initial
    assert actual_feature_initial == expected_feature_initial


def test_svd_sets_custom_user_feature_init_value():
    custom_feature_initial_value = 0.2
    model = svd.SVD(feature_initial=custom_feature_initial_value)
    assert model.feature_initial == custom_feature_initial_value


def test_svd_train_initializes_user_and_movie_feature_matrices():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    model.initialize_users_and_movies = Mock()
    model.update_features = Mock()
    model.train(simple_ratings)
    assert model.initialize_users_and_movies.call_count == 1


def test_svd_train_updates_features_the_expected_number_of_times():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    number_of_epochs = 3
    model.update_features = Mock()
    model.train(simple_ratings, epochs=number_of_epochs)
    assert model.update_features.call_count == number_of_epochs


def test_svd_initialize_users_and_movies_fills_matrices_with_default_value():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    model.update_features = Mock()
    model.train(simple_ratings)
    expected_users = np.full(model.calculate_users_shape(),
                             model.feature_initial)
    expected_movies = np.full(model.calculate_movies_shape(),
                              model.feature_initial)
    model.initialize_users_and_movies()
    actual_users = model.users
    actual_movies = model.movies
    np.testing.assert_array_equal(actual_users, expected_users)
    np.testing.assert_array_equal(actual_movies, expected_movies)


def test_svd_calculate_users_shape_returns_expected_shape():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    expected_users_shape = (simple_ratings.shape[0], model.num_features)
    model.update_features = Mock()
    model.train(simple_ratings)
    actual_users_shape = model.calculate_users_shape()
    assert actual_users_shape == expected_users_shape


def test_svd_calculate_movies_shape_returns_expected_shape():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    expected_movies_shape = (model.num_features, simple_ratings.shape[1])
    model.update_features = Mock()
    model.train(simple_ratings)
    actual_movies_shape = model.calculate_movies_shape()
    assert actual_movies_shape == expected_movies_shape


def test_update_features_updates_each_feature_once():
    model = svd.SVD()
    model.update_feature = Mock()
    model.update_features()
    assert model.update_feature.call_count == model.num_features
    expected_calls = [call(f) for f in range(model.num_features)]
    model.update_feature.assert_has_calls(expected_calls)
