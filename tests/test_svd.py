import numpy as np
from unittest.mock import Mock

from algorithms import svd


def help_get_simple_ratings():
    ratings = ((1, 0, 2),
               (0, 3, 0),
               (4, 0, 5),
               (0, 1, 0))
    return np.array(ratings)


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


def test_svd_train_sets_user_and_movie_feature_matrices():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    model.set_users_and_movies = Mock()
    model.update_features = Mock()
    model.train(simple_ratings)
    assert model.set_users_and_movies.call_count == 1


def test_svd_train_updates_features_the_right_number_of_times():
    simple_ratings = help_get_simple_ratings()
    model = svd.SVD()
    number_of_epochs = 3
    model.set_u_and_v = Mock()
    model.update_features = Mock()
    model.train(simple_ratings, epochs=number_of_epochs)
    assert model.update_features.call_count == number_of_epochs
