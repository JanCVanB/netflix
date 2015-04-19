import numpy as np
from unittest.mock import call, Mock

from algorithms import svd


MockThatAvoidsErrors = Mock
MockThatAvoidsLongRunTime = Mock
MockThatTracksCallsWithoutRunning = Mock


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
                     (1, 3, 0, 0),
                     (5, 2, 0, 1),
                     (4, 1, 0, 2))
    return np.array(train_ratings)


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


def test_svd_set_ratings_sets_ratings_to_expected_matrix():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    np.testing.assert_array_equal(model.train_points, simple_train_points)


def test_svd_train_sets_ratings():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.initialize_users_and_movies = MockThatAvoidsErrors()
    model.update_features = MockThatAvoidsLongRunTime()
    model.set_train_points = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points)
    assert model.set_train_points.call_count == 1


def test_svd_train_initializes_user_and_movie_feature_matrices():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.update_features = MockThatAvoidsLongRunTime()
    model.initialize_users_and_movies = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points)
    assert model.initialize_users_and_movies.call_count == 1


def test_svd_train_updates_features_the_expected_number_of_times():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    number_of_epochs = 3
    model.update_features = MockThatTracksCallsWithoutRunning()
    model.train(simple_train_points, epochs=number_of_epochs)
    assert model.update_features.call_count == number_of_epochs


def test_svd_initialize_users_and_movies_sets_expected_num_users_and_movies():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    expected_num_users = model.calculate_num_users()
    expected_num_movies = model.calculate_num_movies()
    model.initialize_users_and_movies()
    actual_num_users = model.num_users
    actual_num_movies = model.num_movies
    np.testing.assert_array_equal(actual_num_users, expected_num_users)
    np.testing.assert_array_equal(actual_num_movies, expected_num_movies)


def test_svd_initialize_users_and_movies_sets_expected_users_movies_matrices():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.set_train_points(simple_train_points)
    num_users = model.calculate_num_users()
    num_movies = model.calculate_num_movies()
    expected_users = np.full((num_users, model.num_features),
                             model.feature_initial)
    expected_movies = np.full((model.num_features, num_movies),
                              model.feature_initial)
    model.initialize_users_and_movies()
    actual_users = model.users
    actual_movies = model.movies
    np.testing.assert_array_equal(actual_users, expected_users)
    np.testing.assert_array_equal(actual_movies, expected_movies)


def initialize_model_with_simple_train_points_but_do_not_train(model):
    simple_train_points = make_simple_train_points()
    model.set_train_points(make_simple_train_points())
    model.initialize_users_and_movies()
    return simple_train_points


def test_svd_calculate_num_users_returns_expected_number():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    expected_num_users = np.amax(model.train_points[:, 0]) + 1
    actual_num_users = model.calculate_num_users()
    assert actual_num_users == expected_num_users


def test_svd_calculate_num_movies_returns_expected_number():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    expected_num_movies = np.amax(model.train_points[:, 1]) + 1
    actual_num_movies = model.calculate_num_movies()
    assert actual_num_movies == expected_num_movies


def test_svd_update_features_updates_each_feature_once_in_any_order():
    model = svd.SVD()
    model.update_feature = MockThatTracksCallsWithoutRunning()
    model.update_features()
    assert model.update_feature.call_count == model.num_features
    expected_calls = [call(f) for f in range(model.num_features)]
    model.update_feature.assert_has_calls(expected_calls, any_order=True)


def test_svd_update_feature_calculates_prediction_error_at_least_once():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.calculate_prediction_error = Mock()
    model.update_user_and_movie = Mock()
    feature = 0
    model.update_feature(feature)
    assert model.calculate_prediction_error.call_count >= 1


def test_svd_update_feature_updates_user_movie_for_each_train_point_any_order():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    model.update_user_and_movie = Mock()
    feature = 0
    model.update_feature(feature)
    training_points = list(model.iterate_train_points())
    expected_num_calls = len(training_points)
    assert model.update_user_and_movie.call_count == expected_num_calls
    expected_calls = [call(user, movie, feature,
                           model.calculate_prediction_error(user, movie, rating)
                           )
                      for user, movie, _, rating in training_points]
    model.update_user_and_movie.assert_has_calls(expected_calls, any_order=True)


def sort_first_then_second(iterable):
    return sorted(sorted(iterable, key=lambda x: x[1]), key=lambda x: x[0])


def assert_lists_of_tuples_are_equal(a, b):
    a = sort_first_then_second(a)
    b = sort_first_then_second(b)
    assert len(a) == len(b)
    assert all(all(a[i][j] == b[i][j]
                   for j in range(len(a[i])))
               for i in range(len(a)))


def test_svd_iterate_train_points_generates_expected_points():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    expected_training_points = []
    for point in model.train_points:
        rating = point[3]
        if rating > 0:
            expected_training_points.append(point)
    actual_training_points = model.iterate_train_points()
    assert_lists_of_tuples_are_equal(actual_training_points,
                                     expected_training_points)


def test_update_user_and_movie_modifies_matrices_as_expected():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    user, movie, _, rating = next(model.iterate_train_points())
    feature = 0
    error = model.calculate_prediction_error(user, movie, rating)
    expected_final_users = np.copy(model.users)
    expected_final_movies = np.copy(model.movies)
    expected_final_users[user, feature] += error * model.movies[feature, movie]
    expected_final_movies[feature, movie] += error * model.users[user, feature]
    model.update_user_and_movie(user, movie, feature, error)
    actual_final_users = np.copy(model.users)
    actual_final_movies = np.copy(model.movies)
    np.testing.assert_array_equal(actual_final_users, expected_final_users)
    np.testing.assert_array_equal(actual_final_movies, expected_final_movies)


def test_svd_calculate_prediction_error_returns_expected_error():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    user, movie, _, rating = model.train_points[0, :]
    assert rating > 0
    expected_error = (rating - model.calculate_prediction(user, movie))
    actual_error = model.calculate_prediction_error(user, movie, rating)
    assert actual_error == expected_error


def test_svd_predict_returns_expected_ratings():
    model = svd.SVD()
    initialize_model_with_simple_train_points_but_do_not_train(model)
    simple_test_points = make_simple_test_points()
    num_test_points = simple_test_points.shape[0]
    expected_ratings = np.zeros(num_test_points)
    for i, test_point in enumerate(simple_test_points):
        user, movie, _, _ = test_point
        expected_ratings[i] = model.calculate_prediction(user, movie)
    actual_ratings = model.predict(simple_test_points)
    np.testing.assert_array_equal(actual_ratings, expected_ratings)


def test_svd_calculate_prediction_returns_expected_prediction():
    model = svd.SVD()
    simple_train_points = make_simple_train_points()
    model.train(simple_train_points)
    simple_test_points = make_simple_test_points()
    user, movie, _, _ = simple_test_points[0, :]
    expected_prediction = np.dot(model.users[user, :], model.movies[:, movie])
    actual_prediction = model.calculate_prediction(user, movie)
    assert actual_prediction == expected_prediction
