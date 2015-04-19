import numpy as np

from algorithms import svd


def help_get_simple_ratings_array():
    ratings = ((1, 0, 2),
               (0, 3, 0),
               (4, 0, 5),
               (0, 1, 0))
    return np.array(ratings)


def test_run_can_accept_simple_ratings_as_only_argument():
    simple_ratings_array = help_get_simple_ratings_array()
    svd.run(simple_ratings_array)


def test_run_on_no_arguments_returns_numpy_array():
    output_array = svd.run()
    assert isinstance(output_array, np.ndarray)


def test_run_on_no_arguments_returns_result_of_run_all(monkeypatch):
    from random import random
    unique_value = random()
    dummy_run_all = lambda: unique_value
    monkeypatch.setattr(svd, 'run_all', dummy_run_all)

    expected_result = svd.run_all()
    actual_result = svd.run()
    assert actual_result == expected_result


def test_run_on_simple_ratings_returns_numpy_array():
    simple_ratings_array = help_get_simple_ratings_array()
    output_array = svd.run(simple_ratings_array)
    assert isinstance(output_array, np.ndarray)


def test_run_on_simple_ratings_returns_result_of_run_custom(monkeypatch):
    from random import random
    unique_value = random()
    dummy_run_custom = lambda ratings: unique_value
    monkeypatch.setattr(svd, 'run_custom', dummy_run_custom)

    simple_ratings_array = help_get_simple_ratings_array()
    expected_result = svd.run_custom(simple_ratings_array)
    actual_result = svd.run(simple_ratings_array)
    assert actual_result == expected_result
