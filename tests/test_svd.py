import numpy as np


def help_get_simple_ratings_array():
    ratings = ((1, 0, 2),
               (0, 3, 0),
               (4, 0, 5),
               (0, 1, 0))
    return np.array(ratings)


def test_run_can_accept_simple_ratings_array_as_only_argument():
    from algorithms.svd import run
    simple_ratings_array = help_get_simple_ratings_array()
    run(simple_ratings_array)


def test_run_returns_numpy_array():
    from algorithms.svd import run
    output_array = run()
    assert isinstance(output_array, np.ndarray)


def test_run_with_no_arguments_returns_result_of_run_all():
    from algorithms.svd import run, run_all
    expected_array = run_all()
    output_array = run()
    np.testing.assert_array_equal(output_array, expected_array)


def test_run_with_numpy_array_argument_returns_result_of_run_custom():
    from algorithms.svd import run, run_custom
    simple_ratings_array = help_get_simple_ratings_array()
    expected_array = run_custom(simple_ratings_array)
    output_array = run(simple_ratings_array)
    np.testing.assert_array_equal(output_array, expected_array)
