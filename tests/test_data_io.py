import itertools
import numpy as np
import os
import random
import types

from utils import constants, data_io, data_paths


def first_n_indices_are_correct(point_generator, number_of_points, index):
    all_data = data_io.data_points(data_paths.ALL_DATA_FILE_PATH)
    all_indices = data_io.indices(data_paths.ALL_INDEX_FILE_PATH)
    point_generator_first_n = itertools.islice(point_generator, 0,
                                               number_of_points)
    for data_point in point_generator_first_n:
        for some_point in all_data:
            this_index = next(all_indices)
            if this_index == index:
                assert data_point == some_point
                break


def test_all_points_returns_a_generator():
    assert isinstance(data_io.all_points(), types.GeneratorType)


def test_all_points_first_ten_are_correct():
    expected_all_points = data_io.data_points(data_paths.ALL_DATA_FILE_PATH)
    actual_all_points_first_n = itertools.islice(data_io.all_points(), 0, 10)
    for actual_data_point in actual_all_points_first_n:
        expected_data_point = next(expected_all_points)
        assert actual_data_point == expected_data_point


def test_base_points_returns_a_generator():
    assert isinstance(data_io.base_points(), types.GeneratorType)


def test_base_points_first_ten_are_correct():
    first_n_indices_are_correct(data_io.base_points(), number_of_points=10,
                                index=constants.BASE_INDEX)


def test_data_points_returns_a_generator():
    assert isinstance(data_io.data_points(data_paths.ALL_DATA_FILE_PATH),
                      types.GeneratorType)


def test_data_points_first_ten_are_correct():
    data_points_generator = data_io.data_points(data_paths.ALL_DATA_FILE_PATH)

    with open(data_paths.ALL_DATA_FILE_PATH) as all_data_file:
        for _ in range(10):
            point_from_file = next(all_data_file).strip().split()
            point_from_generator = next(data_points_generator)
            assert point_from_generator == point_from_file


def test_get_user_movie_time_rating_returns_correct_values():
    unique_user = random.random()
    unique_movie = random.random()
    unique_time = random.random()
    unique_rating = random.random()
    data_point = [0] * 4
    data_point[constants.USER_INDEX] = unique_user
    data_point[constants.MOVIE_INDEX] = unique_movie
    data_point[constants.TIME_INDEX] = unique_time
    data_point[constants.RATING_INDEX] = unique_rating
    user, movie, time, rating = data_io.get_user_movie_time_rating(data_point)
    assert user == unique_user
    assert movie == unique_movie
    assert time == unique_time
    assert rating == unique_rating


def test_hidden_points_first_ten_are_correct():
    first_n_indices_are_correct(data_io.hidden_points(), number_of_points=10,
                                index=constants.HIDDEN_INDEX)


def test_hidden_points_returns_a_generator():
    assert isinstance(data_io.hidden_points(), types.GeneratorType)


def test_indices_first_ten_correct():
    indices_generator = data_io.indices(data_paths.ALL_INDEX_FILE_PATH)

    with open(data_paths.ALL_INDEX_FILE_PATH) as all_index_file:
        for _ in range(10):
            assert next(indices_generator) == int(next(all_index_file).strip())


def test_load_data_returns_numpy_array():
    expected_array = np.array([1, 2, 3, 4])
    array_file_name = 'test.npy'
    array_file_path = os.path.join(data_paths.DATA_DIR_PATH, array_file_name)
    np.save(array_file_path, expected_array)
    try:
        actual_array = data_io.load_numpy_array_from_file(array_file_path)
        assert isinstance(actual_array, np.ndarray)
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_load_numpy_array_from_file_returns_correct_array():
    expected_array = np.array([1, 2, 3, 4])
    array_file_name = 'test.npy'
    array_file_path = os.path.join(data_paths.DATA_DIR_PATH, array_file_name)
    np.save(array_file_path, expected_array)
    try:
        actual_array = data_io.load_numpy_array_from_file(array_file_path)
        np.testing.assert_array_equal(expected_array, actual_array)
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_probe_points_first_ten_are_correct():
    first_n_indices_are_correct(data_io.probe_points(), number_of_points=10,
                                index=constants.PROBE_INDEX)


def test_probe_points_returns_a_generator():
    assert isinstance(data_io.probe_points(), types.GeneratorType)


def test_qual_points_first_ten_are_correct():
    first_n_indices_are_correct(data_io.qual_points(), number_of_points=10,
                                index=constants.QUAL_INDEX)


def test_qual_points_returns_a_generator():
    assert isinstance(data_io.qual_points(), types.GeneratorType)


def test_valid_points_first_ten_are_correct():
    first_n_indices_are_correct(data_io.valid_points(), number_of_points=10,
                                index=constants.VALID_INDEX)


def test_valid_points_returns_a_generator():
    assert isinstance(data_io.valid_points(), types.GeneratorType)


def test_write_submission_creates_file():
    ratings = (1, 4, 3, 2, 5)
    submission_file_name = 'test.dta'
    submission_file_path = os.path.join(data_paths.SUBMISSIONS_DIR_PATH,
                                        submission_file_name)
    assertion_message = '%s is for test use only' % submission_file_path
    assert not os.path.isfile(submission_file_path), assertion_message

    try:
        data_io.write_submission(ratings, submission_file_name)
        assertion_message = ('write_submission did not create %s' %
                             submission_file_path)
        assert os.path.isfile(submission_file_path), assertion_message
    finally:
        try:
            os.remove(submission_file_path)
        except FileNotFoundError:
            pass


def test_write_submission_writes_correct_ratings():
    ratings = (1, 4.0, 3.1, 2.01, 5.001)
    submission_file_name = 'test.dta'
    submission_file_path = os.path.join(data_paths.SUBMISSIONS_DIR_PATH,
                                        submission_file_name)
    assertion_message = '%s is for test use only' % submission_file_path
    assert not os.path.isfile(submission_file_path), assertion_message

    try:
        data_io.write_submission(ratings, submission_file_name)
        with open(submission_file_path, 'r') as submission_file:
            for rating in ratings:
                assert float(next(submission_file).strip()) == float(rating)
    finally:
        try:
            os.remove(submission_file_path)
        except FileNotFoundError:
            pass
