import numpy as np
import os
import random
try:
    from unittest import mock
except ImportError:  # Python 2
    import mock

from utils import constants, data_paths, data_stats


MockThatTracksCallsWithoutRunning = mock.Mock


def make_simple_test_set():
    test_data_set = np.array([[0, 1, 0, 5],
                              [0, 0, 0, 3],
                              [1, 1, 0, 4],
                              [2, 1, 0, 2],
                              [2, 0, 0, 5]],
                             dtype=np.int32)
    return test_data_set


def make_simple_stats():
    stats = data_stats.DataStats()
    stats.load_data_set(data_set=make_simple_test_set())
    stats.compute_stats()
    return stats


def test_init_movie_and_user_arrays_creates_numpy_arrays():
    stats = data_stats.DataStats()
    test_set = make_simple_test_set()
    stats.load_data_set(data_set=test_set)
    stats.init_movie_and_user_arrays()
    assert isinstance(stats.movie_averages, np.ndarray)
    assert isinstance(stats.movie_rating_count, np.ndarray)
    assert isinstance(stats.movie_rating_count, np.ndarray)
    assert isinstance(stats.movie_rating_sum, np.ndarray)
    assert isinstance(stats.user_rating_count, np.ndarray)
    assert isinstance(stats.user_offsets, np.ndarray)


def test_compute_stats_calls_appropriate_functions():
    stats = data_stats.DataStats()
    test_set = make_simple_test_set()
    stats.load_data_set(data_set=test_set)
    stats.init_movie_and_user_arrays = MockThatTracksCallsWithoutRunning()
    stats.compute_movie_stats = MockThatTracksCallsWithoutRunning()
    stats.compute_user_stats = MockThatTracksCallsWithoutRunning()
    stats.compute_stats()
    assert stats.init_movie_and_user_arrays.call_count == 1
    assert stats.compute_movie_stats.call_count == 1
    assert stats.compute_user_stats.call_count == 1


def test_compute_global_average_rating_returns_correct_value():
    test_set = make_simple_test_set()
    expected_average = np.mean(test_set[:, constants.RATING_INDEX])
    test_average = data_stats.compute_global_average_rating(data_set=test_set)
    np.testing.assert_almost_equal(test_average, expected_average)


@mock.patch('utils.data_stats.compute_simple_indexed_sum_and_count')
@mock.patch('utils.data_stats.compute_blended_indexed_averages')
def test_compute_movie_stats_calls_expected_functions(mock_compute_blended_indexed_average,
                                                      mock_compute_simple_indexed_sum_and_count):
    mock_compute_simple_indexed_sum_and_count.return_value = (np.array([0]),
                                                              np.array([0]))
    mock_compute_blended_indexed_average.return_value = np.array([0])
    stats = data_stats.DataStats()
    stats.load_data_set(make_simple_test_set())
    stats.compute_movie_stats()
    assert mock_compute_simple_indexed_sum_and_count.call_count == 1
    assert mock_compute_blended_indexed_average.call_count == 1


def test_compute_stats_calculates_expected_stats():
    # Using make_simple_test_set, calculate the expected movie_stats by hand
    data_set = make_simple_test_set()
    # Test Set:
    #     np.array([[0, 1, 0, 5],
    #               [0, 0, 0, 3],
    #               [1, 1, 0, 4],
    #               [2, 1, 0, 2],
    #               [2, 0, 0, 5]],
    #              dtype=np.int32)
    expected_num_users = 3
    expected_num_movies = 2
    expected_global_average = 3.8  # (5 + 3 + 4 + 2 + 5) / 5
    # averages (blended): [((3.8 * k + 8) / (k + 2), (3.8 * k + 11) / (k + 3))]
    expected_movie_averages = np.array([3.814814814815, 3.78571428571429],
                                       dtype=np.float32)
    expected_movie_rating_count = np.array([2, 3], dtype=np.int32)
    expected_movie_rating_sum = np.array([8, 11], dtype=np.int32)
    # user 0 offsets: 5 - 3.523809523
    user_0_offsets = np.array([5 - expected_movie_averages[1],
                               3 - expected_movie_averages[0]],
                              dtype=np.float32)
    user_1_offsets = np.array([4 - expected_movie_averages[1]],
                              dtype=np.float32)
    user_2_offsets = np.array([2 - expected_movie_averages[1],
                               5 - expected_movie_averages[0]],
                              dtype=np.float)
    expected_user_rating_count = np.array([2, 1, 2], dtype=np.int32)
    expected_user_offset_sum = np.array([np.sum(user_0_offsets),
                                         np.sum(user_1_offsets),
                                         np.sum(user_2_offsets)],
                                        dtype=np.float32)
    global_user_offset = (np.sum(expected_user_offset_sum) /
                          np.sum(expected_user_rating_count))
    k = constants.BLENDING_RATIO
    expected_user_offsets = np.array(
        [(global_user_offset * k + expected_user_offset_sum[i]) /
         (k + expected_user_rating_count[i])
         for i in range(0, 3)], dtype=np.float32)
    stats = data_stats.DataStats()
    stats.load_data_set(data_set)
    stats.compute_stats()
    assert stats.num_users == expected_num_users
    assert stats.num_movies == expected_num_movies
    assert stats.global_average == expected_global_average
    np.testing.assert_array_almost_equal(stats.movie_averages,
                                         expected_movie_averages)
    np.testing.assert_array_almost_equal(stats.movie_rating_count,
                                         expected_movie_rating_count)
    np.testing.assert_array_almost_equal(stats.movie_rating_sum,
                                         expected_movie_rating_sum)
    np.testing.assert_array_almost_equal(stats.user_offsets,
                                         expected_user_offsets)
    np.testing.assert_array_almost_equal(stats.user_rating_count,
                                         expected_user_rating_count)
    np.testing.assert_array_almost_equal(stats.user_offsets_sum,
                                         expected_user_offset_sum)


@mock.patch('utils.data_stats.compute_simple_indexed_sum_and_count')
@mock.patch('utils.data_stats.compute_blended_indexed_averages')
@mock.patch('utils.data_stats.compute_offsets')
def test_compute_user_stats_calls_appropriate_functions(mock_compute_offsets,
                                                        mock_compute_blended_indexed_averages,
                                                        mock_compute_simple_indexed_sum_and_count):
    mock_compute_simple_indexed_sum_and_count.return_value = (np.array([1]),
                                                              np.array([1]))
    mock_compute_blended_indexed_averages.return_value = np.array([0])
    stats = data_stats.DataStats()
    stats.load_data_set(make_simple_test_set())
    stats.compute_user_stats()
    assert mock_compute_simple_indexed_sum_and_count.call_count == 1
    assert mock_compute_blended_indexed_averages.call_count == 1
    assert mock_compute_offsets.call_count == 1


def test_compute_simple_indexed_sum_and_count_returns_correct_values():
    test_data = np.array([16.5, 33, 4, -5, 12, -7.5], dtype=np.float32)
    index_array = np.array([0, 1, 2,  3,  4,  0], dtype=np.int32)
    test_indexed_sum, test_indexed_count = data_stats.compute_simple_indexed_sum_and_count(
        data_indices=index_array,
        data_values=test_data
    )
    expected_indexed_sum = np.array([9, 33, 4, -5, 12], dtype=np.float32)
    expected_indexed_count = np.array([2, 1, 1, 1, 1], dtype=np.int32)
    np.testing.assert_almost_equal(test_indexed_sum,
                                   expected_indexed_sum)
    np.testing.assert_array_equal(test_indexed_count,
                                  expected_indexed_count)


def test_compute_offsets_returns_correct_array():
    test_data = make_simple_test_set()
    # Test Set:
    #     np.array([[0, 1, 0, 5],
    #               [0, 0, 0, 3],
    #               [1, 1, 0, 4],
    #               [2, 1, 0, 2],
    #               [2, 0, 0, 5]],
    #              dtype=np.int32)
    test_averages = np.array([4, 11 / 3], dtype=np.float32)
    test_offsets = data_stats.compute_offsets(
        data_values=test_data[:, constants.RATING_INDEX],
        data_indices=test_data[:, constants.MOVIE_INDEX],
        averages=test_averages
    )
    expected_offsets = np.array([5 - 11 / 3, -1, 4 - 11 / 3, 2 - 11 / 3, 1],
                                dtype=np.float32)
    np.testing.assert_almost_equal(test_offsets, expected_offsets, decimal=6)


def test_compute_simple_indexed_sum_and_count_expects_arrays_of_same_size():
    test_arr1 = np.ones(shape=(10,))
    test_arr2 = np.ones(shape=(11,))
    try:
        _, _ = data_stats.compute_simple_indexed_sum_and_count(
            test_arr1,
            test_arr2
        )
    except ValueError:
        pass
    else:
        raise Exception(
            'Function should have raised an exception on mismatched array size.'
        )


def test_compute_blended_indexed_averages_returns_correct_values():
    test_data = np.array([16.5, 33, 4, -5, 12, -7.5], dtype=np.float32)
    # index_array = np.array([0, 1, 2,  3,  4,  0], dtype=np.int32)
    #  simple sum:   [9, 33,  4, -5, 12]
    #  simple count: [2,  1,  1,  1,  1]
    #  global_average = (16.5 - 7.5 + 33 + 4 - 5 + 12) / 6 = 8.8666666
    test_global_average = np.mean(test_data)
    # blending formula:
    #     blended_avg[movie] = (mean(all_ratings) * k + sum(movie_ratings)) /
    #                           (k + count(movie_ratings))
    # Assuming k = BLENDING_RATIO = 25
    # blended_avg[0] = (8.866666 * 25 + 9)/(25 + 2) = 8.54320981
    # blended_avg[1] =  etc...
    test_indexed_sum = np.array([9, 33, 4, -5, 12], dtype=np.float32)
    test_indexed_count = np.array([2, 1, 1, 1, 1], dtype=np.int32)
    k = constants.BLENDING_RATIO
    expected_averages = np.array(
        [((test_global_average * k + test_indexed_sum[i]) /
          (k + test_indexed_count[i]))
         for i in range(len(test_indexed_sum))],
        dtype=np.float32)
    returned_blended_averages = data_stats.compute_blended_indexed_averages(
        simple_sum=test_indexed_sum,
        simple_count=test_indexed_count,
        global_average=test_global_average
    )
    np.testing.assert_array_almost_equal(returned_blended_averages,
                                         expected_averages, decimal=6)
    

def test_get_baseline_returns_expected_baseline():
    stats = data_stats.DataStats()
    stats.user_offsets = np.array([1, 3, 4], dtype=np.float32)
    stats.movie_averages = np.array([2.3, 1.2, 0.4], dtype=np.float32)
    user_id = 1
    movie_id = 1
    # According to Simon Funk's article:
    #     baseline = avg_movie_rating[movie] + avg_user_offset[user]
    actual_baseline = stats.get_baseline(user=user_id, movie=movie_id)
    expected_baseline = 4.2
    np.testing.assert_almost_equal(actual_baseline, expected_baseline,
                                   decimal=5)


def test_data_stats_init_can_create_blank_instance():
    stats = data_stats.DataStats()
    assert isinstance(stats, data_stats.DataStats)


def test_load_data_set_loads_data_successfully():
    stats = data_stats.DataStats()
    test_data_set = make_simple_test_set()
    stats.load_data_set(data_set=test_data_set)
    np.testing.assert_array_equal(stats.data_set, test_data_set)


def test_load_data_stats_from_file_returns_data_stats():
    file_name = 'test_stats.p'
    file_path = os.path.join(data_paths.DATA_DIR_PATH, file_name)
    test_stats = data_stats.DataStats()
    unique_identifier = random.random()
    test_stats.global_average = unique_identifier
    test_stats.write_stats_to_file(file_path)
    new_stats = data_stats.load_stats_from_file(file_path=file_path)
    assert isinstance(new_stats, data_stats.DataStats)
    assert new_stats.global_average == test_stats.global_average


def test_saved_stats_file_does_not_contain_the_raw_data_set():
    file_name = 'test_stats.p'
    file_path = os.path.join(data_paths.DATA_DIR_PATH, file_name)
    test_stats = data_stats.DataStats()
    test_stats.data_set = np.array([12, 11, 222], dtype=np.int32)
    test_stats.write_stats_to_file(file_path=file_path)
    stats = data_stats.load_stats_from_file(file_path)
    try:
        assert not stats.data_set
    except ValueError:
        raise Exception('Expected empty data_set in saved stats file.')
    

def test_save_stats_to_file_creates_a_file():
    file_name = 'test_stats.p'
    file_path = os.path.join(data_paths.DATA_DIR_PATH, file_name)
    test_stats = data_stats.DataStats()
    test_stats.data_set = np.array([12, 11, 222], dtype=np.int32)
    test_stats.global_average = 2.344
    test_stats.write_stats_to_file(file_path=file_path)
    assert os.path.isfile(file_path), 'Could not locate the file.'
    os.remove(file_path)
