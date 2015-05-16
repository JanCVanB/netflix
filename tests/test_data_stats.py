import numpy as np
from unittest.mock import Mock, MagicMock

MockThatTracksCallsWithoutRunning = Mock
MockThatReturnsZeroNumpyTuple = Mock(return_value=(np.array([0]),
                                                   np.array([0])))
MockThatReturnsZeroNumpy = Mock(return_value=np.array([0]))


def make_simple_test_set():
    test_data_set = np.array([
        [0, 1, 0, 5],
        [0, 0, 0, 3],
        [1, 1, 0, 4],
        [2, 1, 0, 2],
        [2, 0, 0, 5]], dtype=np.int32)
    return test_data_set


def make_simple_stats():
    from utils.data_stats import DataStats
    stats = DataStats()
    stats.load_data_set(data_set=make_simple_test_set())
    stats.compute_movie_stats()
    stats.compute_user_stats()
    return stats


def test_init_movie_and_user_arrays_creates_numpy_arrays():
    from utils.data_stats import DataStats
    stats = DataStats()
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
    from utils.data_stats import DataStats
    stats = DataStats()
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
    from utils.data_stats import compute_global_average_rating
    from utils.constants import RATING_INDEX
    test_set = make_simple_test_set()
    expected_average = np.mean(test_set[:, RATING_INDEX])
    test_average = compute_global_average_rating(data_set=test_set)
    np.testing.assert_almost_equal(test_average, expected_average)


def test_compute_movie_stats_calls_appropriate_functions():
    import utils.data_stats
    from utils.data_stats import DataStats
    stats = DataStats()
    stats.load_data_set(make_simple_test_set())
    utils.data_stats.compute_simple_indexed_sum_and_count = MockThatReturnsZeroNumpyTuple
    utils.data_stats.compute_blended_indexed_averages = MockThatReturnsZeroNumpy
    stats.compute_movie_stats()
    assert utils.data_stats.compute_simple_indexed_sum_and_count.call_count == 1
    assert utils.data_stats.compute_blended_indexed_averages.call_count == 1
    import importlib
    importlib.reload(utils.data_stats)


def test_compute_user_stats_calls_appropriate_functions():
    import utils.data_stats
    from utils.data_stats import DataStats
    stats = DataStats()
    stats.load_data_set(make_simple_test_set())
    utils.data_stats.compute_simple_indexed_sum_and_count = MagicMock(return_value=(np.array([0]),
                                                                              np.array([0])))
    utils.data_stats.compute_blended_indexed_averages = MagicMock(return_value=np.array([0]))
    stats.compute_user_stats()
    assert utils.data_stats.compute_simple_indexed_sum_and_count.call_count == 1
    assert utils.data_stats.compute_blended_indexed_averages.call_count == 1
    import importlib
    importlib.reload(utils.data_stats)


def test_compute_simple_indexed_sum_and_count_returns_correct_values():
    import utils.data_stats
    import importlib as imp
    imp.reload(utils.data_stats)
    from utils.data_stats import compute_simple_indexed_sum_and_count
    test_data = np.array([16.5, 33, 4, -5, 12, -7.5], dtype=np.float32)
    index_array = np.array([0, 1, 2,  3,  4,  0], dtype=np.int32)
    test_indexed_sum, test_indexed_count = compute_simple_indexed_sum_and_count(
        data_indices=index_array,
        data_values=test_data
    )
    expected_indexed_sum = np.array([9, 33, 4, -5, 12], dtype=np.float32)
    expected_indexed_count = np.array([2, 1, 1, 1, 1], dtype=np.int32)
    np.testing.assert_almost_equal(test_indexed_sum,
                                   expected_indexed_sum)
    np.testing.assert_array_equal(test_indexed_count,
                                  expected_indexed_count)


def test_compute_simple_indexed_sum_and_count_performs_offset_calculations_when_given_data_averages():
    import utils.data_stats
    import importlib as imp
    imp.reload(utils.data_stats)
    from utils.data_stats import compute_simple_indexed_sum_and_count
    from utils.constants import USER_INDEX, MOVIE_INDEX, RATING_INDEX
    test_data = make_simple_test_set()
#    [0, 1, 0, 5],
#    [0, 0, 0, 3],
#    [1, 1, 0, 4],
#    [2, 1, 0, 2],
#    [2, 0, 0, 5]], dtype=np.int32)
    test_averages = np.array([4, 11/3], dtype=np.float32)
    test_indexed_sum, test_indexed_count = compute_simple_indexed_sum_and_count(
        data_indices=test_data[:, USER_INDEX],
        data_values=test_data[:, RATING_INDEX],
        averages_indices=test_data[:, MOVIE_INDEX],
        averages=test_averages
    )
    expected_indexed_sum = np.array([(5-(11/3)+3-4), 4-(11/3), 2-(11/3)+5-4], dtype=np.float32)
    expected_indexed_count = np.array([2, 1, 2], dtype=np.int32)
    np.testing.assert_almost_equal(test_indexed_sum,
                                   expected_indexed_sum, decimal=6)
    np.testing.assert_almost_equal(test_indexed_count,
                                   expected_indexed_count, decimal=6)


def test_compute_simple_indexed_sum_and_count_expects_arrays_of_same_size():
    from utils.data_stats import compute_simple_indexed_sum_and_count
    test_arr1 = np.ones(shape=(10,))
    test_arr2 = np.ones(shape=(11,))
    try:
        _, _ = compute_simple_indexed_sum_and_count(
            test_arr1,
            test_arr2
        )
    except ValueError:
        pass
    else:
        raise Exception('Function should have raised an exception on ' +
                        'mismatched array size.')


def test_compute_blended_indexed_averages_returns_correct_values():
    import utils.data_stats
    import importlib
    importlib.reload(utils.data_stats)
    from utils.data_stats import compute_blended_indexed_averages
    from utils.constants import BLENDING_RATIO
    test_data = np.array([16.5, 33, 4, -5, 12, -7.5], dtype=np.float32)
    # index_array = np.array([0, 1, 2,  3,  4,  0], dtype=np.int32)
    #  simple sum: [   9, 33, 4, -5, 12]
    #  simple count: [ 2,  1, 1,  1,  1]
    #  global_average = (16.5-7.5+33+4-5+12)/6 = 8.8666666
    test_global_average = np.mean(test_data)
    # blending formula: blended_avg[movie] = (mean(all_ratings) * K + sum(movie_ratings)) 
    #                                           / (K + count(movie_ratings))
    # Assuming BLENDING_RATION = K = 25
    # blended_avg[0] = (8.866666 * 25 + 9)/(25+2) = 8.54320981
    # blended_avg[1] =  etc...
    test_indexed_sum = np.array([9, 33, 4, -5, 12], dtype=np.float32)
    test_indexed_count = np.array([2, 1, 1, 1, 1], dtype=np.int32)
    expected_averages = np.array([((test_global_average * BLENDING_RATIO + test_indexed_sum[i]) / 
                                  (BLENDING_RATIO + test_indexed_count[i])) 
                                  for i in range(len(test_indexed_sum))],
                                 dtype=np.float32)
    returned_blended_averages = compute_blended_indexed_averages(simple_sum=test_indexed_sum,
                                                                 simple_count=test_indexed_count,
                                                                 global_average=test_global_average)
    np.testing.assert_array_almost_equal(returned_blended_averages, expected_averages, decimal=6)
    

def test_get_baseline_returns_expected_baseline():
    from utils.data_stats import DataStats
    stats = DataStats()
    stats.user_offsets = np.array([1, 3, 4], dtype=np.float32)
    stats.movie_averages = np.array([2.3, 1.2, 0.4], dtype=np.float32)
    user_id = 1
    movie_id = 1
    # Baseline (according to funny) = avg_movie_rating[movie] + avg_user_offset[user]
    test_baseline = stats.get_baseline(user=user_id, movie=movie_id)
    expected_baseline = 4.2
    np.testing.assert_almost_equal(test_baseline, expected_baseline, decimal=5)


def test_data_stats_init_can_create_blank_instance():
    from utils.data_stats import DataStats
    stats = DataStats()
    assert isinstance(stats, DataStats)


def test_load_data_set_loads_data_successfully():
    from utils.data_stats import DataStats
    stats = DataStats()
    test_data_set = make_simple_test_set()
    stats.load_data_set(data_set=test_data_set)
    np.testing.assert_array_equal(stats.data_set, test_data_set)


def test_load_data_stats_from_file_returns_data_stats():
    from utils.data_stats import load_stats_from_file
    from utils.data_stats import DataStats
    from utils.data_paths import DATA_DIR_PATH
    from random import random
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    test_stats = DataStats()
    unique_identifier = random()
    test_stats.global_average = unique_identifier
    test_stats.write_stats_to_file(file_path)
    new_stats = load_stats_from_file(file_path=file_path)
    assert isinstance(new_stats, DataStats)
    assert new_stats.global_average == test_stats.global_average


def test_saved_stats_file_does_not_contain_the_raw_data_set():
    from utils.data_paths import DATA_DIR_PATH
    from utils.data_stats import load_stats_from_file, DataStats
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    test_stats = DataStats()
    test_stats.data_set = np.array([12, 11, 222], dtype=np.int32)
    test_stats.write_stats_to_file(file_path=file_path)
    stats = load_stats_from_file(file_path)
    try:
        assert not stats.data_set
    except ValueError:
        raise Exception('Expected empty data_set in saved stats file.')
    

def test_save_stats_to_file_creates_a_file():
    from utils.data_paths import DATA_DIR_PATH
    from utils.data_stats import DataStats
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    test_stats = DataStats()
    test_stats.data_set = np.array([12, 11, 222], dtype=np.int32)
    test_stats.global_average = 2.344
    test_stats.write_stats_to_file(file_path=file_path)
    assert os.path.isfile(file_path), 'Could not locate the file.'
    os.remove(file_path)
