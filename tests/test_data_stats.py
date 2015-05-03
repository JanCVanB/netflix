import numpy as np


def make_simple_test_set():
    test_data_set = np.array([
        [0, 1, 0, 5],
        [0, 0, 0, 3],
        [1, 1, 0, 4],
        [2, 1, 0, 2],
        [2, 0, 0, 5]], dtype=np.int32)
    return test_data_set


def get_test_set_stats():
    test_set = make_simple_test_set()
    test_movie_averages = np.array([4, 11/3], dtype=np.float32)  # (3+5)/2, (5+4+2)/3])
    test_movie_rating_count = np.array([2, 3], dtype=np.int32)
    test_user_offsets = np.array([((5-11/3)+(3-4))/2,  # User 0:  (5-11/3  +  3-4)/2
                                  (4-11/3),            # User 1: 4-11/3
                                  (5-4 + 2-11/3)/2],   # User 2: (5-4  +  2-11/3)/2
                                 dtype=np.float32)
    test_user_rating_count = np.array([2, 1, 2], dtype=np.int32)
    return test_movie_averages, test_movie_rating_count, \
        test_user_offsets, test_user_rating_count


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


def test_compute_movie_stats_creates_numpy_array():
    from utils.data_stats import DataStats
    stats = DataStats()
    test_set = make_simple_test_set()
    expected_movie_averages, expected_movie_rating_count, _, _ = get_test_set_stats()
    stats.load_data_set(test_set)
    stats.compute_movie_stats()
    np.testing.assert_almost_equal(stats.movie_averages, expected_movie_averages)
    np.testing.assert_almost_equal(stats.movie_rating_count,
                                  expected_movie_rating_count)


def test_computer_user_stats_creates_numpy_array():
    from utils.data_stats import DataStats
    stats = DataStats()
    test_set = make_simple_test_set()
    _, _, expected_user_offsets, expected_user_rating_count = get_test_set_stats()
    stats.load_data_set(test_set)
    stats.compute_movie_stats()
    stats.compute_user_stats()
    np.testing.assert_almost_equal(stats.user_offsets, expected_user_offsets)
    np.testing.assert_almost_equal(stats.user_rating_count,
                                   expected_user_rating_count)


def test_load_data_stats_from_file_returns_data_stats():
    from utils.data_stats import load_stats_from_file
    from utils.data_stats import DataStats
    from utils.data_paths import DATA_DIR_PATH
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    stats = load_stats_from_file(file_path=file_path)
    assert isinstance(stats, DataStats)


def create_test_stats_file(file_path):
    from utils.data_stats import DataStats
    from utils.constants import MOVIE_INDEX, USER_INDEX
    stats = DataStats()
    test_set = make_simple_test_set()
    movie_averages, movie_rating_count, user_offsets, user_rating_count = get_test_set_stats()
    stats.load_data_set(data_set=test_set)
    stats.movie_averages = movie_averages
    stats.movie_rating_count = movie_rating_count
    stats.user_offsets = user_offsets
    stats.user_rating_count = user_rating_count
    stats.average_of_all_movies = np.mean(movie_averages)
    stats.num_users = np.amax(test_set[:, USER_INDEX]) + 1
    stats.num_movies = np.amax(test_set[:, MOVIE_INDEX]) + 1
    stats.data_set = []
    stats.write_stats_to_file(file_path=file_path)


def test_saved_stats_file_does_not_contain_the_raw_data_set():
    from utils.data_paths import DATA_DIR_PATH
    from utils.data_stats import load_stats_from_file
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    create_test_stats_file(file_path)
    stats = load_stats_from_file(file_path)
    try:
        assert not stats.data_set
    except ValueError:
        raise Exception('Expected empty data_set in saved stats file.')
    
    


def test_save_stats_to_file_creates_a_file():
    from utils.data_paths import DATA_DIR_PATH
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    create_test_stats_file(file_path)
    assert os.path.isfile(file_path), 'Could not locate the file.'
 

def test_load_stats_from_file_loads_correct_data():
    from utils.data_stats import DataStats
    from utils.data_stats import load_stats_from_file
    from utils.constants import MOVIE_INDEX, USER_INDEX
    from utils.data_paths import DATA_DIR_PATH
    import os
    file_name = 'test_stats.p'
    file_path = os.path.join(DATA_DIR_PATH, file_name)
    stats = load_stats_from_file(file_path=file_path)
    expected_movie_averages, expected_movie_rating_count, \
    expected_user_offsets, expected_user_rating_count = get_test_set_stats()
#    stats.movie_averages = movie_averages
#    stats.movie_rating_count = movie_rating_count
#    stats.user_offset = user_offsets
#    stats.user_rating_count = user_rating_count
#    stats.average_of_all_movies = np.mean(movie_averages)
#    stats.num_users = np.amax(test_set[:, USER_INDEX]) + 1
#    stats.num_movies = np.amax(test_set[:, MOVIE_INDEX]) + 1
    stats = load_stats_from_file(file_path=file_path)
    print(dir(stats))
    np.testing.assert_array_equal(stats.movie_averages,
                                  expected_movie_averages)
    np.testing.assert_array_equal(stats.movie_rating_count,
                                  expected_movie_rating_count)
    np.testing.assert_array_equal(stats.user_offsets,
                                  expected_user_offsets)
    np.testing.assert_array_equal(stats.user_rating_count,
                                  expected_user_rating_count)

