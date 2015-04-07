def test_data_points_returns_a_generator():
    from types import GeneratorType
    from utils.data_io import data_points
    from utils.data_paths import ALL_DATA_FILE_PATH

    assert isinstance(data_points(ALL_DATA_FILE_PATH), GeneratorType)


def test_data_points_first_three_correct():
    from utils.data_io import data_points
    from utils.data_paths import ALL_DATA_FILE_PATH

    points = data_points(ALL_DATA_FILE_PATH)

    with open(ALL_DATA_FILE_PATH) as all_data_file:
        for _ in range(3):
            assert next(points) == next(all_data_file).strip().split()


def test_write_submission_creates_file():
    import os
    from utils.data_io import write_submission
    from utils.data_paths import SUBMISSIONS_DIR_PATH
    ratings = (1, 4, 3, 2, 5)
    submission_file_name = 'test.dta'
    submission_file_path = os.path.join(SUBMISSIONS_DIR_PATH, submission_file_name)
    assert not os.path.isfile(submission_file_path), "{} is for test_data_io's use only".format(submission_file_path)

    try:
        write_submission(ratings, submission_file_name)
        assert os.path.isfile(submission_file_path), 'write_submission did not create {}'.format(submission_file_path)
    finally:
        os.remove(submission_file_path)


def test_write_submission_writes_correct_ratings():
    import os
    from utils.data_io import write_submission
    from utils.data_paths import SUBMISSIONS_DIR_PATH
    ratings = (1, 4, 3, 2, 5)
    submission_file_name = 'test.dta'
    submission_file_path = os.path.join(SUBMISSIONS_DIR_PATH, submission_file_name)
    assert not os.path.isfile(submission_file_path), "{} is for test_data_io's use only".format(submission_file_path)

    try:
        write_submission(ratings, submission_file_name)
        with open(submission_file_path, 'r') as submission_file:
            for rating in ratings:
                assert int(float(next(submission_file).strip())) == rating
    finally:
        os.remove(submission_file_path)
