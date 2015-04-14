"""Reading and writing tools for interfacing with the large data files

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""


def _generate_points_from_index(correct_index):
    from utils.data_paths import ALL_DATA_FILE_PATH, ALL_INDEX_FILE_PATH
    indices_generator = indices(ALL_INDEX_FILE_PATH)
    for point in data_points(ALL_DATA_FILE_PATH):
        index = next(indices_generator)
        if index == correct_index:
            yield point


def base_points():
    from utils.constants import BASE_INDEX
    for point in _generate_points_from_index(BASE_INDEX):
        yield point


def data_points(data_file_path):
    with open(data_file_path) as data_file:
        for line in data_file:
            yield line.strip().split()


def hidden_points():
    from utils.constants import HIDDEN_INDEX
    for point in _generate_points_from_index(HIDDEN_INDEX):
        yield point


def indices(index_file_path):
    with open(index_file_path) as index_file:
        for line in index_file:
            yield int(line.strip())


def probe_points():
    from utils.constants import PROBE_INDEX
    for point in _generate_points_from_index(PROBE_INDEX):
        yield point


def qual_points():
    from utils.constants import QUAL_INDEX
    for point in _generate_points_from_index(QUAL_INDEX):
        yield point


def valid_points():
    from utils.constants import VALID_INDEX
    for point in _generate_points_from_index(VALID_INDEX):
        yield point


def write_submission(ratings, submission_file_name):
    import os
    from utils.data_paths import SUBMISSIONS_DIR_PATH
    submission_file_path = os.path.join(SUBMISSIONS_DIR_PATH, submission_file_name)
    with open(submission_file_path, 'w+') as submission_file:
        submission_file.writelines(['{:.3f}\n'.format(r) for r in ratings])
