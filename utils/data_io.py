"""Reading and writing tools for interfacing with the large data files

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""


def data_points(data_file_path):
    with open(data_file_path) as data_file:
        for line in data_file:
            yield line.strip().split()


def write_submission(ratings, submission_file_name):
    import os
    from utils.data_paths import SUBMISSIONS_DIR_PATH
    submission_file_path = os.path.join(SUBMISSIONS_DIR_PATH, submission_file_name)
    with open(submission_file_path, 'w+') as submission_file:
        submission_file.writelines(['{:.3f}\n'.format(r) for r in ratings])
