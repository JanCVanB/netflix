import os


ROOT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')
DATA_MOVIE_USER_DIR_PATH = os.path.join(DATA_DIR_PATH, 'mu')
DATA_USER_MOVIE_DIR_PATH = os.path.join(DATA_DIR_PATH, 'um')
SUBMISSIONS_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'submissions')

ALL_DATA_FILE_PATH = os.path.join(DATA_MOVIE_USER_DIR_PATH, 'all.dta')
ALL_INDEX_FILE_PATH = os.path.join(DATA_MOVIE_USER_DIR_PATH, 'all.idx')
