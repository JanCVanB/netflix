from os.path import abspath, dirname, join


UTILS_DIR_PATH = abspath(dirname(__file__))
ROOT_DIR_PATH = abspath(dirname(UTILS_DIR_PATH))

DATA_DIR_PATH = join(ROOT_DIR_PATH, 'data')
DATA_MOVIE_USER_DIR_PATH = join(DATA_DIR_PATH, 'mu')
DATA_USER_MOVIE_DIR_PATH = join(DATA_DIR_PATH, 'um')
LIBRARY_DIR_PATH = join(ROOT_DIR_PATH, 'lib')
MODELS_DIR_PATH = join(ROOT_DIR_PATH, 'models')
RESULTS_DIR_PATH = join(ROOT_DIR_PATH, 'results')
SUBMISSIONS_DIR_PATH = join(ROOT_DIR_PATH, 'submissions')

ALL_DATA_FILE_PATH = join(DATA_MOVIE_USER_DIR_PATH, 'all.dta')
ALL_INDEX_FILE_PATH = join(DATA_MOVIE_USER_DIR_PATH, 'all.idx')
