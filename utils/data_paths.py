import os


ROOT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

ALL_DATA_FILE_PATH = os.path.join(ROOT_DIR_PATH, 'data/mu/all.dta')
ALL_INDEX_FILE_PATH = os.path.join(ROOT_DIR_PATH, 'data/mu/all.idx')
BASE_DATA_FILE_PATH = os.path.join(ROOT_DIR_PATH, 'data/mu/base.dta')
HIDDEN_DATA_FILE_PATH = os.path.join(ROOT_DIR_PATH, 'data/mu/hidden.dta')
PROBE_DATA_FILE_PATH = os.path.join(ROOT_DIR_PATH, 'data/mu/probe.dta')
