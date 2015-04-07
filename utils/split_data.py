"""Split all.dta into base.dta, hidden.dta, and probe.dta using indices in all.idx

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
from utils.data_paths import BASE_DATA_FILE_PATH, HIDDEN_DATA_FILE_PATH, PROBE_DATA_FILE_PATH


def run():
    new_paths = [BASE_DATA_FILE_PATH, HIDDEN_DATA_FILE_PATH, PROBE_DATA_FILE_PATH]
    for new_path in new_paths:
        open(new_path, 'a').close()


if __name__ == '__main__':
    run()
