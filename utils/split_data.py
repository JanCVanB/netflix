"""Split all.dta into base.dta, hidden.dta, and probe.dta using indices in all.idx

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import os
from utils import constants


def run():
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    new_paths = [constants.BASE_DATA_FILE_PATH, constants.HIDDEN_DATA_FILE_PATH, constants.PROBE_DATA_FILE_PATH]
    for i, new_path in enumerate(new_paths):
        new_paths[i] = os.path.join(root_dir, new_path)
    for new_path in new_paths:
        open(new_path, 'a').close()


if __name__ == '__main__':
    run()
