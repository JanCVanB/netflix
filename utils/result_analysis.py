__author__ = 'quinnosha'

"""Tools for viewing and analyzing prediction results

.. moduleauthor:: Quinn Osha
"""

import sys
from os.path import abspath, dirname, join
sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_paths import RESULTS_DIR_PATH

def find_lowest_rmse(rmse_file_name):
    rmse_file_path = join(RESULTS_DIR_PATH, rmse_file_name)
    read_format = 'r'
    rmse_values = []

    with open(rmse_file_path, read_format) as rmse_file:
        for line in rmse_file:
            rmse_value = line.strip();
            rmse_values.append(rmse_value)

    return min(rmse_values)


if __name__ == '__main__':
    rmse_file_name = 'svd_base_8epochs_100features_rmse_valid_Apr-26-12h-43m.txt'
    lowest_rmse = find_lowest_rmse(rmse_file_name)
    print(lowest_rmse)
