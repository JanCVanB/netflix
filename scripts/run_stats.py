from __future__ import print_function
from os.path import abspath, dirname, join, isfile
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_stats import DataStats
from utils.data_paths import DATA_DIR_PATH
from utils.data_io import load_numpy_array_from_file


def compute_stats_for_data_set_name(name):
    data_set_path = join(DATA_DIR_PATH, name + '.npy')
    stats_path = join(DATA_DIR_PATH, name + '_stats.p')
    if isfile(stats_path):
        raise Exception('Stats file already exists! Please delete stats file ' +
                        'to re-compute stats for set: \'{}\''.format(name))
    stats = DataStats()
    print('Loading data set from {}...'.format(data_set_path))
    data_set = load_numpy_array_from_file(file_name=data_set_path)
    stats.load_data_set(data_set)
    print('Computing stats ...')
    stats.compute_stats()
    print('Saving stats to file: {}'.format(stats_path))
    stats.write_stats_to_file(file_path=stats_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\n\tUSAGE:\n')
        print('\tpython3 scripts/run_stats.py DATASET_NAME')
        print('\n\t\tDATASET_NAME is the prefix of any of the .npy data files '
              'in /netflix/data.')
        print('\n\tEx: python3 scripts/run_stats.py valid\n')
    else:
        compute_stats_for_data_set_name(name=sys.argv[1])
