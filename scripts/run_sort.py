from __future__ import print_function
from os.path import abspath, dirname, join, isfile
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_paths import DATA_DIR_PATH
from utils.data_io import load_numpy_array_from_file
from utils.data_splitting import write_numpy_array_to_file
import numpy as np


def compute_sort_for_data_set(name, no_time=False):
    data_set_path = join(DATA_DIR_PATH, name + '.npy')
    time_string = '_notime' if no_time else ''
    sort_path = join(DATA_DIR_PATH, name + '_um{}.npy'.format(time_string))
    if isfile(sort_path):
        raise Exception('Sorted file already exists! Please delete sort file ' +
                        'to re-compute sort for set: \'{}\''.format(name))
    print('Loading data set from {}...'.format(data_set_path))
    data_set = load_numpy_array_from_file(file_name=data_set_path)
    print('Computing sort...')
    if no_time:
        print('(Excluding time from final numpy)')
        keep_columns = (0, 1, 3)
    else:
        keep_columns = (0, 1, 2, 3)
    sorted_set = np.sort(data_set.view('i4,i4,i4,i4'),
                         order=['f0', 'f1'],
                         kind='mergesort',
                         axis=0).view(np.int32)[:, keep_columns]
    print('Got sort: ')
    print(sorted_set)
    print('Saving sort to file: {}'.format(sort_path))
    write_numpy_array_to_file(file_path=sort_path, array=sorted_set)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\n\tUSAGE:\n')
        print('\tpython3 scripts/run_sort.py DATASET_NAME [notime]')
        print('\n\t\tDATASET_NAME is the prefix of any of the .npy data files '
              'in /netflix/data.')
        print('\n\tEx: python3 scripts/run_sort.py valid\n')
    else:
        compute_sort_for_data_set(name=sys.argv[1],
                                  no_time=('notime' in sys.argv))
