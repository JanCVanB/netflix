from os.path import abspath, dirname, join, isfile
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_stats import DataStats
from utils.data_paths import DATA_DIR_PATH
from utils.data_io import load_numpy_array_from_file


def compute_stats_for_data_set_name(name, use_intermediate=False, fraction=0):
    mu_data_set_path = join(DATA_DIR_PATH, name + '.npy')
    um_data_set_path = join(DATA_DIR_PATH, name + '_um.npy')
    stats_path = join(DATA_DIR_PATH, name + '_stats.p')
    intermediate_stats_path = join(DATA_DIR_PATH, name + '_intermediate_stats.p')
    #if isfile(stats_path):
    #    raise Exception('Stats file already exists! Please delete stats file ' +
    #                    'to re-compute stats for set: \'{}\''.format(name))
    stats = DataStats()
    print('Loading um_data set from {}...'.format(um_data_set_path))
    print('Loading mu_data set from {}...'.format(mu_data_set_path))
    um_data_set = load_numpy_array_from_file(file_name=um_data_set_path)
    mu_data_set = load_numpy_array_from_file(file_name=mu_data_set_path)
    stats.load_data_set(data_set=um_data_set, mu_data_set=mu_data_set)
    if use_intermediate:
        if isfile(intermediate_stats_path):
            print('Loading stats ...')
            if(fraction<=1):
                stats.load_intermediate_stats(intermediate_stats_path)
            else:
                stats.load_intermediate_stats(stats_path)
        else:
            print('Intermediate file doesnt exist, making..')
            stats.compute_intermediate_stats()
            stats.write_pickle_to_file(intermediate_stats_path, keep_data=True)
    print('Computing stats ...')
    stats.compute_stats(use_intermediate=use_intermediate, fraction=fraction, num_pieces=4)
    print('Saving stats to file: {}'.format(stats_path))
    stats.write_stats_to_file(file_path=stats_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('\n\tUSAGE:\n')
        print('\tpython3 scripts/run_stats.py DATASET_NAME (TRUE/FALSE)')
        print('\n\t\tDATASET_NAME is the prefix of any of the .npy data files in /netflix/data.')
        print('\n\t\tT/F is whether to use intermediates stats')
        print('\n\tEx: python3 scripts/run_stats.py valid\n')
    else:
        compute_stats_for_data_set_name(name=sys.argv[1], use_intermediate=sys.argv[2], fraction=sys.argv[3])
