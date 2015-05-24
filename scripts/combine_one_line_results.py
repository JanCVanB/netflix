"""Combine multiple one-epoch results files into one all-epoch result file

Take multiple file paths as command line arguments
Use last file path argument as the output file path
Warn user if the input files are not exactly every epoch file required
Warn user if the output file already exists

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import os
import sys


NUM_PATH_UNDERSCORES_BEFORE_EPOCH_VALUE = 2


input_file_paths = sys.argv[1:-1]
output_file_path = sys.argv[-1]


rmse_values_by_epoch = {}
for rmse_file_path in input_file_paths:
    with open(rmse_file_path, 'r') as rmse_file:
        rmse_value = float(rmse_file.read().strip())
    rmse_file_path_parts = rmse_file_path.split('_')
    epochs_part = rmse_file_path_parts[NUM_PATH_UNDERSCORES_BEFORE_EPOCH_VALUE]
    epoch = int(epochs_part[:epochs_part.index('epochs')])
    rmse_values_by_epoch[epoch] = rmse_value


max_epoch = max(rmse_values_by_epoch.keys())
actual_epochs = sorted(rmse_values_by_epoch.keys())
expected_epochs = range(1, max_epoch + 1)
diff = [epoch for epoch in expected_epochs if epoch not in actual_epochs]
if len(diff) > 1:
    raise Exception('Missing epochs: {}'
                    .format(', '.join([str(x) for x in diff])))
if len(input_file_paths) > len(expected_epochs):
    raise Exception('Too many epochs - any extra files?\n'
                    '{}\n'
                    'Too many epochs - any extra files?'
                    .format('\n'.join(input_file_paths)))


assert not os.path.isfile(output_file_path), 'Output File Already Exists'


with open(output_file_path, 'w+') as output_file:
    for epoch, rmse_value in sorted(rmse_values_by_epoch.items()):
        output_file.write('{}\n'.format(rmse_value))
