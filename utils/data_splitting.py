import numpy as np


def write_numpy_array_to_file(array, file_path):
    np.save(file_path, array)


def create_numpy_array_from_generator(generator, overestimated_shape):
    array = np.zeros(shape=overestimated_shape)
    for i, x in enumerate(generator()):
        array[i, :] = x
    return array[:i + 1, :]

