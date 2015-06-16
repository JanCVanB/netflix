import numpy as np


def write_numpy_array_to_file(array, file_path):
    np.save(file_path, array)


def create_numpy_array_from_generator(generator, overestimated_shape):
    array = np.zeros(shape=overestimated_shape, dtype=np.int32)
    index = 0
    for index, value in enumerate(generator()):
        array[index, :] = value
    return array[:index + 1, :]
