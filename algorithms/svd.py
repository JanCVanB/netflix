import numpy as np


def run(input_array=None):
    if input_array is None:
        return run_all()
    return run_custom(input_array)


def run_all():
    return np.array([])


def run_custom(input_array):
    return np.array([])
