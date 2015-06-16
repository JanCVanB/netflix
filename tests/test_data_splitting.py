import numpy as np
import os

from utils import data_paths, data_splitting


def test_write_numpy_array_to_file_returns_none():
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(data_paths.DATA_DIR_PATH, array_file_name)
    try:
        return_value = data_splitting.write_numpy_array_to_file(input_array,
                                                                array_file_path)
        assert return_value is None
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_write_numpy_array_to_file_creates_file():
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(data_paths.DATA_DIR_PATH, array_file_name)
    assert not os.path.isfile(array_file_path), ('{} is for test use only'
                                                 .format(array_file_path))
    try:
        data_splitting.write_numpy_array_to_file(input_array, array_file_path)
        assertion_message = 'File not created by write_numpy_array_to_file'
        assert os.path.isfile(array_file_path), assertion_message
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_write_numpy_array_to_file_creates_expected_file():
    input_array = np.ones((4, 3))
    expected_array = np.copy(input_array)
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(data_paths.DATA_DIR_PATH, array_file_name)
    assert not os.path.isfile(array_file_path), ('{} is for test use only'
                                                 .format(array_file_path))
    try:
        data_splitting.write_numpy_array_to_file(input_array, array_file_path)
        actual_array = np.load(array_file_path)
        np.testing.assert_array_equal(actual_array, expected_array)
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_create_numpy_array_from_generator_returns_numpy_array():
    overestimated_shape = (10, 1)

    def input_generator():
        yield 0
    return_value = data_splitting.create_numpy_array_from_generator(
        generator=input_generator,
        overestimated_shape=overestimated_shape
    )
    assert isinstance(return_value, np.ndarray)
    assert return_value.dtype == np.int32


def test_create_numpy_array_from_generator_returns_expected_array():
    expected_array = np.random.randint(0, 999, (5, 4)).astype(np.int32)
    overestimated_shape = (10, 4)

    def input_generator():
        for thing in expected_array:
            yield thing
    actual_array = data_splitting.create_numpy_array_from_generator(
        generator=input_generator,
        overestimated_shape=overestimated_shape
    )
    np.testing.assert_array_equal(actual_array, expected_array)
