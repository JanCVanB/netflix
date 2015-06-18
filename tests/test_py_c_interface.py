import ctypes
import numpy as np
import os

from utils import data_paths


def test_py_c_access_test_can_read_numpy_array_in_memory():
    test_array = np.array(range(0, 100), dtype=np.int32)
    library_file_name = 'test_py_c_interface.so'
    library_file_path = os.path.join(data_paths.LIBRARY_DIR_PATH,
                                     library_file_name)
    test_lib = ctypes.cdll.LoadLibrary(library_file_path)
    returned_value = test_lib.py_c_access_test(
        ctypes.c_void_p(test_array.ctypes.data),
        ctypes.c_int32(100)
    )
    read_error_message = 'Py/C cannot read the numpy array in memory.'
    assert returned_value == 0, read_error_message


def test_py_c_can_write_to_numpy_array_in_memory():
    test_array = np.array(range(0, 100), dtype=np.int32)
    expected_array = np.copy(test_array)
    expected_array[5:50] = np.ones(shape=(45,), dtype=np.int32) * 3
    library_file_name = 'test_py_c_interface.so'
    library_file_path = os.path.join(data_paths.LIBRARY_DIR_PATH,
                                     library_file_name)
    test_lib = ctypes.cdll.LoadLibrary(library_file_path)
    returned_value = test_lib.py_c_write_test(test_array[5:50].ctypes.data,
                                              ctypes.c_int32(45))
    write_error_message = 'Py/C cannot write to the numpy array in memory.'
    assert returned_value == 0, write_error_message
    np.testing.assert_array_equal(test_array, expected_array)
