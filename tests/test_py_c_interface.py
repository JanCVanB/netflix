import numpy as np


def test_py_c_access_test_can_read_numpy_array_in_memory():
    import ctypes
    import os
    from utils.data_paths import LIBRARY_DIR_PATH
    test_array = np.array(range(0, 100), dtype=np.int32)
    print(test_array)
    library_file_name = 'test_py_c_interface.so'
    library_file_path = os.path.join(LIBRARY_DIR_PATH, library_file_name)
    test_lib = ctypes.cdll.LoadLibrary(library_file_path)
    returned_value = test_lib.py_c_access_test(ctypes.c_void_p(test_array.ctypes.data),
                                               ctypes.c_int(100))
    assert returned_value == 0, 'Py/C was unable to read the numpy array in memory.'