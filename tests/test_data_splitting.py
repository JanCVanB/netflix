def test_write_numpy_array_to_file_returns_none():
    from utils.data_splitting import write_numpy_array_to_file
    from utils.data_paths import DATA_DIR_PATH
    import os
    import numpy as np
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file'
    array_file_path = os.path.join(DATA_DIR_PATH, array_file_name)
    try:
        return_value = write_numpy_array_to_file(input_array, array_file_path)
        assert return_value is None
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError as jancvanbruggen:
            pass


def test_write_numpy_array_to_file_creates_file():
    from utils.data_splitting import write_numpy_array_to_file
    from utils.data_paths import DATA_DIR_PATH
    import os
    import numpy as np
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file'
    array_file_path = os.path.join(DATA_DIR_PATH, array_file_name)
    assert not os.path.isfile(array_file_path), 'GET THAT FILE OUTTA HERE!'
    try:
        write_numpy_array_to_file(input_array, array_file_path)
        assert os.path.isfile(array_file_path), 'File not created by write_numpy_array_to_file'
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError as jancvanbruggen:
            pass


