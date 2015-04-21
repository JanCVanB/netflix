def test_write_numpy_array_to_file_returns_none():
    from utils.data_splitting import write_numpy_array_to_file
    from utils.data_paths import DATA_DIR_PATH
    import os
    import numpy as np
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(DATA_DIR_PATH, array_file_name)
    try:
        return_value = write_numpy_array_to_file(input_array, array_file_path)
        assert return_value is None
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_write_numpy_array_to_file_creates_file():
    from utils.data_splitting import write_numpy_array_to_file
    from utils.data_paths import DATA_DIR_PATH
    import os
    import numpy as np
    input_array = np.ones((4, 3))
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(DATA_DIR_PATH, array_file_name)
    assert not os.path.isfile(array_file_path), ('{} is for test use only'
                                                 .format(array_file_path))
    try:
        write_numpy_array_to_file(input_array, array_file_path)
        assertion_message = 'File not created by write_numpy_array_to_file'
        assert os.path.isfile(array_file_path), assertion_message
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass


def test_write_numpy_array_to_file_creates_expected_file():
    from utils.data_splitting import write_numpy_array_to_file
    from utils.data_paths import DATA_DIR_PATH
    import os
    import numpy as np
    input_array = np.ones((4, 3))
    expected_array = np.copy(input_array)
    array_file_name = 'array_test_file.npy'
    array_file_path = os.path.join(DATA_DIR_PATH, array_file_name)
    assert not os.path.isfile(array_file_path), ('{} is for test use only'
                                                 .format(array_file_path))
    try:
        write_numpy_array_to_file(input_array, array_file_path)
        actual_array = np.load(array_file_path)
    finally:
        try:
            os.remove(array_file_path)
        except FileNotFoundError:
            pass
    np.testing.assert_array_equal(actual_array, expected_array)


def test_create_numpy_array_from_generator_returns_numpy_array():
    import numpy as np
    from utils.data_splitting import create_numpy_array_from_generator

    def input_generator():
        yield None
    overestimated_shape = (10, 1)
    return_value = create_numpy_array_from_generator(generator=input_generator,
                                                     overestimated_shape=overestimated_shape)
    assert isinstance(return_value, np.ndarray)


def test_create_numpy_array_from_generator_returns_expected_array():
    import numpy as np
    from utils.data_splitting import create_numpy_array_from_generator
    from random import random
    expected_array = np.array([[random() for _ in range(4)] for __ in range(5)])
    overestimated_shape = (10, 4)

    def input_generator():
        for thing in expected_array:
            yield thing
    actual_array = create_numpy_array_from_generator(generator=input_generator,
                                                     overestimated_shape=overestimated_shape)
    np.testing.assert_array_equal(actual_array, expected_array)
