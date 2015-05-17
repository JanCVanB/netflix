from algorithms.model import Model


def test_model_can_create_instance_with_no_arguments():
    Model()


def test_model_load_creates_the_expected_instance():
    import numpy as np
    import os
    import pickle
    from random import random
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model.x = random()
    model.y = np.array([random()])
    load_file_name = 'test.p'
    load_file_path = os.path.join(MODELS_DIR_PATH, load_file_name)
    assert not os.path.isfile(load_file_path), ('{} is for test use only'
                                                .format(load_file_path))
    try:
        with open(load_file_path, 'wb+') as load_file:
            pickle.dump(model, load_file)
        loaded_model = Model.load(load_file_name)
        assert loaded_model.x == model.x
        np.testing.assert_array_equal(loaded_model.y, model.y)
    finally:
        try:
            os.remove(load_file_path)
        except FileNotFoundError:
            pass


def test_model_save_writes_the_expected_file():
    import numpy as np
    import os
    import pickle
    from random import random
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model.x = random()
    model.y = np.array([random()])
    save_file_name = 'test.p'
    save_file_path = os.path.join(MODELS_DIR_PATH, save_file_name)
    assert not os.path.isfile(save_file_path), ('{} is for test use only'
                                                .format(save_file_path))
    try:
        model.save(save_file_name)
        with open(save_file_path, 'rb') as save_file:
            saved_model = pickle.load(save_file)
        assert saved_model.x == model.x
        np.testing.assert_array_equal(saved_model.y, model.y)
    finally:
        try:
            os.remove(save_file_path)
        except FileNotFoundError:
            pass
