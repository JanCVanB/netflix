from algorithms.model import Model


def test_model_can_create_instance_with_no_arguments():
    Model()


def test_model_save_creates_file_at_expected_file_path():
    import os.path
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model_file_name = 'test'
    model_file_path = os.path.join(MODELS_DIR_PATH, model_file_name)
    assertion_message = '%s is for test use only' % model_file_path
    assert not os.path.isfile(model_file_path), assertion_message

    try:
        model.save(model_file_path)
        assertion_message = 'save did not create %s' % model_file_path
        assert os.path.isfile(model_file_path), assertion_message
    finally:
        try:
            os.remove(model_file_path)
        except FileNotFoundError:
            pass


def test_model_load_can_accept_file_path_argument():
    from os.path import join
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model_file_name = 'test'
    model_file_path = join(MODELS_DIR_PATH, model_file_name)
    model.load(model_file_path)


def test_model_load_returns_none():
    from os.path import join
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model_file_name = 'test'
    model_file_path = join(MODELS_DIR_PATH, model_file_name)
    return_value = model.load(model_file_path)
    assert return_value is None
