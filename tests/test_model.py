from algorithms.model import Model


def test_model_can_create_instance_with_no_arguments():
    Model()


def test_model_save_creates_file_at_expected_file_path():
    import os.path
    from utils.data_paths import MODELS_DIR_PATH
    model = Model()
    model_file_name = 'test.p'
    model_file_path = os.path.join(MODELS_DIR_PATH, model_file_name)
    assert not os.path.isfile(model_file_path), '{} is for test use only' % model_file_path

    try:
        model.save(model_file_path)
        assert os.path.isfile(model_file_path), 'save did not create %s' % model_file_path
    finally:
        try:
            os.remove(model_file_path)
        except FileNotFoundError:
            pass


def test_model_load_can_accept_file_path_argument():
    model = Model()
    file_path = 'test'
    model.load(file_path)


def test_model_load_returns_none():
    model = Model()
    file_path = 'test'
    return_value = model.load(file_path)
    assert return_value is None
