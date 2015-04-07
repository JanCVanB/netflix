def test_data_path_directories():
    import os
    from utils import data_paths
    dir_paths = [value for (key, value) in data_paths.__dict__.items() if key.endswith('DIR_PATH')]
    file_paths = [value for (key, value) in data_paths.__dict__.items() if key.endswith('FILE_PATH')]
    for dir_path in dir_paths:
        assert os.path.isdir(dir_path), '{} is not a directory'.format(dir_path)
    for file_path in file_paths:
        assert os.path.isdir(os.path.dirname(file_path)), '{} is not a file'.format(file_path)
