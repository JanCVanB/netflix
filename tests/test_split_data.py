def test_split_data_makes_files():
    import os
    from utils import constants
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    new_paths = [constants.BASE_DATA_FILE_PATH, constants.HIDDEN_DATA_FILE_PATH, constants.PROBE_DATA_FILE_PATH]
    for i, new_path in enumerate(new_paths):
        new_paths[i] = os.path.join(root_dir, new_path)
    for new_path in new_paths:
        os.remove(new_path) if os.path.isfile(new_path) else None
        new_file_exists_before_split = os.path.isfile(new_path)
        assert not new_file_exists_before_split, '{} cannot be removed'.format(new_path)
    from utils import split_data
    split_data.run()
    for new_path in new_paths:
        new_file_exists_after_split = os.path.isfile(new_path)
        assert new_file_exists_after_split, 'split_data.run() did not create {}'.format(new_path)
