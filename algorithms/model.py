import os
import pickle

from utils.data_paths import MODELS_DIR_PATH


class Model:
    @staticmethod
    def load(file_name):
        file_path = os.path.join(MODELS_DIR_PATH, file_name)
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save(self, file_name):
        file_path = os.path.join(MODELS_DIR_PATH, file_name)
        with open(file_path, 'wb+') as file:
            pickle.dump(self, file)
