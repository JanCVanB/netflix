import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.data_io import load_numpy_array_from_file
from utils.data_paths import DATA_DIR_PATH
from utils.constants import RATING_INDEX


PROBE_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'probe.npy')


def main():
    probe_predictions = get_probe_predictions()
    probe = get_probe()
    weights = get_weights(probe_predictions, probe)
    blended_predictions = blend(probe_predictions, weights)
    blend_file_path = get_blend_file_path()
    write(blended_predictions, blend_file_path)


def get_probe_predictions():
    predictions = np.array([])
    prediction_file_paths = sys.argv[1:-1]
    for prediction_file_path in prediction_file_paths:
        with open(prediction_file_path, 'r') as prediction_file:
            prediction = np.transpose(np.array([prediction_file.read().split()],
                                               dtype=np.float32))
            if predictions.size == 0:
                predictions = prediction
            else:
                predictions = np.append(predictions, prediction, axis=1)
    return np.matrix(predictions)


def get_probe():
    probe_points = load_numpy_array_from_file(PROBE_DATA_FILE_PATH)
    probe_ratings = np.array(probe_points, dtype=np.int32)[:, RATING_INDEX]
    return probe_ratings


def get_weights(predictions, ratings, alpha=1.0):
    assert predictions.shape[0] > predictions.shape[1]
    number_of_models = predictions.shape[1]
    gamma = np.eye(number_of_models) * alpha ** 2
    ptp_plus_gamma = predictions.T * predictions + gamma
    weights = np.dot(np.linalg.inv(ptp_plus_gamma) * predictions.T, ratings)
    return np.ravel(weights)


def blend(predictions, weights):
    return np.ravel(weights * predictions.T)


def get_blend_file_path():
    return sys.argv[-1]


def write(predictions, file_path):
    assert not os.path.isfile(file_path), '{} already exists!'.format(file_path)
    with open(file_path, 'w+') as file:
        file.write('\n'.join([str(p) for p in predictions]) + '\n')


if __name__ == '__main__':
    main()
