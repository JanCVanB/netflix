import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.data_paths import SUBMISSIONS_DIR_PATH


OUTPUT_FILE_PATH = os.path.join(SUBMISSIONS_DIR_PATH, 'simple_blend.dta')
PREDICTION_FILE_PATHS = [os.path.join(SUBMISSIONS_DIR_PATH, 'predictions1.dta'),
                         os.path.join(SUBMISSIONS_DIR_PATH, 'predictions2.dta')]
PREDICTION_COEFFICIENTS = [0.4,
                           0.6]


def main():
    predictions = get_predictions()
    write(predictions)


def get_predictions():
    predictions = np.array([])
    for i, prediction_file_path in enumerate(PREDICTION_FILE_PATHS):
        with open(prediction_file_path, 'r') as prediction_file:
            prediction = np.transpose(np.array([prediction_file.read().split()],
                                               dtype=np.float32))
            if predictions.size == 0:
                predictions = prediction
            else:
                predictions = np.append(predictions, prediction, axis=1)
    return np.matrix(predictions)


def write(predictions):
    coefficients = np.array(PREDICTION_COEFFICIENTS)
    with open(OUTPUT_FILE_PATH, 'w+') as output_file:
        for prediction_set in predictions:
            prediction = np.dot(np.ravel(prediction_set), coefficients)
            output_file.write('{}\n'.format(prediction))


if __name__ == '__main__':
    main()
