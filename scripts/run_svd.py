from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from scripts.run_model import run

NUMBER_OF_EPOCHS = 1
NUMBER_OF_FEATURES = 1
TRAIN_SET_NAME = 'valid'
TEST_SET_NAME = 'hidden'

model = SVD(num_features=NUMBER_OF_FEATURES)
model.run_c = True
run(model, TRAIN_SET_NAME, TEST_SET_NAME, epochs=NUMBER_OF_EPOCHS,
    features=NUMBER_OF_FEATURES)
