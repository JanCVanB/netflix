from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from scripts.run_model import run

NUMBER_OF_EPOCHS = 20
NUMBER_OF_FEATURES = 50
TRAIN_SET_NAME = 'base'
TEST_SET_NAME = 'valid'

model = SVD(num_features=NUMBER_OF_FEATURES)
model.run_c = True
run(model, TRAIN_SET_NAME, TEST_SET_NAME,
    epochs=NUMBER_OF_EPOCHS, features=NUMBER_OF_FEATURES)
