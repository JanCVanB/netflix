from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from scripts.run_model import run_multi

NUMBER_OF_EPOCHS = 5
NUMBER_OF_FEATURES = 5
TRAIN_SET_NAME = 'base'
TEST_SET_NAME = 'valid'

model = SVD(num_features=NUMBER_OF_FEATURES)
model.run_c = True
run_multi(model, TRAIN_SET_NAME, TEST_SET_NAME,
          epochs=NUMBER_OF_EPOCHS, features=NUMBER_OF_FEATURES)
