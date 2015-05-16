from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from scripts.run_model import run

NUMBER_OF_EPOCHS = 50
NUMBER_OF_FEATURES = 25
TRAIN_SET_NAME = 'base'
TEST_SET_NAME = 'valid'

model = SVD(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
model.run_c = True
try:
    run(model, TRAIN_SET_NAME, TEST_SET_NAME,
        epochs=NUMBER_OF_EPOCHS, features=NUMBER_OF_FEATURES)
except Exception as exception:
    import pdb
    pdb.set_trace()
