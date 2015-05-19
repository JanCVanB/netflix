from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from algorithms.svd_euclidean import SVDEuclidean
from scripts.run_model import run

NUMBER_OF_EPOCHS = 10
NUMBER_OF_FEATURES = 40
TRAIN_SET_NAME = 'valid'
TEST_SET_NAME = 'hidden'

model = SVD(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
model.run_c = True
try:
    feature_epoch = False
    euclidean = False
    create_files = True
    if len(sys.argv) > 1:
        if 'order' in sys.argv:
            feature_epoch = True
        if 'euclidean' in sys.argv:
            euclidean = True
        if 'nofile' in sys.argv:
            create_files = False
    if euclidean:
        model = SVDEuclidean(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
    run(model, TRAIN_SET_NAME, TEST_SET_NAME,
        epochs=NUMBER_OF_EPOCHS, features=NUMBER_OF_FEATURES,
        feature_epoch_order=feature_epoch, create_files=create_files)
except Exception as exception:
    import pdb
    pdb.set_trace()
