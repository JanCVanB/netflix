from os.path import abspath, dirname
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from algorithms.svd import SVD
from algorithms.svd_euclidean import SVDEuclidean
from scripts.run_model import run

NUMBER_OF_EPOCHS = 120
NUMBER_OF_FEATURES = 50
TRAIN_SET_NAME = 'base'
TEST_SET_NAME = 'probe'

feature_epoch = False
euclidean = False
create_files = True
run_multi = False
run_c = True
if len(sys.argv) > 1:
    if 'order' in sys.argv:
        feature_epoch = True
    if 'euclidean' in sys.argv:
        euclidean = True
    if 'nofile' in sys.argv:
        create_files = False
    if 'multi' in sys.argv:
        run_multi = True
    if 'noc' in sys.argv:
        run_c = False
if euclidean:
    model = SVDEuclidean(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
else:
    model = SVD(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
model.run_c = run_c

try:
    run(model, TRAIN_SET_NAME, TEST_SET_NAME,
        epochs=NUMBER_OF_EPOCHS, features=NUMBER_OF_FEATURES,
        feature_epoch_order=feature_epoch, create_files=create_files,
        run_multi=run_multi)
except Exception as the_exception:
    import pdb
    local_exception = the_exception
    pdb.set_trace()
