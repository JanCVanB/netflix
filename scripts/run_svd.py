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

feature_epoch = 'order' in sys.argv
euclidean = 'euclidean' in sys.argv
create_files = 'nofile' not in sys.argv
run_multi = 'multi' in sys.argv
run_c = 'noc' not in sys.argv
if euclidean:
    model = SVDEuclidean(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
else:
    model = SVD(learn_rate=0.001, num_features=NUMBER_OF_FEATURES)
model.run_c = run_c

try:
    run_name = ''
    while run_name == '':
        run_name = input('Please enter a run name:')
    run(model=model,
        train_set_name=TRAIN_SET_NAME,
        test_set_name=TEST_SET_NAME,
        epochs=NUMBER_OF_EPOCHS,
        feature_epoch_order=feature_epoch,
        run_name=run_name,
        create_files=create_files,
        run_multi=run_multi)
except Exception as the_exception:
    import pdb
    local_exception = the_exception
    pdb.set_trace()
