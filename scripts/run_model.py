from math import sqrt
from os.path import abspath, dirname, join
import sys
import pickle
from time import localtime, strftime

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_io import load_numpy_array_from_file
from utils.data_paths import (DATA_DIR_PATH, MODELS_DIR_PATH_IGNORE,
                              RESULTS_DIR_PATH)


def calculate_rmse(true_ratings, predictions):
    return sqrt(((predictions - true_ratings) ** 2).mean())


def run(model, train_set_name, test_set_name, epochs=None, features=None):
    print('Training {modelclass} on "{train}" ratings'
          .format(modelclass=model.__class__.__name__, train=train_set_name))
    if epochs is not None:
        print('Number of epochs:', epochs)
    if features is not None:
        print('Number of features:', features)
    time_format = '%b-%d-%Hh-%Mm'
    train_file_path = join(DATA_DIR_PATH, train_set_name + '.npy')
    time_stamp = strftime(time_format, localtime())

    model.debug = True
    train_points = load_numpy_array_from_file(train_file_path)
    model.train(train_points, epochs)
    time_stamp += '_to_' + strftime(time_format, localtime())
    model.train_points = None

    epochs_string = '' if epochs is None else ('_%sepochs' % epochs)
    features_string = '' if features is None else ('_%sfeatures' % features)
    template_file_name = ('svd_{train}{e}{f}_xxx_{time}'
                          .format(train=train_set_name, e=epochs_string,
                                  f=features_string, time=time_stamp))

    model_file_name = template_file_name.replace('xxx', 'model') + '.p'
    save_model(model, model_file_name)

    print('Predicting "{test}" ratings'.format(test=test_set_name))
    test_file_path = join(DATA_DIR_PATH, test_set_name + '.npy')
    test_points = load_numpy_array_from_file(test_file_path)
    predictions = model.predict(test_points)
    pred_file_name = template_file_name.replace('xxx', 'predictions') + '.dta'
    save_predictions(predictions, pred_file_name)

    true_ratings = test_points[:, 3]
    rmse = calculate_rmse(true_ratings, predictions)
    print('RMSE:', rmse)
    rmse_file_name = (template_file_name.replace('xxx', 'rmse_' + test_set_name)
                      + '.txt')
    save_rmse(rmse, rmse_file_name)


def save_model(model, model_file_name):
    model_file_path = join(MODELS_DIR_PATH_IGNORE, model_file_name)
    with open(model_file_path, 'wb+') as model_file:
        pickle.dump(model, model_file)


def save_predictions(predictions, predictions_file_name):
    predictions_file_path = join(RESULTS_DIR_PATH, predictions_file_name)
    with open(predictions_file_path, 'w+') as predictions_file:
        predictions_file.writelines(['{:.3f}\n'.format(p) for p in predictions])


def save_rmse(rmse, rmse_file_name):
    rmse_file_path = join(RESULTS_DIR_PATH, rmse_file_name)
    with open(rmse_file_path, 'w+') as rmse_file:
        rmse_file.write(str(rmse))
