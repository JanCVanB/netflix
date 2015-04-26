from math import sqrt
from os.path import abspath, dirname, join
import sys
from time import localtime, strftime

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_io import load_numpy_array_from_file
from utils.data_paths import DATA_DIR_PATH, RESULTS_DIR_PATH


def calculate_rmse(true_ratings, predictions):
    return sqrt(((predictions - true_ratings) ** 2).mean())


def run(model, train_set_name, test_set_name, epochs=None, features=None):
    print('Training {model_class} on "{train}" ratings'
          .format(model_class=model.__class__.__name__, train=train_set_name))
    if epochs is not None:
        print('Number of epochs:', epochs)
    if features is not None:
        print('Number of features:', features)
    time_format = '%b-%d-%Hh-%Mm'
    train_file_path = join(DATA_DIR_PATH, train_set_name + '.npy')

    model.debug = True
    train_points = load_numpy_array_from_file(train_file_path)
    times = strftime(time_format, localtime())
    model.train(train_points, epochs)
    times += '_to_' + strftime(time_format, localtime())
    model.train_points = None

    epochs_string = '' if epochs is None else ('_%sepochs' % epochs)
    features_string = '' if features is None else ('_%sfeatures' % features)
    template_file_name = ('svd_{train}{e}{f}_xxx_{times}'
                          .format(train=train_set_name, e=epochs_string,
                                  f=features_string, times=times))

    model_file_name = template_file_name.replace('xxx', 'model') + '.p'
    model.save(model_file_name)

    print('Predicting "{test}" ratings'.format(test=test_set_name))
    test_file_path = join(DATA_DIR_PATH, test_set_name + '.npy')
    test_points = load_numpy_array_from_file(test_file_path)
    predictions = model.predict(test_points)
    predictions_file_name = (template_file_name.replace('xxx', 'predictions') +
                             '.dta')
    save_predictions(predictions, predictions_file_name)

    true_ratings = test_points[:, 3]
    rmse = calculate_rmse(true_ratings, predictions)
    print('RMSE:', rmse)
    rmse_file_name = (template_file_name.replace('xxx', 'rmse_' + test_set_name)
                      + '.txt')
    save_rmse(rmse, rmse_file_name)


def run_multi(model, train_set_name, test_set_name, epochs=None, features=None):
    print('Training {model_class} on "{train}" ratings'
          .format(model_class=model.__class__.__name__, train=train_set_name))
    if epochs is not None:
        print('Maximum number of epochs:', epochs)
    if features is not None:
        print('Number of features:', features)

    train_file_path = join(DATA_DIR_PATH, train_set_name + '.npy')
    train_points = load_numpy_array_from_file(train_file_path)
    test_file_path = join(DATA_DIR_PATH, test_set_name + '.npy')
    test_points = load_numpy_array_from_file(test_file_path)

    epochs_string = '' if epochs is None else ('_%sepochs' % epochs)
    features_string = '' if features is None else ('_%sfeatures' % features)
    time_format = '%b-%d-%Hh-%Mm'
    rmse_file_name = ('svd_{train}{e}{f}_rmse_{test}_{time}.txt'
                      .format(train=train_set_name, test=test_set_name,
                              e=epochs_string, f=features_string,
                              time=strftime(time_format, localtime())))

    model.debug = True
    for epoch in range(epochs):
        print('Training epoch {}:'.format(epoch))
        if epoch == 0:
            model.train(train_points, epochs=1)
        else:
            model.train_more(epochs=1)
        print('Predicting "{test}" ratings'.format(test=test_set_name))
        predictions = model.predict(test_points)
        true_ratings = test_points[:, 3]
        rmse = calculate_rmse(true_ratings, predictions)
        print('RMSE:', rmse)
        rmse_file_name = rmse_file_name
        save_rmse(rmse, rmse_file_name, append=True)
    model.train_points = None


def save_predictions(predictions, predictions_file_name):
    predictions_file_path = join(RESULTS_DIR_PATH, predictions_file_name)
    with open(predictions_file_path, 'w+') as predictions_file:
        predictions_file.writelines(['{:.3f}\n'.format(p) for p in predictions])


def save_rmse(rmse, rmse_file_name, append=False):
    rmse_file_path = join(RESULTS_DIR_PATH, rmse_file_name)
    write_format = 'w+'
    if append:
        write_format = 'a+'
    with open(rmse_file_path, write_format) as rmse_file:
        rmse_file.write('{}\n'.format(rmse))
