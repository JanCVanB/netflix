from __future__ import print_function
from math import sqrt
from os.path import abspath, dirname, join
import sys
from time import localtime, strftime
from git import Repo
import json

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_io import load_numpy_array_from_file
from utils.data_stats import load_stats_from_file
from utils.data_paths import DATA_DIR_PATH, RESULTS_DIR_PATH


def calculate_rmse(true_ratings, predictions):
    return sqrt(((predictions - true_ratings) ** 2).mean())


def predict_and_save_rmse(model, test_points, rmse_file_path,
                          keep_predictions=False,
                          predictions_file_name='noname'):
    predictions = model.predict(test_points)
    true_ratings = test_points[:, 3]
    rmse = calculate_rmse(true_ratings, predictions)
    print('RMSE:', rmse)
    save_rmse(rmse, rmse_file_path, append=True)
    if keep_predictions:
        save_predictions(predictions, predictions_file_name)


def save_model(model, model_file_name):
    print('Saving model to {}'.format(model_file_name))
    model.save(model_file_name)


def save_run_info(model, test_set_name, train_set_name, date_string,
                  time_string, feature_epoch_order, create_files,
                  epochs, run_multi, run_name, commit):
    info_file_name = ('{model_class}_{run_name}_{short_commit}_{start_time}'
                      '_info.json'
                      .format(model_class=model.__class__.__name__,
                              short_commit=commit[:5],
                              run_name=run_name,
                              start_time=time_string)
                      )
    info_file_path = join(RESULTS_DIR_PATH, info_file_name)
    # Create a dict of data
    excluded_params = ['users', 'movies', 'train_points', 'residuals',
                       'stats', 'max_movie', 'max_user']
    run_info = {key: value for key, value in model.__dict__.items()
                if key not in excluded_params}
    run_info['algorithm'] = model.__class__.__name__
    run_info['last_commit'] = commit
    run_info['train_set_name'] = train_set_name
    run_info['name'] = run_name
    run_info['time'] = time_string
    run_info['date'] = date_string
    run_info['test_set_name'] = test_set_name
    run_info['num_epochs'] = epochs
    run_info['create_files'] = create_files
    run_info['run_multi'] = run_multi
    run_info['feature_epoch_order'] = feature_epoch_order
    json.dump(run_info, open(info_file_path, 'w'), indent=4,
              sort_keys=True)
    return info_file_path


def run(model, train_set_name, test_set_name, run_name, epochs=None,
        feature_epoch_order=False, create_files=True, run_multi=False):
    print('Training {model_class} on "{train}" ratings'
          .format(model_class=model.__class__.__name__, train=train_set_name))
    if not create_files:
        print("WARNING: 'nofile' flag detected. No model file will be " +
              "saved to disk after this run.\n***MODEL WILL BE LOST.")
        confirm = input("Are you sure you want to continue? [Y/n]")
        if confirm == 'Y' or confirm == 'y' or confirm == '':
            pass
        else:
            return
    if epochs is not None:
        print('Number of epochs:', epochs)
    if model.num_features is not None:
        print('Number of features:', model.num_features)
    train_file_path = join(DATA_DIR_PATH, train_set_name + '.npy')
    stats_file_path = join(DATA_DIR_PATH, 'old_stats', train_set_name +
                           '_stats.p')

    model.debug = True
    train_points = load_numpy_array_from_file(train_file_path)
    stats = load_stats_from_file(stats_file_path)
    test_file_path = join(DATA_DIR_PATH, test_set_name + '.npy')
    test_points = load_numpy_array_from_file(test_file_path)

    # Save run information in [...]_info.txt file
    date_format = '%b-%d'
    time_format = '%H%M'
    latest_commit = Repo('.').commit('HEAD').hexsha
    date_string = strftime(date_format, localtime())
    time_string = strftime(time_format, localtime())
    run_info_file_path = save_run_info(
        model=model, test_set_name=test_set_name,
        train_set_name=train_set_name,
        epochs=epochs,
        time_string=time_string,
        date_string=date_string,
        feature_epoch_order=feature_epoch_order,
        create_files=create_files,
        run_multi=run_multi,
        run_name=run_name,
        commit=latest_commit
    )
    print('Wrote run info to ', run_info_file_path)
    rmse_file_path = run_info_file_path.replace('info.json', 'rmse.txt')
    predictions_file_name = (run_info_file_path.split('/')[-1]
                             .replace('info.json', 'predictions.dta'))
    if not run_multi:
        if not feature_epoch_order:
            model.train(train_points, stats=stats, epochs=epochs)
        else:
            model.train_feature_epoch(train_points=train_points, stats=stats,
                                      epochs=epochs)
    else:
        print("Training multi!")
        for epoch in range(epochs):
            if epoch == 0:
                model.train(train_points, stats=stats, epochs=1)
            else:
                model.train_more(epochs=1)
            if create_files:
                print('Predicting "{test}" ratings'.format(test=test_set_name))
                predict_and_save_rmse(
                    model, test_points=test_points,
                    rmse_file_path=rmse_file_path,
                    keep_predictions=(create_files and epoch == epochs-1),
                    predictions_file_name=predictions_file_name
                )
    model.train_points = None
    if create_files:
        model_file_name = (run_info_file_path.split('/')[-1]
                           .replace('info.json', 'model.p'))
        save_model(model, model_file_name)
        if not run_multi:
            # duplicate save if run_multi
            print('Predicting "{test}" ratings'.format(test=test_set_name))
            predict_and_save_rmse(model, test_points=test_points,
                                  rmse_file_path=rmse_file_path,
                                  keep_predictions=True,
                                  predictions_file_name=predictions_file_name)


def save_predictions(predictions, predictions_file_name):
    print('Saving predictions to {}'.format(predictions_file_name))
    predictions_file_path = join(RESULTS_DIR_PATH, predictions_file_name)
    with open(predictions_file_path, 'w+') as predictions_file:
        predictions_file.writelines(['{:.3f}\n'.format(p) for p in predictions])


def save_rmse(rmse, rmse_file_path, append=True):
    write_format = 'w+'
    if append:
        write_format = 'a+'
    with open(rmse_file_path, write_format) as rmse_file:
        rmse_file.write('{}\n'.format(rmse))
