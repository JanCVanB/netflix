"""Tools for viewing and analyzing prediction results

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
.. moduleauthor:: Quinn Osha <qosha@caltech.edu>
"""
from fnmatch import fnmatch
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from os.path import abspath, dirname, join
from os import listdir
import sys

sys.path.append(abspath(dirname(dirname(__file__))))
from utils.data_paths import RESULTS_DIR_PATH


def graph_rmse_surface(algorithm_name, train_name, test_name, max_epochs,
                       min_features, max_features):
    desired_file_paths = []
    file_name_pattern = ('{alg}_{train}_*epochs_*features_rmse_{test}_*.txt'
                         .format(alg=algorithm_name, train=train_name,
                                 test=test_name))
    for file_name in listdir(RESULTS_DIR_PATH):
        if fnmatch(file_name, file_name_pattern):
            file_name_parts = split_results_file_name(file_name)
            file_name_epochs = file_name_parts['epochs']
            file_name_features = file_name_parts['features']
            result = file_name_parts['result']
            if (result == 'rmse' and file_name_epochs == max_epochs and
                    min_features <= file_name_features <= max_features):
                file_path = join(RESULTS_DIR_PATH, file_name)
                desired_file_paths.append(file_path)

    points = []
    for file_path in desired_file_paths:
        file_name_parts = split_results_file_name(file_path)
        features = file_name_parts['features']
        rmse_values = []
        with open(file_path) as rmse_file:
            for line in rmse_file:
                rmse_value = float(line.strip())
                rmse_values.append(rmse_value)
        if len(rmse_values) == 1:
            points.append((max_epochs, features, rmse_values[0]))
        else:
            epoch = 0
            for rmse_value in rmse_values:
                epoch += 1
                points.append((epoch, features, rmse_value))
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    x = [value[0] for value in points]
    y = [value[1] for value in points]
    z = [value[2] for value in points]
    axes.set_xlim(min(x), max(x))
    axes.set_ylim(min(y), max(y))
    axes.set_zlim(int(min(z) * 100) / 100, int(max(z) * 100 + 1) / 100)
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    min_rmse_index = z.index(min(z))
    min_rmse_x = x[min_rmse_index]
    min_rmse_y = y[min_rmse_index]
    min_rmse_z = z[min_rmse_index]
    min_rmse_color = '#00DD00'
    axes.plot([min_rmse_x] * 2, axes.get_ylim(), zs=[axes.get_zlim()[0]] * 2,
              color=min_rmse_color, ls=':')
    axes.plot(axes.get_xlim(), [min_rmse_y] * 2, zs=[axes.get_zlim()[0]] * 2,
              color=min_rmse_color, ls=':')
    axes.plot([min_rmse_x] * 2, [min_rmse_y] * 2,
              zs=[axes.get_zlim()[0], min_rmse_z], color=min_rmse_color, ls=':')
    if len(set(x)) == 1 or len(set(y)) == 1:
        axes.plot(x, y, z)
    else:
        axes.plot_trisurf(x, y, z, cmap=cm.CMRmap, linewidth=0)
    axes.set_title('Results from training on {train}, testing on {test}'
                   .format(train=train_name, test=test_name))
    axes.set_xlabel('Number of Epochs')
    axes.set_ylabel('Number of Features')
    axes.set_zlabel('RMSE ')

    xp, yp, _ = proj3d.proj_transform(min_rmse_x, min_rmse_y, min_rmse_z,
                                      axes.get_proj())
    label = axes.annotate(
        '{:.3g}'.format(min_rmse_z), xy=(xp, yp), xytext = (-20, 40),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle='round,pad=0.5', fc=min_rmse_color, alpha=0.5),
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                          color=min_rmse_color))

    def update_position(e):
        xp, yp, _ = proj3d.proj_transform(min_rmse_x, min_rmse_y, min_rmse_z,
                                          axes.get_proj())
        label.xy = xp, yp
        label.update_positions(figure.canvas.renderer)
        figure.canvas.draw()
    figure.canvas.mpl_connect('button_release_event', update_position)
    plt.savefig(join(RESULTS_DIR_PATH, 'analyze_results.png'))
    plt.show()


def lowest_rmse(file_name):
    rmse_values = []
    rmse_file_path = join(RESULTS_DIR_PATH, file_name)
    with open(rmse_file_path, 'r') as rmse_file:
        for line in rmse_file:
            rmse_value = line.strip();
            rmse_values.append(rmse_value)
    return min(rmse_values)


def split_results_file_name(file_name):
    algorithm, train, epochs, features, result, test, *_ = file_name.split('_')
    parts = {'algorithm': algorithm,
             'epochs': int(epochs[:epochs.index('e')]),
             'features': int(features[:features.index('f')]),
             'result': result,
             'test': test,
             'train': train}
    return parts


if __name__ == '__main__':
    graph_rmse_surface(algorithm_name='svd', train_name='base',
                       test_name='valid', max_epochs=8,
                       min_features=40, max_features=60)
