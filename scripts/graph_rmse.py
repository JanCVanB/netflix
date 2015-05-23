"""Graph prediction results from the file paths in the command-line arguments

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import json
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.data_paths import RESULTS_DIR_PATH


def main():
    rmse_file_paths = sys.argv[1:]
    info = get_info(rmse_file_paths)
    points = get_points(rmse_file_paths)
    graph_all_surfaces(info, points)
    plt.show()


def get_info(rmse_file_paths):
    combined_info_dict = {}
    info_dicts = []
    for rmse_file_path in rmse_file_paths:
        result = Result(rmse_file_path)
        info_dicts.append(result.info)
    for info_dict in info_dicts:
        for key, value in info_dict.items():
            if key not in combined_info_dict:
                combined_info_dict[key] = [value]
            else:
                combined_info_dict[key].append(value)
    for key, value in combined_info_dict.items():
        if len(value) == 1:
            combined_info_dict[key] = value[0]
    return ResultInfo(combined_info_dict)


def get_points(rmse_file_paths):
    points = []
    for rmse_file_path in rmse_file_paths:
        result = Result(rmse_file_path)
        with open(rmse_file_path) as rmse_file:
            for epoch, line in enumerate(rmse_file):
                points.append(Point(epoch=epoch + 1,
                                    feature=result.info.num_features,
                                    learn=result.info.learning_rate,
                                    rmse=float(line.strip())))
    return points


class Result:
    def __init__(self, rmse_file_path):
        self.rmse_file_path = rmse_file_path
        self.info_file_path = rmse_file_path.replace('rmse.txt', 'info.json')
        with open(self.info_file_path, 'r') as info_file:
            info_dict = json.load(info_file)
        self.info = ResultInfo(info_dict)

    def __repr__(self):
        return self.rmse_file_path


class ResultInfo:
    def __init__(self, info_dict):
        self.algorithm = info_dict['algorithm']
        self.last_commit = info_dict['commit']
        self.name = info_dict['name']
        self.time = info_dict['time']
        self.num_epochs = info_dict['epochs']
        self.num_features = info_dict['features']
        self.learning_rate = info_dict['learn']
        self.train_set_name = info_dict['train']
        self.test_set_name = info_dict['test']


class Point:
    def __init__(self, epoch, feature, learn, rmse):
        self.epoch = epoch
        self.feature = feature
        self.learn = learn
        self.rmse = rmse


def graph_all_surfaces(info, points):
    figure_ef, axes_ef = get_epoch_feature_figure_and_axes(info)
    figure_el, axes_el = get_epoch_learn_figure_and_axes(info)
    figure_fl, axes_fl = get_feature_learn_figure_and_axes(info)
    epochs = [point.epoch for point in points]
    features = [point.feature for point in points]
    learns = [point.learning_rate for point in points]
    rmses = [point.rmse for point in points]
    graph_surface(figure_ef, axes_ef, epochs, features, rmses)
    graph_surface(figure_el, axes_el, epochs, learns, rmses)
    graph_surface(figure_fl, axes_fl, features, learns, rmses)
    plt.savefig(os.path.join(RESULTS_DIR_PATH, 'rmse_graph.png'))


def get_epoch_feature_figure_and_axes(info):
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title('Epochs vs. Features ({train} to {test})'
                   .format(train=info.train_set_name,
                           test=info.test_set_name))
    axes.set_xlabel('Number of Epochs')
    axes.set_ylabel('Number of Features')
    axes.set_zlabel('RMSE ')
    return figure, axes


def get_epoch_learn_figure_and_axes(info):
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    # TODO: learning rate locator
    # axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title('Epochs vs. Learning Rates ({train} to {test})'
                   .format(train=info.train_set_name,
                           test=info.test_set_name))
    axes.set_xlabel('Number of Epochs')
    axes.set_ylabel('Learning Rate')
    axes.set_zlabel('RMSE ')
    return figure, axes


def get_feature_learn_figure_and_axes(info):
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    # TODO: learning rate locator
    # axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title('Features vs. Learning Rates ({train} to {test})'
                   .format(train=info.train_set_name,
                           test=info.test_set_name))
    axes.set_xlabel('Number of Features')
    axes.set_ylabel('Learning Rate')
    axes.set_zlabel('RMSE ')
    return figure, axes


def graph_surface(figure, axes, xs, ys, rmse_values):
    min_rmse_value = min(rmse_values)
    min_rmse_index = rmse_values.index(min_rmse_value)
    min_rmse_x_value = xs[min_rmse_index]
    min_rmse_y_value = ys[min_rmse_index]
    min_rmse_color = '#00DD00'
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    zlim = axes.get_zlim()
    axes.plot([min_rmse_x_value, min_rmse_x_value],
              ylim,
              zs=[zlim[0], zlim[0]],
              color=min_rmse_color, ls=':')
    axes.plot(xlim,
              [min_rmse_y_value, min_rmse_y_value],
              zs=[zlim[0], zlim[0]],
              color=min_rmse_color, ls=':')
    axes.plot([min_rmse_x_value, min_rmse_x_value],
              [min_rmse_y_value, min_rmse_y_value],
              zs=[zlim[0], min_rmse_value],
              color=min_rmse_color, ls=':')
    # TODO: always do surface?
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        axes.plot(xs, ys, rmse_values)
    else:
        axes.plot_trisurf(xs, ys, rmse_values, cmap=cm.CMRmap, linewidth=0)
    annotate_point(figure, axes,
                   min_rmse_x_value, min_rmse_y_value, min_rmse_value,
                   min_rmse_color)


def annotate_point(figure, axes, x, y, z, color):
    xp1, yp1, _ = proj3d.proj_transform(x, y, z, axes.get_proj())
    label = axes.annotate(
        '{:.3g}'.format(z), xy=(xp1, yp1), xytext = (-20, 40),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                          color=color))

    def update_position(_):
        xp2, yp2, _ = proj3d.proj_transform(x, y, z, axes.get_proj())
        label.xy = xp2, yp2
        label.update_positions(figure.canvas.renderer)
        figure.canvas.draw()
    figure.canvas.mpl_connect('button_release_event', update_position)


if __name__ == '__main__':
    main()
