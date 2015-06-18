"""Graph prediction results from the file paths in the command-line arguments

Produce three graphs:

- RMSE surface over epochs vs. features
- RMSE surface over epochs vs. learning rates
- RMSE surface over features vs. learning rates

Every graph shows the minimum RMSE value for each x, y point
If the third parameter also varies over the same space,
this might produce misleading or disjointed surfaces
(This was chosen instead of drawing multiple surfaces or all z-values)

.. moduleauthor:: Jan Van Bruggen <jancvanbruggen@gmail.com>
"""
import json
import math
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
    infos = []
    for rmse_file_path in rmse_file_paths:
        result = Result(rmse_file_path)
        infos.append(result.info)
    for info in infos:
        for key, value in info.__dict__.items():
            if key not in combined_info_dict:
                combined_info_dict[key] = [value]
            elif value not in combined_info_dict[key]:
                combined_info_dict[key].append(value)
    for key, value in combined_info_dict.items():
        if len(value) == 1:
            combined_info_dict[key] = value[0]
        else:
            value.sort()
            combined_info_dict[key] = ' and '.join(
                [', '.join(str(v) for v in value[:-1]),
                 str(value[-1])]
            )
    return ResultInfo(combined_info_dict)


def get_points(rmse_file_paths):
    points = []
    for rmse_file_path in rmse_file_paths:
        result = Result(rmse_file_path)
        with open(rmse_file_path) as rmse_file:
            for epoch, line in enumerate(rmse_file):
                points.append(Point(epoch=epoch + 1,
                                    feature=result.info.num_features,
                                    learn_rate=result.info.learn_rate,
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
        self.num_epochs = info_dict['num_epochs']
        self.num_features = info_dict['num_features']
        self.learn_rate = info_dict['learn_rate']
        self.train_set_name = info_dict['train_set_name']
        self.test_set_name = info_dict['test_set_name']


class Point:
    def __init__(self, epoch, feature, learn_rate, rmse):
        self.epoch = epoch
        self.feature = feature
        self.learn_rate = learn_rate
        self.rmse = rmse

    def __repr__(self):
        return ('Epoch {e}, Feature {f}, Learn {lr}, RMSE {r}'
                .format(e=self.epoch, f=self.feature, lr=self.learn_rate,
                        r=self.rmse))


def graph_all_surfaces(info, points):
    figure_e_vs_f, axes_e_vs_f = get_figure_and_axes_for_epoch_vs_feature(info)
    figure_e_vs_l, axes_e_vs_l = get_figure_and_axes_for_epoch_vs_learn(info)
    figure_f_vs_l, axes_f_vs_l = get_figure_and_axes_for_feature_vs_learn(info)
    epochs = [point.epoch for point in points]
    features = [point.feature for point in points]
    learns = [point.learn_rate for point in points]
    rmses = [point.rmse for point in points]
    graph_surface(figure_e_vs_f, axes_e_vs_f, epochs, features, rmses)
    graph_surface(figure_e_vs_l, axes_e_vs_l, epochs, learns, rmses)
    graph_surface(figure_f_vs_l, axes_f_vs_l, features, learns, rmses)
    figure_e_vs_f.savefig(os.path.join(RESULTS_DIR_PATH,
                                       'last_epoch_vs_feature_graph.png'))
    figure_e_vs_l.savefig(os.path.join(RESULTS_DIR_PATH,
                                       'last_epoch_vs_learn_graph.png'))
    figure_f_vs_l.savefig(os.path.join(RESULTS_DIR_PATH,
                                       'last_feature_vs_learn_graph.png'))


def get_figure_and_axes_for_epoch_vs_feature(info):
    title = ('Epochs vs. Features ({train} to {test})\n Learning Rate {lr}'
             .format(train=info.train_set_name,
                     test=info.test_set_name,
                     lr=info.learn_rate))
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title(title)
    axes.set_xlabel('Number of Epochs')
    axes.set_ylabel('Number of Features')
    axes.set_zlabel('RMSE ')
    return figure, axes


def get_figure_and_axes_for_epoch_vs_learn(info):
    title = ('Epochs vs. Learning Rates ({train} to {test})\n {f} Features'
             .format(train=info.train_set_name,
                     test=info.test_set_name,
                     f=info.num_features))
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title(title)
    axes.set_xlabel('Number of Epochs')
    axes.set_ylabel('Learning Rate')
    axes.set_zlabel('RMSE ')
    return figure, axes


def get_figure_and_axes_for_feature_vs_learn(info):
    title = ('Features vs. Learning Rates ({train} to {test})\n {e} Epochs'
             .format(train=info.train_set_name,
                     test=info.test_set_name,
                     e=info.num_epochs))
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    axes = figure.add_subplot(111, projection='3d')
    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    axes.set_title(title)
    axes.set_xlabel('Number of Features')
    axes.set_ylabel('Learning Rate')
    axes.set_zlabel('RMSE ')
    return figure, axes


def graph_surface(figure, axes, xs, ys, rmse_values):
    xs, ys, rmse_values = sorted_minima(xs, ys, rmse_values)
    min_rmse_value = min(rmse_values)
    min_rmse_index = rmse_values.index(min_rmse_value)
    min_rmse_x_value = xs[min_rmse_index]
    min_rmse_y_value = ys[min_rmse_index]
    min_rmse_color = '#00DD00'
    only_one_x_value = len(set(xs)) == 1
    only_one_y_value = len(set(ys)) == 1
    if only_one_x_value:
        axes.plot(xs, ys, rmse_values)
        axes.set_xlim(get_one_below_and_one_above(xs[0]))
    elif only_one_y_value:
        axes.plot(xs, ys, rmse_values)
        axes.set_ylim(get_one_below_and_one_above(ys[0]))
    else:
        axes.set_xlim(min(xs), max(xs))
        axes.set_ylim(min(ys), max(ys))
        try:
            axes.plot_trisurf(xs, ys, rmse_values, cmap=cm.CMRmap, linewidth=0)
        except ValueError:
            axes.plot(xs, ys, rmse_values)
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
    annotate_point(figure, axes,
                   min_rmse_x_value, min_rmse_y_value, min_rmse_value,
                   min_rmse_color)


def sorted_minima(xs, ys, zs):
    points = zip(xs, ys, zs)
    unique_points = set(points)
    minimum_points = []
    for new_point in unique_points:
        for old_index, old_point in enumerate(minimum_points):
            if old_point[:2] == new_point[:2]:
                if new_point[2] < old_point[2]:
                    minimum_points[old_index] = new_point
                break
        else:
            minimum_points.append(new_point)
    sorted_minimum_points = sorted(minimum_points)
    xs, ys, zs = zip(*sorted_minimum_points)
    return xs, ys, zs


def get_one_below_and_one_above(x):
    delta = math.pow(10, math.floor(math.log10(x)))
    return x - delta, x + delta


def annotate_point(figure, axes, x, y, z, color):
    xp1, yp1, _ = proj3d.proj_transform(x, y, z, axes.get_proj())
    label = axes.annotate(
        '{:.5g}'.format(z), xy=(xp1, yp1), xytext = (-20, 40),
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
