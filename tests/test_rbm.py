import numpy as np

from algorithms.model import Model
from algorithms.rbm import RBM


def make_simple_train_points():
    train_ratings = ((1, 2, 0, 1),
                     (3, 4, 0, 2),
                     (5, 1, 0, 3),
                     (2, 3, 0, 4),
                     (4, 5, 0, 5),
                     (1, 3, 0, 1),
                     (5, 2, 0, 2))
    return np.array(train_ratings, dtype=np.int32)


def test_rbm_init_can_create_instance():
    model = RBM()
    assert isinstance(model, RBM)


def test_rbm_init_inherits_model_init():
    model = RBM()
    assert isinstance(model, Model)


def test_rbm_init_sets_default_learn_rate():
    default_learn_rate = 0.001
    model = RBM()
    assert model.learn_rate == default_learn_rate


def test_rbm_init_sets_default_num_hidden():
    default_num_hidden = 100
    model = RBM()
    assert model.num_hidden == default_num_hidden


def test_rbm_init_sets_default_num_visible():
    default_num_visible = 100
    model = RBM()
    assert model.num_visible == default_num_visible


def test_rbm_init_can_set_custom_learn_rate():
    from random import random
    unique_learn_rate = random()
    model = RBM(learn_rate=unique_learn_rate)
    assert model.learn_rate == unique_learn_rate


def test_rbm_init_can_set_custom_num_hidden():
    from random import random
    unique_num_hidden = random()
    model = RBM(num_hidden=unique_num_hidden)
    assert model.num_hidden == unique_num_hidden


def test_rbm_init_can_set_custom_num_visible():
    from random import random
    unique_num_visible = random()
    model = RBM(num_visible=unique_num_visible)
    assert model.num_visible == unique_num_visible


def test_rbm_init_sets_null_train_points():
    expected_train_points = np.array([])
    model = RBM()
    np.testing.assert_array_equal(model.train_points, expected_train_points)


def test_rbm_set_train_points_sets_expected_train_points():
    train_points = make_simple_train_points()
    model = RBM()
    model.set_train_points(train_points)
    np.testing.assert_array_equal(model.train_points, train_points)


