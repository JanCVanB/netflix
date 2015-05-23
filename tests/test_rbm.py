import numpy
import random
import unittest.mock

import algorithms.model
import algorithms.rbm


MockToSkip = unittest.mock.Mock
MockToTrack = unittest.mock.Mock


def assert_almost_equal(a, b, delta):
    assert abs(a - b) <= delta


def make_simple_train_points():
    train_ratings = ((1, 2, 0, 1),
                     (3, 4, 0, 2),
                     (5, 1, 0, 3),
                     (2, 3, 0, 4),
                     (4, 5, 0, 5),
                     (1, 3, 0, 1),
                     (5, 2, 0, 2))
    return numpy.array(train_ratings, dtype=numpy.int32)


def test_rbm_init_can_create_instance():
    model = algorithms.rbm.RBM()
    assert isinstance(model, algorithms.rbm.RBM)


def test_rbm_init_inherits_model_init():
    model = algorithms.rbm.RBM()
    assert isinstance(model, algorithms.model.Model)


def test_rbm_init_sets_default_learn_rate():
    default_learn_rate = 0.001
    model = algorithms.rbm.RBM()
    assert model.learn_rate == default_learn_rate


def test_rbm_init_sets_default_num_hidden():
    default_num_hidden = 100
    model = algorithms.rbm.RBM()
    assert model.num_hidden == default_num_hidden


def test_rbm_init_can_set_custom_learn_rate():
    from random import random
    unique_learn_rate = random()
    model = algorithms.rbm.RBM(learn_rate=unique_learn_rate)
    assert model.learn_rate == unique_learn_rate


def test_rbm_init_can_set_custom_num_hidden():
    from random import random
    unique_num_hidden = random()
    model = algorithms.rbm.RBM(num_hidden=unique_num_hidden)
    assert model.num_hidden == unique_num_hidden


def test_rbm_init_sets_null_train_points():
    expected_train_points = numpy.array([])
    model = algorithms.rbm.RBM()
    numpy.testing.assert_array_equal(model.train_points, expected_train_points)


def test_rbm_init_sets_null_weights():
    expected_weights = numpy.array([])
    model = algorithms.rbm.RBM()
    numpy.testing.assert_array_equal(model.weights, expected_weights)


def test_rbm_train_initializes_weights():
    model = algorithms.rbm.RBM()
    model.initialize_weights = MockToTrack()
    model.set_train_points = MockToSkip()
    model.run_multiple_epochs = MockToSkip()
    model.train(train_points=None)
    assert model.initialize_weights.call_count == 1


def test_rbm_train_sets_training_points():
    train_points = object()
    model = algorithms.rbm.RBM()
    model.initialize_weights = MockToSkip()
    model.set_train_points = MockToTrack()
    model.run_multiple_epochs = MockToSkip()
    model.train(train_points=train_points)
    model.set_train_points.assert_called_once_with(train_points)


def test_rbm_train_runs_default_number_of_epochs():
    default_number_of_epochs = 1
    model = algorithms.rbm.RBM()
    model.initialize_weights = MockToSkip()
    model.set_train_points = MockToSkip()
    model.run_multiple_epochs = MockToTrack()
    model.train(train_points=None)
    model.run_multiple_epochs.assert_called_once_with(default_number_of_epochs)


def test_rbm_train_runs_custom_number_of_epochs():
    custom_numbers_of_epochs = 2, 3, 10
    model = algorithms.rbm.RBM()
    model.initialize_weights = MockToSkip()
    model.set_train_points = MockToSkip()
    model.run_multiple_epochs = MockToTrack()
    for custom_number_of_epochs in custom_numbers_of_epochs:
        model.train(train_points=None, number_of_epochs=custom_number_of_epochs)
        model.run_multiple_epochs.assert_called_with(custom_number_of_epochs)


def test_rbm_set_train_points_sets_same_train_points_passed_to_it():
    train_points = make_simple_train_points()
    model = algorithms.rbm.RBM()
    model.set_train_points(train_points)
    numpy.testing.assert_array_equal(model.train_points, train_points)


def test_rbm_set_train_points_sets_num_visible_to_max_movie_id():
    train_points = make_simple_train_points()
    expected_num_visible = max(train_points[:, 1])
    model = algorithms.rbm.RBM()
    model.set_train_points(train_points)
    assert model.num_visible == expected_num_visible


def test_rbm_initialize_weights_and_biases_sets_expected_shapes():
    arbitrary_num_hidden = 12
    arbitrary_num_visible = 34
    expected_shape = arbitrary_num_visible, arbitrary_num_hidden
    model = algorithms.rbm.RBM(num_hidden=arbitrary_num_hidden)
    model.num_visible = arbitrary_num_visible
    model.initialize_weights()
    numpy.testing.assert_array_equal(model.weights.shape, expected_shape)


def test_rbm_initialize_weights_and_biases_sets_expected_weights_distribution():
    expected_mean = 0
    expected_stddev = 0.1
    delta = 0.01
    arbitrary_large_num_hidden = 99
    arbitrary_large_num_visible = 111
    model = algorithms.rbm.RBM(num_hidden=arbitrary_large_num_hidden)
    model.num_visible = arbitrary_large_num_visible
    model.initialize_weights()
    actual_mean = numpy.mean(model.weights)
    actual_stddev = numpy.std(model.weights)
    assert_almost_equal(actual_mean, expected_mean, delta)
    assert_almost_equal(actual_stddev, expected_stddev, delta)


def test_rbm_initialize_weights_and_biases_sets_expected_hidden_bias_values():
    arbitrary_num_hidden = 5
    arbitrary_num_visible = 7
    # TODO: log of the base rate? see Gilles Louppe's paper
    expected_hidden_biases = numpy.zeros(arbitrary_num_hidden)
    model = algorithms.rbm.RBM(num_hidden=arbitrary_num_hidden)
    model.num_visible = arbitrary_num_visible
    model.initialize_weights()
    numpy.testing.assert_array_equal(model.hidden_biases,
                                     expected_hidden_biases)


def test_rbm_initialize_weights_and_biases_sets_expected_visible_bias_zeros():
    arbitrary_num_hidden = 5
    arbitrary_num_visible = 7
    expected_visible_biases = numpy.zeros(arbitrary_num_visible)
    model = algorithms.rbm.RBM(num_hidden=arbitrary_num_hidden)
    model.num_visible = arbitrary_num_visible
    model.initialize_weights()
    numpy.testing.assert_array_equal(model.visible_biases,
                                     expected_visible_biases)


def test_rbm_run_multiple_epochs_runs_one_epoch_n_times_for_n_epochs():
    numbers_of_epochs = 2, 3, 10
    model = algorithms.rbm.RBM()
    model.run_one_epoch = MockToTrack()
    for number_of_epochs in numbers_of_epochs:
        model.run_multiple_epochs(number_of_epochs=number_of_epochs)
        assert model.run_one_epoch.call_count == number_of_epochs
        model.run_one_epoch.reset_mock()


# TODO: rename "positive associations"?
def test_rbm_run_one_epoch_gets_positive_associations_from_probabilities():
    model = algorithms.rbm.RBM()
    pos_hid_probs, pos_hid_states = object(), None
    neg_hid_probs, neg_vis_probs = None, None
    model.positive_cd_results = MockToSkip(return_value=(pos_hid_probs,
                                                         pos_hid_states))
    model.negative_cd_results = MockToSkip(return_value=(neg_hid_probs,
                                                         neg_vis_probs))
    model.positive_associations = MockToTrack()
    model.negative_associations = MockToSkip()
    model.update_weights = MockToSkip()
    model.run_one_epoch()
    model.positive_associations.assert_called_once_with(pos_hid_probs)


# TODO: rename "negative associations"?
def test_rbm_run_one_epoch_gets_negative_associations_from_probabilities():
    model = algorithms.rbm.RBM()
    pos_hid_probs, pos_hid_states = None, None
    neg_hid_probs, neg_vis_probs = object(), object()
    model.positive_cd_results = MockToSkip(return_value=(pos_hid_probs,
                                                         pos_hid_states))
    model.negative_cd_results = MockToSkip(return_value=(neg_hid_probs,
                                                         neg_vis_probs))
    model.positive_associations = MockToSkip()
    model.negative_associations = MockToTrack()
    model.update_weights = MockToSkip()
    model.run_one_epoch()
    model.negative_associations.assert_called_once_with(neg_hid_probs,
                                                        neg_vis_probs)


def test_rbm_run_one_epoch_updates_weights_with_associations():
    positive_association = object()
    negative_association = object()
    model = algorithms.rbm.RBM()
    pos_hid_probs, pos_hid_states = None, None
    neg_hid_probs, neg_vis_probs = None, None
    model.positive_cd_results = MockToSkip(return_value=(pos_hid_probs,
                                                         pos_hid_states))
    model.negative_cd_results = MockToSkip(return_value=(neg_hid_probs,
                                                         neg_vis_probs))
    model.positive_associations = MockToSkip(return_value=positive_association)
    model.negative_associations = MockToSkip(return_value=negative_association)
    model.update_weights = MockToTrack()
    model.run_one_epoch()
    model.update_weights.assert_called_once_with(positive_association,
                                                 negative_association)


def test_positive_cd_results_gets_hidden_activations():
    model = algorithms.rbm.RBM()
    model.hidden_activations = MockToTrack()
    model.hidden_probabilities = MockToSkip()
    model.hidden_states = MockToSkip()
    model.positive_cd_results()
    assert model.hidden_activations.call_count == 1


def test_positive_cd_results_gets_hidden_probabilities_from_activations():
    model = algorithms.rbm.RBM()
    hidden_activations = object()
    model.hidden_activations = MockToSkip(return_value=hidden_activations)
    model.hidden_probabilities = MockToTrack()
    model.hidden_states = MockToSkip()
    model.positive_cd_results()
    model.hidden_probabilities.assert_called_once_with(hidden_activations)


def test_positive_cd_results_gets_hidden_states_from_probabilities():
    model = algorithms.rbm.RBM()
    hidden_probabilities = object()
    model.hidden_activations = MockToSkip()
    model.hidden_probabilities = MockToSkip(return_value=hidden_probabilities)
    model.hidden_states = MockToTrack()
    model.positive_cd_results()
    model.hidden_states.assert_called_once_with(hidden_probabilities)


def test_positive_cd_results_returns_hidden_probabilities_and_states():
    model = algorithms.rbm.RBM()
    hidden_probabilities, hidden_states = object(), object()
    model.hidden_activations = MockToSkip()
    model.hidden_probabilities = MockToSkip(return_value=hidden_probabilities)
    model.hidden_states = MockToSkip(return_value=hidden_states)
    return_value = model.positive_cd_results()
    assert return_value == (hidden_probabilities, hidden_states)
