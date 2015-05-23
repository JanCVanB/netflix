# Inspired by Edwin Chen: github.com/echen/restricted-boltzmann-machines
# and Salakhutdinov/Mnih/Hinton 2007: www.cs.toronto.edu/~amnih/papers/rbmcf.pdf
# and Louppe/Guerts 2010: http://orbi.ulg.ac.be/handle/2268/74400
import numpy

import algorithms.model


class RBM(algorithms.model.Model):
    def __init__(self, learn_rate=0.001, num_hidden=100):
        self.learn_rate = learn_rate
        self.num_hidden = num_hidden
        self.num_visible = 0  # later set to the number of movies
        # TODO: momentum, multiple learning rates
        # TODO (maybe): soft_max, weight_cost, final_momentum?
        self.train_points = numpy.array([])
        self.weights = numpy.array([])
        self.hidden_biases = numpy.array([])
        self.visible_biases = numpy.array([])

    def train(self, train_points, number_of_epochs=1):
        self.set_train_points(train_points)
        self.initialize_weights()
        self.run_multiple_epochs(number_of_epochs)

    def set_train_points(self, train_points):
        self.train_points = train_points
        self.num_visible = max(train_points[:, 1])

    def initialize_weights(self):
        self.weights = 0.1 * numpy.random.randn(self.num_visible,
                                                self.num_hidden)
        self.hidden_biases = numpy.zeros(self.num_hidden)
        self.visible_biases = numpy.zeros(self.num_visible)

    def run_multiple_epochs(self, number_of_epochs):
        for epoch in range(number_of_epochs):
            self.run_one_epoch()

    def run_one_epoch(self):
        pos_hid_probs, pos_hid_states = self.positive_cd_results()
        neg_hid_probs, neg_vis_probs = self.negative_cd_results(pos_hid_states)
        positive_associations = self.positive_associations(pos_hid_probs)
        negative_associations = self.negative_associations(neg_hid_probs,
                                                           neg_vis_probs)
        self.update_weights(positive_associations, negative_associations)

    def positive_cd_results(self):
        hidden_activations = self.hidden_activations()
        hidden_probabilities = self.hidden_probabilities(hidden_activations)
        hidden_states = self.hidden_states(hidden_probabilities)
        return hidden_probabilities, hidden_states
