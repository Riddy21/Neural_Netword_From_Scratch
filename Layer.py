#!/usr/local/bin/python3

import numpy as np
import random

class Layer(object):
    """
    Layer of neurons
    """
    def __init__(self, n_inputs, n_nodes, is_output_layer=False):
        # size info
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        # initialize weights as list of length n_nodes of random floats from -1to1
        # NOTE: self.weights[nodes][inputs] if using non-numpy lists
        #self.weights = self.random_sample(n_nodes, n_inputs, -1, 1)
        # trying better initializations...
        self.weights = self.he_random_init(n_nodes, n_inputs)

        # initalize biases to be the same size list of 0s
        # NOTE: self.biases[nodes] if using non-numpy lists
        self.biases = np.zeros(n_nodes)

        # initialize empty list of output values
        self.activations = None

        self.is_output_layer = is_output_layer

    # --------- Static Methods ---------
    @staticmethod
    def random_sample(width, height, start, end):
        """
        Returns a random sample of floats in the range of (start-end) exclusive
        """
        # Seed if we want to test with predictable results
        np.random.seed(60)

        num_range = end - start

        shape = []
        for x in range(width):
            shape.append(np.random.random(height) * num_range + start)

        return np.array(shape)

    @staticmethod
    def he_random_init(width, height):
        """
        Returns a random sample using the He method 
        https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        """
        # Seed if we want to test with predictable results
        np.random.seed(60)

        return np.random.randn(width, height) * np.sqrt(2/height)

    @staticmethod
    def activation_func(inputs):
        """
        Activation function for applying to the weights and biases
        """
        # ReLU function
        #return np.maximum(0, inputs)
        # Sigmoid function
        return 1/(1 + np.exp(-inputs))

    # --------- Class Methods ----------
    def forward(self, inputs):
        """
        processes input data and spits the ouputs applied through the weights, biases and activation func
        """
        # Apply weights and biases
        output = np.dot(inputs, self.weights.T) + self.biases

        # Apply activation function
        output = self.activation_func(output)

        self.activations = output

        return output

    def tune(self, weight_deltas, bias_deltas):
        """
        Edit weights and biases correspondingly
        """
        self.weights += weight_deltas
        self.biases += bias_deltas

    def calc_ll_activation_gradients(self, expected):
        """
        Calculate the activation of the layer compared to the expected values

        Intended for use ONLY on the OUTPUT LAYER
        """
        if not self.is_output_layer:
            print("Error: is not output layer, cost calculation is not needed")
            return
        # Make sure output and expected are the same size
        if len(self.activations) != len(expected):
            print("Error: output list and expected list are different sizes")
            return

        # Apply cost function
        cost_list = 2 * (self.activations - expected)

        return cost_list

    def clear_activations(self):
        self.activations = None

    def __str__(self):
        return '%s -> %s' % (self.n_inputs, self.n_nodes)
