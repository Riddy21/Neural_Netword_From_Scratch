#!/usr/local/bin/python3

from Network import Network

# Making network with network object
network = Network([4, 3, 3, 2])
batch = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
expected = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
network.tune_batch(batch, expected, 0.5)
