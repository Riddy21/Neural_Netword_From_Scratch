#!/usr/local/bin/python3
from Layer import Layer
import numpy as np

class Network(object):
    """
    Network class to do pack propagation and overall network functions
    """
    def __init__(self, layer_list):  
        """
        Initializes a Network

        Inputs:
        layer_list = Layers are recieved as a list of integers
        """
        # keep input and output numbers and layer numbers as info
        self.n_inputs = layer_list[0]
        self.inputs = None
        self.n_outputs = layer_list[-1]
        self.n_layers = layer_list

        # initialize layers
        self.layers = []

        # loop through every element of the layer tuple
        # NOTE: There will be one less layer in self.layers because the input layer doesn't count
        for layer_ind in range(1, len(self.n_layers)):
            # make that layer of the network
            if layer_ind == (len(self.n_layers) - 1):
                layer = Layer(self.n_layers[layer_ind-1], self.n_layers[layer_ind], is_output_layer=True)
            else:
                layer = Layer(self.n_layers[layer_ind-1], self.n_layers[layer_ind])
            self.layers.append(layer)

    def forward(self, inputs):
        """
        Method for forward propagating a set of data through the network
        """
        # Check the size of the input list must be same as input of the 
        if len(inputs) != self.n_inputs:
            print("Error: Input dataset list does not match input size of network")
            return None

        self.inputs = inputs

        # forward propagate the network fully
        inp = inputs
        for layer in self.layers:
            out = layer.forward(inp)
            inp = out

        # return final results
        return out

    def tune_batch(self, test, expected, learning_rate):
        """
        Calculate accumulated weights and biases gradients for mini batch and then tune network
        """
        if len(test) != len(expected):
            print("Error: batch size does not match expected results size")

        deltas = None

        # combine batch and expected values together
        batch = zip(test, expected)

        # iterate over batch
        for test, expected in batch:
            # Forward propagate data
            results = self.forward(test)
            # Back propagate data
            part_deltas = self.backward(expected)
            # Add to deltas
            if deltas is None:
                deltas = part_deltas
            else:
                deltas += part_deltas

        # Multiply deltas by learning rate
        deltas *= learning_rate

        # edit layers
        for layer, (weight_deltas, bias_deltas) in zip(self.layers, deltas):
            layer.tune(weight_deltas, bias_deltas)

    def backward(self, expected):
        """
        Method to backwards propagate the network and returns the needed deltas
        """
        if self.inputs is None:
            print("Error: did not forward propagate yet")
            return

        # List for saving network information about weight and bias deltas
        deltas = [None]*len(self.layers)

        # iterate layers starting with last layer and calculate cost gradient for each layer
        for ind, layer in reversed(list(enumerate(self.layers))):
            # --------------
            # 1. Find activation gradients of that layer
            # --------------

            # if the starting layer
            if ind == 0:
                curr_activ_grad_arr = next_activ_grad_arr
                next_layer_act_arr = self.inputs

            # if last layer, then use the cost function derivative and skip
            elif ind == len(self.layers) - 1:
                curr_activ_grad_arr = layer.calc_ll_activation_gradients(expected)
                next_layer = self.layers[ind - 1]
                next_layer_act_arr = next_layer.activations

            # if not last layer, use previous iteration's activation derivative
            else:
                curr_activ_grad_arr = next_activ_grad_arr
                next_layer = self.layers[ind - 1]
                next_layer_act_arr = next_layer.activations

            # --------------
            # 2. Iterate through each combo of nodes between l and l+1
            # --------------
            curr_layer = self.layers[ind]
            curr_layer_act_arr = curr_layer.activations

            curr_layer_n_nodes = len(curr_layer_act_arr)
            next_layer_n_nodes = len(next_layer_act_arr)

            # Initialize activation_gradient list with 0s to add
            next_activ_grad_arr = np.zeros(next_layer_n_nodes)
            # Initialize weight_gradient list as 2d list to add
            curr_weight_grad_arr = np.zeros((curr_layer_n_nodes, next_layer_n_nodes))
            # Initialize bias_gradient list as list with 0s to add to
            curr_bias_grad_arr = np.zeros(curr_layer_n_nodes)

            # Iterate l-1 nodes as k
            for k in range(next_layer_n_nodes):
                # iterate l nodes j
                for j in range(curr_layer_n_nodes):
                    # ---------------
                    # 3. Find wieght gradient
                    # ---------------
                    # multiply activation gradient corresponding by ak*aj*(1-aj)
                    # where ak is the current l node and aj is the current l+1 node
                    weight_grad = curr_activ_grad_arr[j] * next_layer_act_arr[k] * curr_layer_act_arr[j] * (1 - curr_layer_act_arr[j])
                    # set weights of that connection to weight gradient list
                    curr_weight_grad_arr[j][k] = weight_grad
                    # ---------------
                    # 4. Find partial bias gradient
                    # ---------------
                    # multiply activation gradient corresping by ak
                    part_bias_grad = curr_activ_grad_arr[j] * curr_layer_act_arr[j] * (1 - curr_layer_act_arr[j])
                    # add partial bias grad to bias grad of that node
                    curr_bias_grad_arr[j] = part_bias_grad
                    # ---------------
                    # 5. Find partial activation gradient
                    # ---------------
                    # multiply activ gradient corresponding by w_jk*aj*(1-aj) where w_jk is the current weight connecting ak and aj
                    w_jk = curr_layer.weights[j][k]
                    part_activ_grad = curr_activ_grad_arr[j] * w_jk * curr_layer_act_arr[j] * (1 - curr_layer_act_arr[j])
                    # add partial activation grad to activation grad of that node
                    next_activ_grad_arr[k] += part_activ_grad
            # -----------
            # 6. Calculate desired change coefficient for weights
            # -----------
            weight_deltas = - curr_weight_grad_arr
            # Multiply each weight connected to each node by the activation
            for j in range(len(weight_deltas)):
                # multiply corresponding inputs values the connections
                for k in range(len(weight_deltas[j])):
                    weight_deltas[j][k] *= next_layer_act_arr[k]

            # ----------
            # 7. Calculate desired change coefficient for biases
            # ----------
            bias_deltas = - curr_bias_grad_arr

            # Save the cost and weight gradients dict
            deltas[ind] = (weight_deltas, bias_deltas)

        # Reset inputs so you can see there isn't a forward prop yet
        self.inputs = None

        return np.array(deltas)


    def __str__(self):
        """
        Easy print for debug
        """
        string = ''
        for layer in self.layers:
            string += "%s \n" % layer
        return string

