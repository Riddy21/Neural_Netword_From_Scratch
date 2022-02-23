#!/usr/local/bin/python3
import numpy as np
from Data import Data
from Network import Network

# Get data
data = Data()
inp = data.x_train_flattened
results = Data.encode_results(data.y_train)

train_inputs = inp[:55000]
train_results = results[:55000]

test_inputs = inp[55001:55050]
test_results = results[55001:55050]

# Making network with network object
network = Network([len(inp[0]), 16, 16, 10])

# Split test data into batches
inp_batches = list(Data.split(train_inputs, 10))
res_batches = list(Data.split(train_results, 10))

for ind, (inp_batch, res_batch) in enumerate(zip(inp_batches, res_batches)):
    network.tune_batch(inp_batch, res_batch, 0.005)
    print(ind)

print("------------DONE!!!!!---------")
correct_count = 0
for i in range(len(test_inputs)):
    print('\n\n------------------')
    print('Data:')
    data.print_data('train', i + 55001)
    output = network.forward(test_inputs[i])
    output_ind = np.argmax(output)
    expected_ind = np.argmax(test_results[i])
    print('Results: %s' % output_ind)
    print('Expected: %s' % expected_ind)
    if output_ind == expected_ind:
        correct_count += 1

print('Accuracy: %s' % (correct_count/len(test_inputs)))


