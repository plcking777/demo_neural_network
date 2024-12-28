import neuralNetwork
import numpy as np
import random


train_data_inp = np.array([
    [0, 0],
    [0, 0.2],
    [0, 0.4],
    [0, 0.6],
    [0, 0.8],
    [0, 1],
    [0.2, 0],
    [0.2, 0.2],
    [0.2, 0.4],
    [0.2, 0.6],
    [0.2, 0.8],
    [0.4, 0],
    [0.4, 0.2],
    [0.4, 0.4],
    [0.4, 0.6],
    [0.4, 0.8],
    [0.6, 0],
    [0.6, 0.2],
    [0.6, 0.4],
    [0.6, 0.6],
])

train_data_out = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
])


BATCH_SIZE = 8

nn = neuralNetwork.NeuralNetwork([2, 15, 15, 1], len(train_data_inp), BATCH_SIZE, 0.1)

for i in range(100000):

    # BATCH - START
    batch_input = []
    batch_output = []
    for b in range(BATCH_SIZE):
        index = random.randint(0, len(train_data_inp) - 1)
        batch_input.append(train_data_inp[index])
        batch_output.append(train_data_out[index])

    batch_input = np.array(batch_input).T
    batch_output = np.array(batch_output).T

    # BATCH - STOP


    nn.set_input(batch_input)
    nn.forward()
    cost = nn.get_cost(batch_output)
    print('cost: ', cost)
    nn.backward(batch_output)








# TEST
print(' --- TEST --- ')
nn.set_input(train_data_inp.T)
nn.forward()

print('out: ', nn.get_output())




train_data_inp = np.array([[0.00, 0.00]]).T
nn.set_input(train_data_inp)
nn.forward()

print('should be 0', nn.get_output())


train_data_inp = np.array([[0.00, 0.20]]).T
nn.set_input(train_data_inp)
nn.forward()

print('should be 1', nn.get_output())
