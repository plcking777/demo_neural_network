import neuralNetwork
import numpy as np


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
]).T

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
]).T


nn = neuralNetwork.NeuralNetwork([2, 20, 20, 1], len(train_data_inp), 0.01)

for i in range(10000):
    nn.set_input(train_data_inp)
    nn.forward()
    cost = nn.get_cost(train_data_out)
    print('cost: ', cost)
    nn.backward(train_data_out)


# TEST
nn.set_input(train_data_inp)
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
