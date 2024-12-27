import neuralNetwork
import numpy as np


train_inp_1 = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]).T

train_out_1 = np.array([
    [0],
    [1],
    [1],
    [0],
]).T




nn = neuralNetwork.NeuralNetwork([2, 5, 5, 1], 2, 0.1)

for i in range(10000):
    nn.set_input(train_inp_1)
    nn.forward()
    cost = nn.get_cost(train_out_1)
    print('cost: ', cost)
    nn.backward(train_out_1)


# TESTS
print('--- TESTING ---')
test_inp1 = np.array([
    [0, 0]
]).T
nn.set_input(test_inp1)
nn.forward()
print('test1: ', nn.get_output())

test_inp2 = np.array([
    [1, 0]
]).T
nn.set_input(test_inp2)
nn.forward()
print('test2: ', nn.get_output())


test_inp3 = np.array([
    [0, 1]
]).T
nn.set_input(test_inp3)
nn.forward()
print('test3: ', nn.get_output())


test_inp4 = np.array([
    [1, 1]
]).T
nn.set_input(test_inp4)
nn.forward()
print('test4: ', nn.get_output())
