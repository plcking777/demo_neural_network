import neuralNetwork
import numpy as np


train_inp = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]).T

train_out = np.array([
    [0],
    [1],
    [1],
    [0],
]).T


nn = neuralNetwork.NeuralNetwork([2, 4, 4, 1], len(train_inp), 0.1)

for i in range(1000):
    nn.set_input(train_inp)
    nn.forward()
    cost = nn.get_cost(train_out)
    print('cost: ', cost)
    nn.backward(train_out)



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
nn.set_input(test_inp1)
nn.forward()
print('test2: ', nn.get_output())


test_inp3 = np.array([
    [0, 1]
]).T
nn.set_input(test_inp1)
nn.forward()
print('test3: ', nn.get_output())


test_inp3 = np.array([
    [1, 1]
]).T
nn.set_input(test_inp1)
nn.forward()
print('test4: ', nn.get_output())


