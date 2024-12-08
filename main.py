import numpy as np
import neuralNetwork

nn = neuralNetwork.NeuralNetwork([1, 50, 1], 1, 0.0001)

for _ in range(500000):
    # set input
    nn.set_input(0)

    nn.forward()

    print('Output: ', nn.get_output())

    nn.backward(np.array([[1]]))
