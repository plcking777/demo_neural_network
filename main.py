import numpy as np
import neuralNetwork

nn = neuralNetwork.NeuralNetwork([6, 10, 6], 1, 0.1)

for _ in range(1000):
    nn.set_input(
        np.array(
            [[0], [1], [0], [1], [0], [1]]
        )
    )

    nn.forward()

    print('Output: ', nn.get_output())

    nn.backward(
        np.array(
            [[1], [0], [1], [0], [1], [0]]
        )
    )
