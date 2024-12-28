import numpy as np
import math


class NeuralNetwork:
    Ws = []
    Bs = []
    As = []
    Zs = []
    def __init__(self, layers, epoch, batch_size, learning_rate):
        self.layers = layers
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        for i in range(len(layers) - 1):
            nodes = layers[i]
            next_nodes = layers[i + 1]

            self.Ws.append(np.random.rand(next_nodes, nodes) - 0.5)
            self.Bs.append(np.random.rand(next_nodes, 1) - 0.5)
            #self.Ws.append(np.random.randn(next_nodes, nodes) * np.sqrt(2 / next_nodes))
            #self.Bs.append(np.random.randn(next_nodes, 1) * np.sqrt(2 / next_nodes))
        self.As = [np.array([[]]) for _ in range(len(layers))]
        self.Zs = [np.array([[]]) for _ in range(len(layers) - 1)]

    def forward(self):
        prevAct = self.As[0]
        for i in range(len(self.Ws)):
            Z = self.Ws[i].dot(prevAct) + self.Bs[i]

            if i == len(self.Ws) - 1:
                prevAct = map_sigmoid(Z)
            else:
                prevAct = map_relu(Z)

            self.Zs[i] = Z
            self.As[i + 1] = prevAct


    def backward(self, target):
        WsDerivs = [np.array([[]]) for _ in range(len(self.Ws))]
        BsDerivs = [np.array([[]]) for _ in range(len(self.Bs))]

        zDeriv = self.cost_derivative(target) * map_dsigmoid(self.Zs[len(self.Zs) - 1])

        WsDerivs[len(WsDerivs) - 1] = zDeriv.dot(self.As[len(self.As) - 2].T) / self.epoch
        BsDerivs[len(BsDerivs) - 1] = np.sum(zDeriv, axis=1) / self.epoch

        for i in range(len(self.Ws) - 1):
            zDeriv = self.Ws[len(self.Ws) - 1 - i].T.dot(zDeriv) * map_drelu(self.Zs[len(self.Zs) - 2 - i])

            #WsDerivs[len(WsDerivs) - 2 - i] = zDeriv.dot(self.As[len(self.As) - 3 - i].T) / self.epoch
            #BsDerivs[len(WsDerivs) - 2 - i] = np.sum(zDeriv) / self.epoch
            WsDerivs[len(WsDerivs) - 2 - i] = np.clip(zDeriv.dot(self.As[len(self.As) - 3 - i].T) / self.epoch, -1, 1)
            BsDerivs[len(WsDerivs) - 2 - i] = np.clip(np.sum(zDeriv) / self.epoch, -1, 1)

        self.update_weights(WsDerivs, BsDerivs)


    def update_weights(self, WsDerivs, BsDerivs):
        for i in range(len(self.Ws)):
            self.Ws[i] = self.Ws[i] + (WsDerivs[i] * self.learning_rate)
            self.Bs[i] = self.Bs[i] + (BsDerivs[i] * self.learning_rate)

    def cost_derivative(self, target):
        return (self.As[len(self.As) - 1] - target) * 2.00

    def get_cost(self, target):
        return np.sum((self.As[len(self.As) - 1] - target) ** 2) / self.epoch

    def set_input(self, x):
        self.As[0] = x

    def get_output(self):
        return self.As[len(self.As) - 1]

    def print_state(self):
        for i in range(len(self.Ws)):
            print(f'    --- L{i} --- ')
            print(
                '   Ws: \n  ', self.Ws[i].shape, '\n\n',
                '   Bs: \n  ', self.Bs[i].shape, '\n\n',
                '   As: \n  ', self.As[i + 1].shape, '\n\n',
                '   Zs: \n  ', self.Zs[i].shape, '\n\n',
            )

def map_lineaire(xs):
    return xs

def map_relu(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            #xs[i][j] = max(0.00, xs[i][j])
            xs[i][j] = max(0.01 * xs[i][j], xs[i][j])
    return xs

def map_drelu(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            if xs[i][j] <= 0.00:
                xs[i][j] = 0.00
            else:
                xs[i][j] = 1
    return xs


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def map_sigmoid(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            xs[i][j] = sigmoid(-xs[i][j])

    return xs

def map_dsigmoid(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            xs[i][j] = sigmoid(xs[i][j]) * (1 - sigmoid(xs[i][j]))

    return xs
