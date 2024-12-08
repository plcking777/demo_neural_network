import numpy as np



class NeuralNetwork:
    Ws = []
    Bs = []
    As = []
    Zs = []
    def __init__(self, layers, epoch, learning_rate):
        self.layers = layers
        self.epoch = epoch
        self.learning_rate = learning_rate

        for i in range(len(layers) - 1):
            nodes = layers[i]
            next_nodes = layers[i + 1]

            self.Ws.append(np.random.rand(next_nodes, nodes))
            self.Bs.append(np.random.rand(next_nodes, 1))
        self.As = [np.array([[]]) for _ in range(len(layers))]
        self.Zs = [np.array([[]]) for _ in range(len(layers) - 1)]

    def forward(self):
        prevAct = self.As[0]

        for i in range(len(self.Ws)):
            Z = self.Ws[i].dot(prevAct) + self.Bs[i]

            if i == len(self.Ws) - 1:
                prevAct = map_lineaire(Z)
            else:
                prevAct = map_relu(Z)

            self.Zs[i] = Z
            self.As[i + 1] = prevAct


    def backward(self, target):
        WsDerivs = [np.array([[]]) for _ in range(len(self.Ws))]
        BsDerivs = [np.array([[]]) for _ in range(len(self.Bs))]

        zDeriv = self.costDerivative(target)

        WsDerivs[len(WsDerivs) - 1] = zDeriv.dot(self.As[len(self.As) - 2].T) / self.epoch
        BsDerivs[len(BsDerivs) - 1] = zDeriv / self.epoch


        for i in range(len(self.Ws)):
            zDeriv = self.Ws[len(self.Ws) - 1 - i].T.dot(zDeriv) * map_drelu(self.Zs[len(self.Zs) - 2 - i])

            WsDerivs[len(WsDerivs) - 2 - i] = zDeriv.dot(self.As[len(self.As) - 3 - i].T) * 1.00 / self.epoch
            BsDerivs[len(WsDerivs) - 2 - i] = zDeriv / self.epoch

        self.update_weights(WsDerivs, BsDerivs)


    def update_weights(self, WsDerivs, BsDerivs):
        '''
        print(' --- UPDATE --- ')
        print(WsDerivs)
        for i in range(len(self.Ws)):
            print(f'    --- L{i} --- ')
            print(
                '   Ws: \n  ', WsDerivs[i], '\n',
            )
        print(' --- UPDATE ---  \n\n\n\n')
        '''

        for i in range(len(self.Ws)):
            self.Ws[i] = self.Ws[i] - (WsDerivs[i] * self.learning_rate)
            self.Bs[i] = self.Bs[i] - (BsDerivs[i] * self.learning_rate)

    def costDerivative(self, target):
        return (self.As[len(self.As) - 1] - target) * 2.00

    def set_input(self, x):
        self.As[0] = np.array([[x]])

    def get_output(self):
        return self.As[len(self.As) - 1]

    def print_state(self):

        for i in range(len(self.Ws)):
            print(f'    --- L{i} --- ')
            print(
                '   Ws: \n  ', self.Ws[i], '\n\n',
                '   Bs: \n  ', self.Bs[i], '\n\n',
                '   As: \n  ', self.As[i + 1], '\n\n',
                '   Zs: \n  ', self.Zs[i], '\n\n',
            )

def map_lineaire(xs):
    return xs

def map_relu(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            xs[i][j] = max(0.00, xs[i][j])
    return xs

def map_drelu(xs):
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            if xs[i][j] <= 0.00:
                xs[i][j] = 0.00
            else:
                xs[i][j] = 1
    return xs
