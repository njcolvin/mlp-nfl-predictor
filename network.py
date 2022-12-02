from loss import Loss, MSE
import numpy as np
from layer import Layer
from initialization import Initialization, GaussianInitialization, UniformInitialization
from activation import Activation, SigmoidActivation, TanhActivation, ReLUActivation

class Network:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200,
    initialization:Initialization=GaussianInitialization(), activation:Activation=ReLUActivation(),
    loss:Loss=MSE(), learning_rate:float=0.001):
        assert max_iter > 0
        assert learning_rate > 0
        self.layers = []
        self.sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.initialization = initialization
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate

    # predict output for given input
    def predict(self, samples):
        # sample dimension first
        n = len(samples)
        result = []

        # run network over all samples
        for i in range(n):
            # forward propagation
            output = samples[i]
            for layer in self.layers:
                output = layer.forward_propagate(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train):
        # sample dimension first
        n = len(x_train)
        assert n > 0
        assert len(y_train) == n

        num_layers = len(self.sizes)
        assert num_layers > 0

        # build network
        # input layer
        print(np.shape(x_train))
        print(np.shape(y_train))
        self.initialization.set_layer_size(len(x_train[0][0]), self.sizes[0])
        self.layers.append(Layer(self.initialization))
        # hidden layer(s)
        for i in range(1, num_layers):
            self.initialization.set_layer_size(self.sizes[i - 1], self.sizes[i])
            self.layers.append(Layer(self.initialization))
        # output layer
        self.initialization.set_layer_size(self.sizes[num_layers - 1], len(y_train[0][3]))
        self.layers.append(Layer(self.initialization))
        print(len(self.layers))
        print(self.layers[len(self.layers) - 1].weights)

        # training loop
        for i in range(self.max_iter):
            err = 0
            for j in range(n):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # compute loss (for display purpose only)
                err += self.loss.forward_propagate(output, y_train[j])

                # backward propagation
                de_dy = self.loss.back_propagate(output, y_train[j])
                for layer in reversed(self.layers):
                    de_dy = layer.back_propagate(de_dy, self.learning_rate)

            # calculate average error on all samples
            err /= n
            print('iter %d/%d   error=%f' % (i+1, self.max_iter, err))


def main():
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[0], [1], [1], [0]])

    # network
    net = Network(hidden_layer_sizes=(3,), max_iter=2000, learning_rate=0.05,
                    initialization=UniformInitialization())
    # train
    net.fit(x_train, y_train)

    # test
    out = net.predict(x_train)
    print(out)

main()