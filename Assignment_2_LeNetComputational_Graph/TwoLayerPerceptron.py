import numpy as np

class Perceptron:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        a = self.sigmoid(z)
        return a

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def backward(self, x, y, a):
        dz = (a - y) * self.sigmoid_derivative(a)
        dw = np.dot(dz, x.T)
        db = dz
        dx = np.dot(self.weights.T, dz)
        return dx, dw, db

class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.perceptron1 = Perceptron(input_size, hidden_size)
        self.perceptron2 = Perceptron(hidden_size, output_size)

    def forward(self, x):
        a1 = self.perceptron1.forward(x)
        a2 = self.perceptron2.forward(a1)
        return a2

    def backward(self, x, y, a):
        da2, dw2, db2 = self.perceptron2.backward(a1, y, a)
        da1, dw1, db1 = self.perceptron1.backward(x, a1, da2)
        return dw1, db1, dw2, db2
