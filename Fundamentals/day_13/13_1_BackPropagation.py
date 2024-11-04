import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward_propagation(self, x):
        self.hidden = sigmoid(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward_propagation(self, x, y, lr):
        output_error = self.output - y
        output_gradient = output_error * sigmoid_derivative(self.output)

        hidden_error = np.dot(output_gradient, self.weights2.T)
        hidden_gradient = hidden_error * sigmoid_derivative(self.hidden)

        self.weights2 += np.dot(self.hidden.T, hidden_gradient)*lr
        self.weights1 += np.dot(x.T, hidden_gradient)*lr

    def train(self, x, y, epochs, lr):
        for epoch in range(epochs):
            output = self.forward_propagation(x)
            self.backward_propagation(x, y, lr)
            if epoch % 10 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch:{epoch}, Loss:{loss}")





