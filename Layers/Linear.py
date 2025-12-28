import numpy as np
from typing import Iterable
from Initializers import LinearInitializer


class Linear:
    def __init__(self, in_size: int, out_size: int, initialization_method: str = "he"):
        self.in_size = in_size
        self.out_size = out_size
        self.initialization_method = initialization_method
        self.initializer = LinearInitializer()
        self.weights = self.initializer.generate(in_size, out_size, initialization_method)
        self.biases = np.zeros((1, self.out_size))
        self.input = None # the input from the previous layer is stored here
        self.output = None # the output from the previous forward pass will be stored here
    

    def reset_weights(self):
        self.weights = self.initializer.generate(self.in_size, self.out_size, self.initialization_method)
        self.biases = np.zeros((1, self.out_size))


    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.matmul(input, self.weights) + self.biases
        return self.output
    

    def backward(self, propogated_grad: np.ndarray):
        self.weights_grad = np.dot(self.input.T, propogated_grad)
        self.bias_grad = np.sum(propogated_grad, axis=0, keepdims=True)
        self.input_gradient = np.matmul(propogated_grad, self.weights.T)
        return self.input_gradient
