"""
The canonical example of a function that can't be learned by a simple linear model
because it is not linearly separable
"""

import numpy as np

from mydeepnet.train import train
from mydeepnet.layers import Linear, Tanh
from mydeepnet.nn import NeuralNet

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets, num_epochs=10000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)