"""
5. We use an optmizer to adjust the parameters
of our network based on the gradients computerd during
back propagation
"""

from mydeepnet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -=  self.lr * grad
