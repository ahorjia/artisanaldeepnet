"""
4. A collection of layers
"""

from typing import Sequence, Iterator, Tuple

from mydeepnet.tensor import Tensor
from mydeepnet.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grads: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
