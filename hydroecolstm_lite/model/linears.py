import torch
from torch import nn
from typing import List


class Linears(nn.Module):
    """Simple fully-connected block built from a sequence of Linear layers.

    The block is constructed from a list of neuron sizes and activation names
    and returns a `nn.Sequential` module available via `self.model`.
    """

    def __init__(self, num_layers: int, activation_function: List[str], num_neurons: List[int]):
        super(Linears, self).__init__()

        # Activation functions
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softtplus": nn.Softplus(),
            "identity": nn.Identity(),
        }

        # Create list to store different linear layers
        layers = []

        # Create layers of user-defined network
        for i in range(num_layers):
            layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))

            if i < num_layers - 1:
                layers.append(activation_functions[activation_function[i]])

        # Combined all layers together using sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        """Forward pass through the sequential fully-connected block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, ..., features) compatible with the
            first linear layer.

        Returns
        -------
        torch.Tensor
            Output tensor produced by the sequential layers.
        """
        output = self.model(x)

        return output

