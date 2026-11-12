"""A fully connected regression network with a linear output."""

from collections.abc import Sequence

from torch import nn


class RentalsNetwork(nn.Module):
    """Learn nonlinear demand with at least two ReLU hidden layers."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (64, 32)):
        super().__init__()
        if input_size < 1 or len(hidden_sizes) < 2 or min(hidden_sizes) < 1:
            raise ValueError("Use positive widths and at least two hidden layers.")
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)
