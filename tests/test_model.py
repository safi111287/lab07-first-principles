"""Check output dimensions, hidden layers and parameter updates."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import RentalsNetwork
from train import run_epoch


@pytest.mark.parametrize("hidden_sizes", [(8, 4), (32, 16), (16, 8, 4)])
def test_network_shape_for_several_widths(hidden_sizes):
    model = RentalsNetwork(8, hidden_sizes)
    output = model(torch.randn(7, 8))
    assert output.shape == (7, 1)
    assert torch.isfinite(output).all()
    assert sum(isinstance(layer, nn.ReLU) for layer in model.layers) == len(
        hidden_sizes
    )
    assert isinstance(model.layers[-1], nn.Linear)


def test_one_optimizer_step_changes_every_layer(prepared):
    model = RentalsNetwork(8, (16, 8))
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(prepared.train_dataset, batch_size=len(prepared.train_dataset))
    run_epoch(model, loader, optimizer)
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name


def test_output_is_not_constrained_to_a_probability():
    model = RentalsNetwork(8, (8, 4))
    with torch.no_grad():
        model.layers[-1].weight.zero_()
        model.layers[-1].bias.fill_(-2)
    torch.testing.assert_close(model(torch.zeros(3, 8)), torch.full((3, 1), -2.0))
