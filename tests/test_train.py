"""Exercise learning, clean gradients, evaluation and checkpoint restoration."""

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from model import RentalsNetwork
from prepare import make_loaders
from train import fit_network, run_epoch


def test_validation_does_not_change_weights_or_compute_gradients(prepared):
    model = RentalsNetwork(8)
    before = copy.deepcopy(model.state_dict())
    seen_grad_modes = []
    hook = model.register_forward_hook(
        lambda module, inputs, output: seen_grad_modes.append(torch.is_grad_enabled())
    )
    loader = make_loaders(prepared)[1]
    result = run_epoch(model, loader)
    hook.remove()
    assert result["loss"] >= 0 and result["mae"] >= 0
    assert not model.training
    assert seen_grad_modes and not any(seen_grad_modes)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_loss_and_mae_are_weighted_by_rows_not_batches():
    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1), torch.tensor([[0.0], [0.0], [9.0]])),
        batch_size=2,
    )
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    assert run_epoch(model, loader) == {"loss": 27.0, "mae": 3.0}


def test_epoch_clears_gradients_before_every_step_and_updates_weights():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 2 * x + 1
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    actual = nn.Linear(1, 1)
    expected = copy.deepcopy(actual)
    # Pre-existing gradients must also be cleared before the first batch.
    for parameter in actual.parameters():
        parameter.grad = torch.full_like(parameter, 500)
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
    run_epoch(actual, loader, optimizer)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
    for features, targets in loader:
        reference_optimizer.zero_grad()
        nn.functional.mse_loss(expected(features), targets).backward()
        reference_optimizer.step()
    for found, wanted in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(found, wanted)


def test_network_memorizes_a_small_real_batch(prepared):
    features = prepared.train_dataset.features[:16]
    targets = prepared.train_dataset.targets[:16]
    loader = DataLoader(TensorDataset(features, targets), batch_size=16)
    model = RentalsNetwork(8, (64, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = run_epoch(model, loader)["loss"]
    for _ in range(500):
        run_epoch(model, loader, optimizer)
    final = run_epoch(model, loader)["loss"]
    assert final < 0.01 * initial
    assert final < 0.5


def test_history_records_every_epoch_and_restores_lowest_validation_loss(monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    validation_losses = iter([3.0, 1.0, 4.0])
    epoch = 0

    def controlled_epoch(network, loader, optimizer=None):
        nonlocal epoch
        if optimizer is not None:
            epoch += 1
            with torch.no_grad():
                network.weight.fill_(epoch)
                network.bias.fill_(epoch)
            return {"loss": 10.0 / epoch, "mae": 2.0}
        return {"loss": next(validation_losses), "mae": 1.0}

    monkeypatch.setattr(train, "run_epoch", controlled_epoch)
    history, best_epoch = fit_network(model, [], [], optimizer, epochs=3)
    assert list(history["epoch"]) == [1, 2, 3]
    assert set(history) == {
        "epoch",
        "train_loss",
        "train_mae",
        "validation_loss",
        "validation_mae",
    }
    assert best_epoch == 2
    assert history["validation_loss"].tolist() == [3, 1, 4]
    for value in model.state_dict().values():
        assert torch.all(value == 2)


def test_real_fit_logs_finite_metrics_and_restores_recorded_best(prepared):
    train_loader, validation_loader, _ = make_loaders(prepared, batch_size=32)
    model = RentalsNetwork(8, (16, 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history, best = fit_network(model, train_loader, validation_loader, optimizer, 8)
    assert len(history) == 8
    assert np.isfinite(history.to_numpy()).all()
    assert best == int(history.loc[history["validation_loss"].idxmin(), "epoch"])
    restored = run_epoch(model, validation_loader)
    assert restored["loss"] == pytest.approx(history["validation_loss"].min())


def test_target_shape_mismatch_is_rejected():
    loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4)), batch_size=4)
    with pytest.raises(ValueError, match="shape"):
        run_epoch(nn.Linear(1, 1), loader)
