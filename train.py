"""Train with an explicit PyTorch loop and reproduce the committed reports."""

import copy
import json
import random
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader

from bikes_data import TARGET, make_rentals
from model import RentalsNetwork
from prepare import SEED, PreparedData, make_loaders, prepare_data

REPORTS = Path(__file__).resolve().parent / "reports"


def seed_everything(seed: int = SEED) -> None:
    """Keep CPU training repeatable, including batches and initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Train if an optimizer is supplied; otherwise measure without gradients."""
    training = optimizer is not None
    model.train(training)
    total_loss = total_absolute_error = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for features, targets in loader:
            if training:
                optimizer.zero_grad()
            predictions = model(features)
            if predictions.shape != targets.shape or targets.ndim != 2:
                raise ValueError("Predictions and targets must both have shape (N, 1).")
            loss = nn.functional.mse_loss(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
            rows = len(targets)
            total_loss += loss.item() * rows
            total_absolute_error += (predictions - targets).abs().sum().item()
            count += rows
    if count == 0:
        raise ValueError("Cannot measure an empty loader.")
    return {"loss": total_loss / count, "mae": total_absolute_error / count}


def fit_network(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> tuple[pd.DataFrame, int]:
    """Log every epoch and restore an independent copy of the best weights."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    history = []
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        training = run_epoch(model, train_loader, optimizer)
        validation = run_epoch(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training["loss"],
                "train_mae": training["mae"],
                "validation_loss": validation["loss"],
                "validation_mae": validation["mae"],
            }
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}: train MSE={training['loss']:.3f}, "
                f"validation MSE={validation['loss']:.3f}, "
                f"validation MAE={validation['mae']:.3f}",
                flush=True,
            )
    model.load_state_dict(best_weights)
    model.eval()
    return pd.DataFrame(history), best_epoch


@dataclass(frozen=True)
class RunConfig:
    name: str
    learning_rate: float = 0.003
    hidden_sizes: tuple[int, ...] = (64, 32)
    batch_size: int = 64
    epochs: int = 250


def plot_curves(history: pd.DataFrame, best_epoch: int, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Training")
    ax.plot(history["epoch"], history["validation_loss"], label="Validation")
    ax.axvline(
        best_epoch, color="black", linestyle="--", label=f"Best epoch {best_epoch}"
    )
    ax.set(xlabel="Epoch", ylabel="MSE (rentals squared)", title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_demand(training: pd.DataFrame, path: Path):
    """Explore training rows only, leaving the held-out rows untouched."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for weekend, ax in enumerate(axes):
        subset = training[training["is_weekend"] == weekend]
        for station in ["downtown", "residential", "park"]:
            means = (
                subset[subset["station_type"] == station].groupby("hour")[TARGET].mean()
            )
            ax.plot(means.index, means.values, marker=".", label=station)
        ax.set(
            xlabel="Hour",
            title="Weekend" if weekend else "Weekday",
            xticks=range(0, 24, 3),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean rentals per station-hour")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_baselines(data: PreparedData):
    x = data.train_dataset.features.numpy()
    y = data.train[TARGET].to_numpy()
    models = {
        "Training mean": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
    }
    validation_scores = {}
    for name, model in models.items():
        model.fit(x, y)
        prediction = model.predict(data.validation_dataset.features.numpy())
        validation_scores[name] = float(
            mean_absolute_error(data.validation[TARGET], prediction)
        )
    return models, validation_scores


def main() -> None:
    seed_everything()
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data(make_rentals())
    print(data.train.head().to_string(index=False), flush=True)
    plot_demand(data.train, REPORTS / "rentals_by_hour.png")
    baselines, baseline_validation = fit_baselines(data)
    # Each experiment changes only learning rate; all other settings stay fixed.
    configs = [
        RunConfig("main"),
        RunConfig("slow_learning", learning_rate=0.00001),
        RunConfig("fast_learning", learning_rate=0.1),
    ]
    records = []
    trained = {}
    for config in configs:
        print(f"\nRun: {config.name}", flush=True)
        seed_everything()
        train_loader, validation_loader, _ = make_loaders(data, config.batch_size)
        model = RentalsNetwork(
            data.train_dataset.features.shape[1], config.hidden_sizes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        history, best_epoch = fit_network(
            model, train_loader, validation_loader, optimizer, config.epochs
        )
        main_run = config.name == "main"
        history_name = "history.csv" if main_run else f"history_{config.name}.csv"
        curve_name = "learning_curves.png" if main_run else f"curves_{config.name}.png"
        history.to_csv(REPORTS / history_name, index=False)
        plot_curves(history, best_epoch, REPORTS / curve_name, config.name)
        restored = run_epoch(model, validation_loader)
        records.append(
            {
                **asdict(config),
                "best_epoch": best_epoch,
                "best_validation_loss": restored["loss"],
                "validation_mae_at_best_loss": restored["mae"],
                "minimum_validation_mae": float(history["validation_mae"].min()),
                "final_train_loss": float(history.iloc[-1]["train_loss"]),
                "final_validation_loss": float(history.iloc[-1]["validation_loss"]),
            }
        )
        trained[config.name] = model
    experiments = pd.DataFrame(records)
    # Select using validation MSE before inspecting any test predictions.
    chosen = min(records, key=lambda row: row["best_validation_loss"])
    selected_name = chosen["name"]
    experiments["selected"] = experiments["name"] == selected_name
    experiments.to_csv(REPORTS / "experiments.csv", index=False)
    _, _, test_loader = make_loaders(data)
    test_metrics = run_epoch(trained[selected_name], test_loader)
    results = {
        name: {
            "validation_mae": baseline_validation[name],
            "test_mae": float(
                mean_absolute_error(
                    data.test[TARGET], model.predict(data.test_dataset.features.numpy())
                )
            ),
        }
        for name, model in baselines.items()
    }
    results["Neural network"] = {
        "validation_mae": chosen["validation_mae_at_best_loss"],
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["loss"],
    }
    metrics = {
        "seed": SEED,
        "split_sizes": {
            "train": len(data.train),
            "validation": len(data.validation),
            "test": len(data.test),
        },
        "selected_run": selected_name,
        "selection_rule": "Lowest validation MSE at the restored best epoch",
        "selected_best_epoch": chosen["best_epoch"],
        "feature_names": data.preprocessor.get_feature_names_out().tolist(),
        "training_target_mean": float(data.train[TARGET].mean()),
        "results": results,
        "versions": {
            name: version(name)
            for name in ["torch", "numpy", "pandas", "scikit-learn", "matplotlib"]
        },
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
