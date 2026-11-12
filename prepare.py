"""Split first, fit preparation on training rows, and build tensor loaders."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET

SEED = 2026
NUMERIC_FEATURES = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL_FEATURES = ["station_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class RentalsDataset(Dataset):
    """Return a feature vector and a one-element target, both float32."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32).reshape(
            -1, 1
        )
        if self.features.ndim != 2 or len(self.features) != len(self.targets):
            raise ValueError("Features must be a matrix with one target per row.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass
class PreparedData:
    """Keep raw splits for inspection and fitted preparation for reuse."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    preprocessor: ColumnTransformer
    train_dataset: RentalsDataset
    validation_dataset: RentalsDataset
    test_dataset: RentalsDataset


def prepare_data(frame: pd.DataFrame, seed: int = SEED) -> PreparedData:
    """Make disjoint 60/20/20 splits before fitting any transformation."""
    train, remainder = train_test_split(frame, test_size=0.4, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    x_train = preprocessor.fit_transform(train[FEATURES])
    x_validation = preprocessor.transform(validation[FEATURES])
    x_test = preprocessor.transform(test[FEATURES])
    return PreparedData(
        train,
        validation,
        test,
        preprocessor,
        RentalsDataset(x_train, train[TARGET].to_numpy()),
        RentalsDataset(x_validation, validation[TARGET].to_numpy()),
        RentalsDataset(x_test, test[TARGET].to_numpy()),
    )


def make_loaders(
    data: PreparedData, batch_size: int = 64, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle training only, with an independent seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            data.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        DataLoader(data.validation_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(data.test_dataset, batch_size=batch_size, shuffle=False),
    )
