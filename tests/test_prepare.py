"""Protect row separation, train-only preparation and seeded batches."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from bikes_data import make_rentals
from prepare import NUMERIC_FEATURES, make_loaders, prepare_data


def test_given_data_is_repeatable():
    pd.testing.assert_frame_equal(make_rentals(), make_rentals())
    assert len(make_rentals()) == 6000


def test_dataset_length_shapes_and_dtypes(prepared):
    dataset = prepared.train_dataset
    features, target = dataset[0]
    assert len(dataset) == 180
    assert features.shape == (8,)
    assert target.shape == (1,)
    assert features.dtype == target.dtype == torch.float32
    torch.testing.assert_close(
        target, torch.tensor([prepared.train.iloc[0]["rentals"]], dtype=torch.float32)
    )


def test_split_sizes_disjoint_and_repeatable():
    frame = make_rentals()
    first = prepare_data(frame)
    second = prepare_data(frame)
    splits = [first.train, first.validation, first.test]
    assert [len(part) for part in splits] == [3600, 1200, 1200]
    indices = [set(part.index) for part in splits]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert set.union(*indices) == set(frame.index)
    for name in ["train", "validation", "test"]:
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))


def test_scaler_and_encoder_fit_training_only(prepared):
    scaler = prepared.preprocessor.named_transformers_["numeric"]
    np.testing.assert_allclose(scaler.mean_, prepared.train[NUMERIC_FEATURES].mean())
    assert scaler.n_samples_seen_ == len(prepared.train)
    train = prepared.train_dataset.features.numpy()[:, :5]
    validation = prepared.validation_dataset.features.numpy()[:, :5]
    np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-6)
    assert not np.allclose(validation.mean(axis=0), 0, atol=1e-3)
    encoder = prepared.preprocessor.named_transformers_["categorical"]
    assert set(encoder.categories_[0]) == set(prepared.train["station_type"])


def test_held_out_values_cannot_change_fitted_preparation():
    frame = make_rentals(n=300)
    first = prepare_data(frame)
    changed = frame.copy()
    held_out = list(first.validation.index) + list(first.test.index)
    changed.loc[held_out, "temperature_c"] = 10000
    changed.loc[held_out, "station_type"] = "unseen"
    second = prepare_data(changed)
    torch.testing.assert_close(
        first.train_dataset.features, second.train_dataset.features
    )
    encoder = second.preprocessor.named_transformers_["categorical"]
    assert "unseen" not in encoder.categories_[0]


def test_only_training_loader_shuffles_with_repeatable_seed(prepared):
    first = make_loaders(prepared, batch_size=17, seed=7)
    second = make_loaders(prepared, batch_size=17, seed=7)
    different = make_loaders(prepared, batch_size=17, seed=8)
    assert isinstance(first[0].sampler, RandomSampler)
    assert isinstance(first[1].sampler, SequentialSampler)
    assert isinstance(first[2].sampler, SequentialSampler)
    order = list(first[0].sampler)
    assert order == list(second[0].sampler)
    assert order != list(range(len(prepared.train_dataset)))
    assert order != list(different[0].sampler)
    for loader, dataset in zip(
        first[1:], [prepared.validation_dataset, prepared.test_dataset], strict=True
    ):
        actual = torch.cat([features for features, _ in loader])
        torch.testing.assert_close(actual, dataset.features)
