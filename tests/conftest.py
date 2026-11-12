"""Small deterministic data shared by the tests."""

import pytest

from bikes_data import make_rentals
from prepare import prepare_data
from train import seed_everything


@pytest.fixture(autouse=True)
def fixed_seed():
    seed_everything()


@pytest.fixture
def prepared():
    return prepare_data(make_rentals(n=300))
