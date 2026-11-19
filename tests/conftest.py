# Shared set-up for the tests. You don't need to change this file.
# Each test that asks for homes gets its own fresh copy of the 700 homes in home_energy.csv, read from this
# exercise's folder, so one test can't change the table another test reads.
from pathlib import Path

import pandas as pd
import pytest

HOME_ENERGY = Path(__file__).parent.parent / "home_energy.csv"


@pytest.fixture
def homes():
    """The 700 homes in home_energy.csv, freshly read for one test."""
    return pd.read_csv(HOME_ENERGY)
