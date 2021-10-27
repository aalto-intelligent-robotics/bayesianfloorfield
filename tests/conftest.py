import pytest
from unittest import mock

from PIL import Image
from deepflow.data import DiscreteDirectionalDataset
from mod.OccupancyMap import OccupancyMap
from mod.Grid import Grid


@pytest.fixture
def dataset():
    occupancy = mock.MagicMock(spec=OccupancyMap)
    grid = mock.MagicMock(spec=Grid)
    occupancy.configure_mock(**{"binary_map": Image.new("L", (2, 2))})
    grid.configure_mock(**{"cells": {(0, 0): "", (0, 1): ""}})
    return DiscreteDirectionalDataset(
        occupancy=occupancy, dynamics=grid, window_size=2
    )
