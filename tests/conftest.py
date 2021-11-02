import pytest
from unittest import mock

from PIL import Image
from deepflow.data import DiscreteDirectionalDataset
from mod.OccupancyMap import OccupancyMap
from mod.Grid import Grid
import mod.Models as mod


@pytest.fixture
def dataset() -> DiscreteDirectionalDataset:
    occupancy = mock.MagicMock(spec=OccupancyMap)
    grid = mock.MagicMock(spec=Grid)
    cell1 = mod.DiscreteDirectional(coords=(0, 0), index=(0, 0))
    cell2 = mod.DiscreteDirectional(coords=(0, 1), index=(0, 1))
    occupancy.configure_mock(**{"binary_map": Image.new("L", (2, 2))})
    grid.configure_mock(**{"cells": {(0, 0): cell1, (0, 1): cell2}})
    return DiscreteDirectionalDataset(
        occupancy=occupancy, dynamics=grid, window_size=2
    )
