from unittest import mock

import mod.Models as mod
import pytest
from deepflow.data import DiscreteDirectionalDataset
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from PIL import Image


@pytest.fixture
def occupancy() -> OccupancyMap:
    occupancy = mock.MagicMock(spec=OccupancyMap)
    occupancy.configure_mock(**{"binary_map": Image.new("L", (2, 2))})
    return occupancy


@pytest.fixture
def grid() -> Grid:
    grid = mock.MagicMock(spec=Grid)
    cell1 = mod.DiscreteDirectional(coords=(0, 0), index=(0, 0))
    cell2 = mod.DiscreteDirectional(coords=(0, 1), index=(0, 1))
    grid.configure_mock(**{"cells": {(0, 0): cell1, (0, 1): cell2}})
    return grid


@pytest.fixture
def dataset(occupancy: OccupancyMap, grid: Grid) -> DiscreteDirectionalDataset:
    return DiscreteDirectionalDataset(
        occupancy=occupancy, dynamics=grid, window_size=2
    )
