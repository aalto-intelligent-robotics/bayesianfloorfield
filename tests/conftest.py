from unittest import mock
from deepflow.nets import DiscreteDirectional

import mod.Models as mod
import pytest
from deepflow.data import DiscreteDirectionalDataset
from deepflow.utils import Trainer
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import SGD


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


@pytest.fixture
def trainer(dataset: DiscreteDirectionalDataset) -> Trainer:
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    net = DiscreteDirectional()
    return Trainer(
        net=net,
        trainloader=dataloader,
        valloader=dataloader,
        optimizer=SGD(net.parameters(), lr=0.001, momentum=0.9),
    )
