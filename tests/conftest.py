from unittest import mock

import mod.Models as mod
import pytest
from deepflow.data import DiscreteDirectionalDataset
from deepflow.nets import DiscreteDirectional
from deepflow.utils import Trainer
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader


@pytest.fixture
def occupancy() -> OccupancyMap:
    occupancy = mock.MagicMock(spec=OccupancyMap)
    map = Image.new("L", (2, 2))
    map.putpixel((1, 0), 255)
    occupancy.configure_mock(
        **{
            "resolution": 1,
            "map": map,
            "binary_map": map,
            "origin": [0, 2, 0],
        }
    )
    return occupancy


@pytest.fixture
def grid() -> Grid:
    grid = mock.MagicMock(spec=Grid)
    cell1 = mod.DiscreteDirectional(coords=(0, 0), index=(0, 0))
    cell2 = mod.DiscreteDirectional(coords=(0, 1), index=(0, 1))
    cell1.bins[cell1.directions[0]]["probability"] = 1.0 / 2
    cell1.bins[cell1.directions[5]]["probability"] = 1.0 / 2
    cell2.bins[cell1.directions[1]]["probability"] = 1.0
    grid.configure_mock(
        **{"resolution": 1000, "cells": {(0, 0): cell1, (0, 1): cell2}}
    )
    return grid


@pytest.fixture
def dataset(occupancy: OccupancyMap, grid: Grid) -> DiscreteDirectionalDataset:
    return DiscreteDirectionalDataset(
        occupancy=occupancy, dynamics=grid, window_size=2
    )


@pytest.fixture
def trainer(dataset: DiscreteDirectionalDataset) -> Trainer:
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    net = DiscreteDirectional(window_size=2)
    return Trainer(
        net=net,
        trainloader=dataloader,
        valloader=dataloader,
        optimizer=SGD(net.parameters(), lr=0.001, momentum=0.9),
    )
