from unittest import mock

import pytest
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader

import mod.models as mod
from directionalflow.data import DiscreteDirectionalDataset
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Trainer
from mod.grid import Grid
from mod.occupancy import OccupancyMap
from mod.utils import RCCoords, XYCoords


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
    cell1 = mod.DiscreteDirectional(
        coords=XYCoords(0, 0), index=RCCoords(0, 0), resolution=1
    )
    cell2 = mod.DiscreteDirectional(
        coords=XYCoords(0, 1000), index=RCCoords(0, 1), resolution=1
    )
    cell1.bins = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    cell2.bins = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    grid.configure_mock(
        **{
            "resolution": 1000,
            "cells": {RCCoords(0, 0): cell1, RCCoords(0, 1): cell2},
            "origin": [0, 2000, 0],
        }
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
