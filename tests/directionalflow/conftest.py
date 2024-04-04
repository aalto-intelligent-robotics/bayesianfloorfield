from unittest import mock

import numpy as np
import pandas as pd
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
            "origin": XYCoords(0, 2),
        }
    )
    return occupancy


@pytest.fixture
def grid() -> Grid:
    cell1 = mod.BayesianDiscreteDirectional(
        coords=XYCoords(0, 2), index=RCCoords(0, 0), resolution=1
    )
    cell2 = mod.BayesianDiscreteDirectional(
        coords=XYCoords(1, 2), index=RCCoords(0, 1), resolution=1
    )
    cell1.bins = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    cell2.bins = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    columns = [
        "time",
        "person_id",
        "x",
        "y",
        "motion_angle",
        "row",
        "col",
        "bin",
    ]
    cell1.data = pd.DataFrame(
        [
            [0, 0, 0.4, 2.5, 0, 0, 0, 0],
            [0.1, 0, 0.6, 2.2, np.pi * 5 / 4, 0, 0, 5],
        ],
        columns=columns,
    )
    cell2.data = pd.DataFrame(
        [[0.2, 1, 1.5, 2.8, np.pi / 4, 0, 1, 1]], columns=columns
    )
    return Grid(
        resolution=1,
        origin=XYCoords(0, 2),
        model=mod.BayesianDiscreteDirectional,
        cells={RCCoords(0, 0): cell1, RCCoords(0, 1): cell2},
        total_count=3,
    )


@pytest.fixture
def tracks() -> list[np.ndarray]:
    return [
        np.array(
            [
                [0.4, 2.5, 0],  # (0, 0) E -> P=0.5
                [1.5, 2.8, np.pi / 4],  # (0, 1) NE -> P=1.0
                [1.7, 3.3, np.pi],  # (1, 1) W -> missing (P=0.125)
            ]
        ).T,
        np.array(
            [
                [0.4, 2.5, 0],
                [1.5, 2.8, np.pi / 4],
                [1.7, 3.3, np.pi],
            ]
        ).T,
    ]


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
