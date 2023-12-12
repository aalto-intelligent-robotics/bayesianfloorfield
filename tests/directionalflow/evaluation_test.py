from typing import Literal

import numpy as np
import pytest

from directionalflow.evaluation import (
    pixels2grid,
    pixels2grid_complete,
    track2pixels,
    track_likelihood_model,
    track_likelihood_net,
)
from directionalflow.nets import DiscreteDirectional
from mod.grid import Grid
from mod.occupancy import OccupancyMap


def test_track2pixels(occupancy: OccupancyMap) -> None:
    track = np.array(
        [[0.2, 3.7, np.pi], [1.5, 3.2, np.pi / 2], [1.2, 2.1, np.pi * 2]]
    ).T
    expected = np.array(
        [[0.3, 0.2, np.pi], [0.8, 1.5, np.pi / 2], [1.9, 1.2, np.pi * 2]]
    ).T
    assert track2pixels(track, occupancy) == pytest.approx(expected)


def test_pixels2grid() -> None:
    pixels = np.array(
        [
            [0.3, 0.2],
            [0.5, 0.8],
            [0.8, 1.5],
            [0.8, 1.5],
            [1.5, 1.1],
            [1.7, 1.4],
            [1.9, 1.4],
        ]
    ).T
    expected = np.array(
        [[0.4, 0.5, 0, 0], [0.8, 1.5, 0, 1], [1.7, 1.3, 1, 1]]
    ).T
    assert pixels2grid(pixels, 1, 1) == pytest.approx(expected)


def test_pixels2grid_complete() -> None:
    pixels = np.array(
        [
            [0.3, 0.2, np.pi / 4],
            [0.5, 0.8, np.pi / 2],
            [0.8, 1.5, np.pi / 8],
            [0.8, 1.5, np.pi * 2],
            [1.5, 1.1, np.pi / 3],
            [1.7, 1.4, np.pi / 4],
            [1.9, 1.4, np.pi / 6],
        ]
    ).T
    expected = np.array(
        [
            [0.3, 0.2, np.pi / 4, 0, 0],
            [0.5, 0.8, np.pi / 2, 0, 0],
            [0.8, 1.5, np.pi / 8, 0, 1],
            [0.8, 1.5, np.pi * 2, 0, 1],
            [1.5, 1.1, np.pi / 3, 1, 1],
            [1.7, 1.4, np.pi / 4, 1, 1],
            [1.9, 1.4, np.pi / 6, 1, 1],
        ]
    ).T
    assert pixels2grid_complete(pixels, 1, 1) == pytest.approx(expected)


def test_track_likelihood_net(occupancy: OccupancyMap) -> None:
    track = np.array(
        [
            [0.4, 0.5, np.pi, 0, 0],
            [0.8, 1.5, np.pi / 2, 0, 1],
            [1.7, 1.3, np.pi * 2, 1, 1],
        ]
    ).T
    net = DiscreteDirectional(window_size=2)
    like, matches = track_likelihood_net(
        track, occupancy, window_size=2, scale=1, net=net
    )
    assert matches == 3
    assert 0 <= like / matches <= 1


@pytest.mark.parametrize(
    ["missing_strategy", "expected_like", "expected_matches"],
    [("skip", 1.5, 2), ("uniform", 1.625, 3), ("zero", 1.5, 3)],
)
def test_track_likelihood_model(
    occupancy: OccupancyMap,
    grid: Grid,
    missing_strategy: Literal["skip", "uniform", "zero"],
    expected_like: float,
    expected_matches: int,
) -> None:
    track = np.array(
        [
            [0.4, 0.5, np.pi, 0, 0],
            [1.5, 0.8, 0, 1, 0],
            [1.7, 1.3, np.pi / 4, 1, 1],
        ]
    ).T
    like, matches, missing = track_likelihood_model(
        track, occupancy, grid, missing_cells=missing_strategy
    )
    assert like == pytest.approx(expected_like)
    assert matches == expected_matches
    assert missing == 1
