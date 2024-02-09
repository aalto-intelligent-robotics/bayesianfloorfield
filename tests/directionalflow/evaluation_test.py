from typing import Literal

import numpy as np
import pytest

from directionalflow.evaluation import (
    append_cell_indeces_to_track,
    pixels_from_track,
    track_likelihood_model,
    track_likelihood_net,
    track_likelihood_net_2,
)
from directionalflow.nets import DiscreteDirectional
from mod.grid import Grid
from mod.occupancy import OccupancyMap


def test_pixels_from_track(occupancy: OccupancyMap) -> None:
    track = np.array(
        [[0.2, 3.7, np.pi], [1.5, 3.2, np.pi / 2], [1.2, 2.1, np.pi * 2]]
    ).T
    expected = np.array(
        [[0.3, 0.2, np.pi], [0.8, 1.5, np.pi / 2], [1.9, 1.2, np.pi * 2]]
    ).T
    assert pixels_from_track(track, occupancy) == pytest.approx(expected)


def test_append_cell_indeces_to_track(grid: Grid) -> None:
    track = np.array(
        [
            [0.3, 2.2, np.pi / 4],
            [0.5, 2.8, np.pi / 2],
            [1.8, 2.5, np.pi / 8],
            [1.8, 2.8, np.pi * 2],
            [1.5, 3.1, np.pi / 3],
            [1.7, 3.4, np.pi / 4],
            [1.9, 3.4, np.pi / 6],
        ]
    ).T
    expected = np.array(
        [
            [0.3, 2.2, np.pi / 4, 0, 0],
            [0.5, 2.8, np.pi / 2, 0, 0],
            [1.8, 2.5, np.pi / 8, 0, 1],
            [1.8, 2.8, np.pi * 2, 0, 1],
            [1.5, 3.1, np.pi / 3, 1, 1],
            [1.7, 3.4, np.pi / 4, 1, 1],
            [1.9, 3.4, np.pi / 6, 1, 1],
        ]
    ).T
    assert append_cell_indeces_to_track(track, grid) == pytest.approx(expected)


def test_track_likelihood_net(occupancy: OccupancyMap) -> None:
    pixels = np.array(
        [
            [0.4, 0.5, np.pi],
            [0.8, 1.5, np.pi / 2],
            [1.7, 1.3, np.pi * 2],
        ]
    ).T
    net = DiscreteDirectional(window_size=2)
    like, matches = track_likelihood_net(
        pixels, occupancy, window_size=2, scale=1, net=net
    )
    assert matches == 3
    assert 0 <= like / matches <= 1


def test_track_likelihood_net_2() -> None:
    pixels = np.array(
        [
            [0.4, 0.5, np.pi],
            [0.8, 1.5, np.pi / 2],
            [0.7, 1.3, np.pi * 2],
        ]
    ).T
    dynamics = np.array([[[1 / 8] * 8, [1 / 4] * 2 + [1 / 12] * 6]])
    like, matches = track_likelihood_net_2(pixels, dynamics)
    assert matches == 3
    assert like == pytest.approx(1 / 8 + 1 / 4 + 1 / 12)


@pytest.mark.parametrize(
    ["missing_strategy", "expected_like", "expected_matches"],
    [("skip", 1.5, 2), ("uniform", 1.625, 3), ("zero", 1.5, 3)],
)
def test_track_likelihood_model(
    grid: Grid,
    missing_strategy: Literal["skip", "uniform", "zero"],
    expected_like: float,
    expected_matches: int,
) -> None:
    track = np.array(
        [
            [0.4, 2.5, 0, 0, 0],
            [1.5, 2.8, np.pi / 4, 0, 1],
            [1.7, 3.3, np.pi, 1, 1],
        ]
    ).T
    like, matches, missing = track_likelihood_model(
        track, grid, missing_cells=missing_strategy
    )
    assert like == pytest.approx(expected_like)
    assert matches == expected_matches
    assert missing == 1
