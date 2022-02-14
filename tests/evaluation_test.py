import numpy as np
import pytest
from deepflow.evaluation import pixels2grid, track2pixels, track_likelihood
from mod.OccupancyMap import OccupancyMap
from deepflow.nets import DiscreteDirectional


def test_track2pixels(occupancy: OccupancyMap):
    track = np.array([[0.2, 3.7], [1.5, 3.2], [1.2, 2.1]]).T
    expected = np.array([[0.3, 0.2], [0.8, 1.5], [1.9, 1.2]]).T
    assert track2pixels(track, occupancy) == pytest.approx(expected)


def test_pixels2grid():
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


def test_track_likelihood(occupancy: OccupancyMap):
    track = np.array([[0.4, 0.5, 0, 0], [0.8, 1.5, 0, 1], [1.7, 1.3, 1, 1]]).T
    net = DiscreteDirectional(window_size=2)
    like = track_likelihood(track, occupancy, window_size=2, scale=1, net=net)
    assert like >= 0 and like <= 1
