import pytest
from deepflow.nets import DiscreteDirectional
from deepflow.utils import Window, estimate_dynamics
from mod.OccupancyMap import OccupancyMap


def test_window_size():
    w = Window(4)
    assert w.size == 4
    assert w.half_size == 2
    w = Window(7)
    assert w.half_size == 3


def test_window_corners():
    size = 32
    w = Window(size)
    corners = w.corners((2, 3))
    assert corners == (-14, -13, 18, 19)
    assert corners[2] - corners[0] == size
    assert corners[3] - corners[1] == size


def test_window_corners_odd():
    size = 5
    w = Window(size)
    corners = w.corners((2, 3))
    assert corners == (0, 1, 5, 6)
    assert corners[2] - corners[0] == size
    assert corners[3] - corners[1] == size


def test_window_indeces():
    w = Window(5)
    indeces = w.indeces((0, 0))
    assert len(indeces) == 25


def test_window_indeces_bound():
    w = Window(5)
    indeces = w.indeces((0, 0), bounds=[0, 10, 0, 1])
    assert indeces == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}


def test_estimate_dynamics(occupancy: OccupancyMap):
    net = DiscreteDirectional()
    dyn_map = estimate_dynamics(net, occupancy)
    assert dyn_map.shape == (2, 2, 8)
    assert sum(dyn_map[0, 0, :]) == pytest.approx(1)
    assert sum(dyn_map[0, 1, :]) == pytest.approx(1)