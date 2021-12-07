from pathlib import Path

import numpy as np
import pytest
from deepflow.nets import ConditionalDiscreteDirectional, DiscreteDirectional
from deepflow.utils import Trainer, Window, estimate_dynamics
from mod.OccupancyMap import OccupancyMap
from torch import device


def test_window_size():
    w = Window(4)
    assert w.size == 4
    assert w.half_size == 2
    assert w.center == (2, 2)
    assert w.pad_amount == (2, 1)
    w = Window(7)
    assert w.half_size == 3
    assert w.center == (3, 3)
    assert w.pad_amount == (3, 3)


def test_window_corners():
    size = 32
    w = Window(size)
    corners = w.corners((2, 3))
    assert corners[2] - corners[0] == size
    assert corners[3] - corners[1] == size
    assert corners == (-13, -14, 19, 18)


def test_window_corners_odd():
    size = 5
    w = Window(size)
    corners = w.corners((2, 3))
    assert corners[2] - corners[0] == size
    assert corners[3] - corners[1] == size
    assert corners == (1, 0, 6, 5)


def test_window_corners_bounds():
    size = 16
    w = Window(size)
    corners = w.corners((1, 3), (0, 10, 0, 2))
    assert corners == (0, 0, 2, 9)


def test_window_corners_outside_bounds():
    size = 4
    w = Window(size)
    corners = w.corners((2, 3), (0, 2, 0, 2))
    assert corners == (1, 0, 2, 2)


def test_window_indeces():
    w = Window(2)
    indeces = w.indeces((0, 2))
    assert len(indeces) == 4
    assert indeces == {(-1, 1), (-1, 2), (0, 1), (0, 2)}


def test_window_indeces_bounds():
    w = Window(5)
    indeces = w.indeces((0, 1), bounds=[0, 10, 0, 2])
    assert indeces == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}


def test_window_unit():
    w = Window(1)
    assert w.size == 1
    assert w.half_size == 0
    assert w.corners((0, 1)) == (1, 0, 2, 1)
    assert w.indeces((0, 1)) == {(0, 1)}


def test_window_center():
    w = Window(6)
    a = np.zeros((10, 10))
    a[5, 4] = 1
    l, t, r, b = w.corners((5, 4))
    crop = a[t:b, l:r]
    assert crop[w.center[0], w.center[1]] == 1


def test_estimate_discretedirectional(occupancy: OccupancyMap):
    net = DiscreteDirectional(window_size=2)
    dyn_map = estimate_dynamics(net, occupancy, device=device("cpu"))
    assert dyn_map.shape == (2, 2, 8)
    assert sum(dyn_map[0, 0, :]) == pytest.approx(1)
    assert sum(dyn_map[0, 1, :]) == pytest.approx(1)


def test_estimate_conditionaldirectional(occupancy: OccupancyMap):
    net = ConditionalDiscreteDirectional(window_size=2)
    dyn_map = estimate_dynamics(net, occupancy, device=device("cpu"))
    assert dyn_map.shape == (2, 2, 64)
    assert sum(dyn_map[0, 0, :]) == pytest.approx(1)
    assert sum(dyn_map[0, 1, :]) == pytest.approx(1)


def test_trainer(trainer: Trainer):
    trainer.train(epochs=1)
    assert trainer.train_epochs == 1


def test_save_load_trainer(trainer: Trainer, tmp_path: Path):
    trainer.train(epochs=2)
    path = tmp_path / "trainer_state.pth"
    trainer.save(path.as_posix())
    trainer.train(epochs=1)
    assert trainer.train_epochs == 3
    trainer.load(path.as_posix())
    assert trainer.train_epochs == 2
