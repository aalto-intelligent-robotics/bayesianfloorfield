from pathlib import Path

import numpy as np
import pytest
from deepflow.nets import ConditionalDiscreteDirectional, DiscreteDirectional
from deepflow.utils import (
    Direction,
    Trainer,
    Window,
    estimate_dynamics,
    random_input,
    scale_quivers,
    switch_directions,
)
from mod.OccupancyMap import OccupancyMap
from torch import device, float32


@pytest.mark.parametrize(
    ["dir", "rad", "range"],
    [
        (Direction.E, 0, (15 / 8, 1 / 8)),
        (Direction.NE, 1 / 4, (1 / 8, 3 / 8)),
        (Direction.N, 1 / 2, (3 / 8, 5 / 8)),
        (Direction.NW, 3 / 4, (5 / 8, 7 / 8)),
        (Direction.W, 1, (7 / 8, 9 / 8)),
        (Direction.SW, 5 / 4, (9 / 8, 11 / 8)),
        (Direction.S, 3 / 2, (11 / 8, 13 / 8)),
        (Direction.SE, 7 / 4, (13 / 8, 15 / 8)),
    ],
)
def test_directions(dir: Direction, rad: float, range: tuple[float, float]):
    assert dir.rad() == pytest.approx(np.pi * rad)
    assert dir.range()[0] == pytest.approx(np.pi * range[0])
    assert dir.range()[1] == pytest.approx(np.pi * range[1])


def test_scale_quivers():
    outputs = np.zeros((3, 8))
    outputs[1, :] = [1 / 8] * 8
    outputs[2, :] = [2 / 8, 4 / 8, 1 / 8, 1 / 8, 0, 0, 0, 0]
    scaled = scale_quivers(outputs)
    assert outputs.shape == scaled.shape
    assert (scaled[0, :] == np.zeros(8)).all()
    assert (scaled[1, :] == np.ones(8)).all()
    assert (scaled[2, :] == [1 / 2, 1, 1 / 4, 1 / 4, 0, 0, 0, 0]).all()


@pytest.mark.parametrize(
    ["p_occupied", "expected"],
    [(0, np.zeros((1, 1, 16, 16))), (1, np.ones((1, 1, 16, 16)))],
)
def test_random_input(p_occupied: float, expected: np.ndarray):
    expected[0, 0, 8, 8] = 0  # center should always be zero
    tensor = random_input(size=16, p_occupied=p_occupied)
    assert tensor.dtype == float32
    assert (tensor.numpy() == expected).all()


@pytest.mark.parametrize(
    ["dirA", "dirB", "expected"],
    [
        (Direction.N, Direction.N, [1, 2, 3, 4, 5, 6, 7, 8]),
        (Direction.E, Direction.NE, [2, 1, 3, 4, 5, 6, 7, 8]),
        (Direction.SE, Direction.SW, [1, 2, 3, 4, 5, 8, 7, 6]),
        (Direction.N, Direction.S, [1, 2, 7, 4, 5, 6, 3, 8]),
    ],
)
def test_switch_directions(dirA: Direction, dirB: Direction, expected: list):
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    assert (switch_directions(a, dirA, dirB) == np.array(expected)).all()


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
