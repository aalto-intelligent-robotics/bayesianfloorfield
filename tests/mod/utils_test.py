import numpy as np
import pytest

from mod.utils import Direction, RCCoords, XYCoords


def test_xy_coord() -> None:
    c = XYCoords(1.23, 2.11)
    assert c.x == 1.23
    assert c.y == 2.11


def test_rc_coord() -> None:
    c = RCCoords(1, 2)
    assert c.row == 1
    assert c.column == 2


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
def test_directions(
    dir: Direction, rad: float, range: tuple[float, float]
) -> None:
    assert dir.rad == pytest.approx(np.pi * rad)
    assert dir.range[0] == pytest.approx(np.pi * range[0])
    assert dir.range[1] == pytest.approx(np.pi * range[1])


@pytest.mark.parametrize(
    ["dir", "rad", "expected"],
    [
        (Direction.E, -1 / 8, True),
        (Direction.E, 1 / 8, False),
        (Direction.NE, 1 / 8, True),
        (Direction.NE, -1 / 8, False),
        (Direction.SE, -1 / 8, False),
        (Direction.W, 3, True),
    ],
)
def test_directions_contains(
    dir: Direction, rad: float, expected: bool
) -> None:
    assert dir.contains(np.pi * rad) == expected


@pytest.mark.parametrize(
    ["dir", "rad"],
    [
        (Direction.E, 0),
        (Direction.E, 6),
        (Direction.E, -1 / 8),
        (Direction.E, 1 / 16),
        (Direction.NE, 1 / 8),
        (Direction.N, 9 / 16),
        (Direction.NW, 5 / 8),
        (Direction.W, -3),
        (Direction.SW, -3 / 4),
        (Direction.S, 3 / 2),
        (Direction.SE, -35 / 16),
    ],
)
def test_directions_from_rad(dir: Direction, rad: float) -> None:
    assert dir == Direction.from_rad(np.pi * rad)


@pytest.mark.parametrize(
    ["dir", "p1", "p2"],
    [
        (Direction.E, (0, 0), (3, 1)),
        (Direction.NE, (0, 0), (3, 2)),
        (Direction.S, (-1, -3), (-2, -20)),
        (Direction.N, (-2, -20), (-1, -3)),
    ],
)
def test_directions_from_points(
    dir: Direction, p1: tuple[float, float], p2: tuple[float, float]
) -> None:
    assert dir == Direction.from_points(p1, p2)


def test_directions_from_same_point() -> None:
    with pytest.raises(AssertionError):
        Direction.from_points((0.0, 0.0), (0, 0))
