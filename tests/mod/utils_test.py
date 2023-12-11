import numpy as np
import pytest

from mod.utils import (
    Direction,
    RC_from_TDRC,
    RC_from_XY,
    RCCoords,
    TDRC_from_RC,
    TDRC_from_XY,
    TDRCCoords,
    XY_from_RC,
    XY_from_TDRC,
    XYCoords,
)


def test_xy_coord() -> None:
    c = XYCoords(1.23, 2.11)
    assert c.x == 1.23
    assert c.y == 2.11


def test_rc_coord() -> None:
    c = RCCoords(1, 2)
    assert c.row == 1
    assert c.column == 2


def test_tdrc_coord() -> None:
    c = TDRCCoords(1, 2)
    assert c.row == 1
    assert c.column == 2


@pytest.mark.parametrize(
    ["xy", "rc"],
    [(XYCoords(3, 1), RCCoords(4, 4)), (XYCoords(-2, 2), RCCoords(5, 2))],
)
def test_RC_from_XY(xy: XYCoords, rc: RCCoords) -> None:
    assert RC_from_XY(xy, XYCoords(-6, -8), resolution=2) == rc


def test_RC_from_XY_raises() -> None:
    origin = XYCoords(-6, -8)

    with pytest.raises(ValueError):
        RC_from_XY(XYCoords(-7, 0), origin, resolution=2)
    with pytest.raises(ValueError):
        RC_from_XY(XYCoords(0, -10), origin, resolution=2)

    assert RC_from_XY(origin, origin, resolution=2) == RCCoords(0, 0)


@pytest.mark.parametrize(
    ["xy", "rc"],
    [(XYCoords(2, 0), RCCoords(4, 4)), (XYCoords(-2, 2), RCCoords(5, 2))],
)
def test_XY_from_RC(xy: XYCoords, rc: RCCoords) -> None:
    assert XY_from_RC(rc, XYCoords(-6, -8), resolution=2) == xy


@pytest.mark.parametrize(
    ["xy", "rc"],
    [(XYCoords(3, 1), TDRCCoords(1, 4)), (XYCoords(-2, 2), TDRCCoords(0, 2))],
)
def test_TDRC_from_XY(xy: XYCoords, rc: TDRCCoords) -> None:
    assert TDRC_from_XY(xy, XYCoords(-6, -8), resolution=2, num_rows=6) == rc


def test_TDRC_from_XY_raises() -> None:
    origin = XYCoords(-6, -8)

    with pytest.raises(ValueError):
        TDRC_from_XY(XYCoords(-7, 0), origin, resolution=2, num_rows=3)
    with pytest.raises(ValueError):
        TDRC_from_XY(XYCoords(0, -10), origin, resolution=2, num_rows=3)

    assert TDRC_from_XY(origin, origin, 2, num_rows=3) == TDRCCoords(2, 0)


@pytest.mark.parametrize(
    ["rc", "n", "tdrc"],
    [
        (RCCoords(0, 1), 1, TDRCCoords(0, 1)),
        (RCCoords(1, 0), 4, TDRCCoords(2, 0)),
        (RCCoords(32, 122), 100, TDRCCoords(67, 122)),
    ],
)
def test_TDRC_RC_conversion(rc: RCCoords, n: int, tdrc: TDRCCoords) -> None:
    assert TDRC_from_RC(rc, num_rows=n) == tdrc
    assert RC_from_TDRC(tdrc, num_rows=n) == rc


def test_TDRC_RC_conversion_raises() -> None:
    with pytest.raises(ValueError):
        TDRC_from_RC(RCCoords(6, 40), num_rows=6)
    with pytest.raises(ValueError):
        RC_from_TDRC(TDRCCoords(1, 2), num_rows=1)


@pytest.mark.parametrize(
    ["xy", "rc"],
    [(XYCoords(2, 0), TDRCCoords(1, 4)), (XYCoords(-2, 2), TDRCCoords(0, 2))],
)
def test_XY_from_TDRC(xy: XYCoords, rc: TDRCCoords) -> None:
    assert XY_from_TDRC(rc, XYCoords(-6, -8), resolution=2, num_rows=6) == xy


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
